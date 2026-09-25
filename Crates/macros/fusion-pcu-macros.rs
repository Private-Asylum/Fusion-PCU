use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{
    format_ident,
    quote,
    ToTokens,
};
use syn::parse::{
    Parse,
    ParseStream,
};
use syn::spanned::Spanned;
use syn::{
    BinOp,
    Error,
    Expr,
    ExprAssign,
    ExprBinary,
    ExprField,
    ExprIndex,
    ExprLit,
    FnArg,
    GenericParam,
    Ident,
    ItemFn,
    Lit,
    LitFloat,
    Pat,
    Path,
    ReturnType,
    Stmt,
    Token,
    Type,
    parse_macro_input,
};

struct PcuDispatchArgs {
    kernel_id: u32,
    invocations: Expr,
    crate_path: Path,
}

impl Parse for PcuDispatchArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self, Error> {
        let mut kernel_id = None;
        let mut invocations = None;
        let mut crate_path = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            let _: Token![=] = input.parse()?;
            match key.to_string().as_str() {
                "kernel_id" => {
                    let value: syn::LitInt = input.parse()?;
                    let parsed = value.base10_parse::<u32>()?;
                    if kernel_id.is_some() {
                        return Err(Error::new(key.span(), "duplicate `kernel_id` argument"));
                    }
                    kernel_id = Some(parsed);
                }
                "invocations" => {
                    if invocations.is_some() {
                        return Err(Error::new(key.span(), "duplicate `invocations` argument"));
                    }
                    invocations = Some(input.parse::<Expr>()?);
                }
                "crate_path" => {
                    if crate_path.is_some() {
                        return Err(Error::new(key.span(), "duplicate `crate_path` argument"));
                    }
                    crate_path = Some(input.parse::<Path>()?);
                }
                _ => {
                    return Err(Error::new(
                        key.span(),
                        "#[pcu_dispatch] supports `kernel_id = <u32>`, `invocations = <const expression>`, and `crate_path = <path>`",
                    ));
                }
            }

            if input.is_empty() {
                break;
            }
            let _: Token![,] = input.parse()?;
        }

        let invocations = invocations.ok_or_else(|| {
            Error::new(
                input.span(),
                "#[pcu_dispatch] requires `invocations = <const expression>`",
            )
        })?;
        Ok(Self {
            kernel_id: kernel_id.map_or(1, core::convert::identity),
            invocations,
            crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::fusion_pcu)),
        })
    }
}

#[derive(Clone, Copy)]
enum BindingAccess {
    ReadOnly,
    ReadWrite,
}

struct BindingSpec {
    ident: Ident,
    access: BindingAccess,
    binding: u32,
}

struct ExprEmitter<'a> {
    bindings: &'a [BindingSpec],
    invocation_ident: &'a Ident,
    crate_path: &'a Path,
    grid_stride: bool,
    next_value: u16,
    ops: Vec<TokenStream2>,
}

impl<'a> ExprEmitter<'a> {
    const fn new(
        bindings: &'a [BindingSpec],
        invocation_ident: &'a Ident,
        crate_path: &'a Path,
        grid_stride: bool,
    ) -> Self {
        Self {
            bindings,
            invocation_ident,
            crate_path,
            grid_stride,
            next_value: 1,
            ops: Vec::new(),
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<u16, Error> {
        match expr {
            Expr::Binary(binary) => self.emit_binary(binary),
            Expr::Index(index) => self.emit_index_load(index),
            Expr::Lit(lit) => self.emit_lit(lit),
            Expr::Paren(paren) => self.emit_expr(&paren.expr),
            _ => Err(Error::new(
                unsupported_expression_span(expr),
                "unsupported PCU expression; supported subset is binding[index], f32 literals, parentheses, and + - * /",
            )),
        }
    }

    fn emit_binary(&mut self, binary: &ExprBinary) -> Result<u16, Error> {
        let lhs = self.emit_expr(&binary.left)?;
        let rhs = self.emit_expr(&binary.right)?;
        let result = self.alloc_value(binary.span())?;
        let pcu = self.crate_path;
        let op = match &binary.op {
            BinOp::Add(_) => quote! { #pcu::PcuDispatchAluOp::Add },
            BinOp::Sub(_) => quote! { #pcu::PcuDispatchAluOp::Sub },
            BinOp::Mul(_) => quote! { #pcu::PcuDispatchAluOp::Mul },
            BinOp::Div(_) => quote! { #pcu::PcuDispatchAluOp::Div },
            _ => {
                return Err(Error::new(
                    binary.op.span(),
                    "unsupported PCU binary operator; supported operators are + - * /",
                ));
            }
        };
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::Alu {
                result: #pcu::PcuDispatchValueId(#result),
                op: #op,
                lhs: #pcu::PcuDispatchValueId(#lhs),
                rhs: #pcu::PcuDispatchValueId(#rhs),
            }
        });
        Ok(result)
    }

    fn emit_index_load(&mut self, index: &ExprIndex) -> Result<u16, Error> {
        validate_invocation_index(&index.index, self.invocation_ident)?;
        let Some(binding_ident) = expr_ident(&index.expr) else {
            return Err(Error::new(
                index.expr.span(),
                "PCU binding load must use `binding[invocation]`",
            ));
        };
        let slot = self
            .binding(binding_ident, BindingAccess::ReadOnly)?
            .binding;
        let result = self.alloc_value(index.span())?;
        let pcu = self.crate_path;
        let dispatch_index = self.index();
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::BindingLoad {
                result: #pcu::PcuDispatchValueId(#result),
                binding: #pcu::PcuBindingRef::new(0, #slot),
                index: #dispatch_index,
            }
        });
        Ok(result)
    }

    fn emit_lit(&mut self, lit: &ExprLit) -> Result<u16, Error> {
        let Lit::Float(float) = &lit.lit else {
            return Err(Error::new(
                lit.span(),
                "PCU constants support f32 float literals in this first cut",
            ));
        };
        let bits = parse_f32_bits(float)?;
        let result = self.alloc_value(lit.span())?;
        let pcu = self.crate_path;
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::Constant {
                result: #pcu::PcuDispatchValueId(#result),
                value: #pcu::PcuParameterValue::from_f32_bits(#bits),
            }
        });
        Ok(result)
    }

    fn index(&self) -> TokenStream2 {
        let pcu = self.crate_path;
        if self.grid_stride {
            quote! { #pcu::PcuDispatchIndex::GridStrideId }
        } else {
            quote! { #pcu::PcuDispatchIndex::InvocationId }
        }
    }

    fn binding(&self, ident: &Ident, required: BindingAccess) -> Result<&BindingSpec, Error> {
        let Some(binding) = self.bindings.iter().find(|binding| binding.ident == *ident) else {
            return Err(Error::new(ident.span(), "unknown PCU binding"));
        };
        if !matches!(
            (binding.access, required),
            (
                BindingAccess::ReadOnly | BindingAccess::ReadWrite,
                BindingAccess::ReadOnly
            ) | (BindingAccess::ReadWrite, BindingAccess::ReadWrite)
        ) {
            return Err(Error::new(
                ident.span(),
                "PCU binding access does not match the expression context",
            ));
        }
        Ok(binding)
    }

    fn alloc_value(&mut self, span: proc_macro2::Span) -> Result<u16, Error> {
        let value = self.next_value;
        self.next_value = self.next_value.checked_add(1).ok_or_else(|| {
            Error::new(span, "too many virtual PCU values for this dispatch macro")
        })?;
        Ok(value)
    }
}

#[proc_macro_attribute]
pub fn pcu_dispatch(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as PcuDispatchArgs);
    let function = parse_macro_input!(item as ItemFn);
    match expand_pcu_dispatch(args, &function) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

/// Define a bounded PCU kernel with Rust-style invocation-count syntax.
///
/// `#[pcu(invocations = R * C)]` is the concise spelling of [`pcu_dispatch`]. The attribute
/// accepts the same arguments and currently lowers the same deliberately bounded source subset.
#[proc_macro_attribute]
pub fn pcu(attr: TokenStream, item: TokenStream) -> TokenStream {
    pcu_dispatch(attr, item)
}

fn build_body_operations(
    data_ops: &[TokenStream2],
    loop_extent: Option<&Expr>,
    const_generics: &[Ident],
    crate_path: &Path,
) -> Result<(TokenStream2, usize), Error> {
    if let Some(extent) = loop_extent {
        let extent = lower_invocation_expr(extent, const_generics)?;
        let body_len = data_ops.len();
        let loop_body = data_ops
            .iter()
            .map(|op| quote! { #crate_path::PcuDispatchOp::Data(#op) })
            .collect::<Vec<_>>();
        let operations = quote! {
            const __PCU_GRID_STRIDE_BODY: [#crate_path::PcuDispatchOp<'static>; #body_len] = [
                #(#loop_body),*
            ];
            let builder = builder.with_op(#crate_path::PcuDispatchOp::GridStrideLoop {
                extent: const {
                    let extent: usize = #extent;
                    assert!(extent != 0, "PCU grid-stride extent must be nonzero");
                    assert!(extent <= u32::MAX as usize, "PCU grid-stride extent exceeds u32");
                    extent as u32
                },
                body: &__PCU_GRID_STRIDE_BODY,
            })?;
        };
        Ok((operations, 2))
    } else {
        Ok((
            quote! { #(let builder = builder.with_data_op(#data_ops)?;)* },
            data_ops.len() + 1,
        ))
    }
}

fn expand_pcu_dispatch(args: PcuDispatchArgs, function: &ItemFn) -> Result<TokenStream2, Error> {
    let vis = function.vis.clone();
    let function_ident = function.sig.ident.clone();
    let bindings_ident = format_ident!("{}_bindings", function_ident);
    let crate_path = args.crate_path;
    let const_generics = validate_const_generics(function)?;
    let invocation_expr = lower_invocation_expr(&args.invocations, &const_generics)?;
    let generated_generics = if function.sig.generics.params.is_empty() {
        quote! { <'a> }
    } else {
        let params = &function.sig.generics.params;
        quote! { <'a, #params> }
    };
    let binding_specs = parse_bindings(&function.sig.inputs)?;
    let body = validate_body(function)?;
    let (invocation_ident, assignment, loop_extent) = match &body {
        ValidatedBody::Indexed {
            invocation,
            assignment,
        } => (invocation, *assignment, None),
        ValidatedBody::GridStride {
            invocation,
            assignment,
            extent,
        } => (invocation, *assignment, Some(*extent)),
    };
    let output_binding = validate_assignment_target(assignment, &binding_specs, invocation_ident)?;
    let mut emitter = ExprEmitter::new(
        &binding_specs,
        invocation_ident,
        &crate_path,
        loop_extent.is_some(),
    );
    let result_value = emitter.emit_expr(&assignment.right)?;
    let output_slot = output_binding.binding;
    let mut data_ops = emitter.ops;
    let store_index = if loop_extent.is_some() {
        quote! { GridStrideId }
    } else {
        quote! { InvocationId }
    };
    let pcu = &crate_path;
    data_ops.push(quote! {
        #pcu::PcuDispatchDataOp::BindingStore {
            binding: #pcu::PcuBindingRef::new(0, #output_slot),
            index: #pcu::PcuDispatchIndex::#store_index,
            value: #pcu::PcuDispatchValueId(#result_value),
        }
    });

    let (operations, op_count) =
        build_body_operations(&data_ops, loop_extent, &const_generics, &crate_path)?;

    let binding_items = binding_specs
        .iter()
        .map(|binding| binding_tokens(binding, pcu))
        .collect::<Vec<_>>();
    let binding_count = binding_items.len();
    let kernel_id = args.kernel_id;
    Ok(quote! {
        #vis const fn #bindings_ident() -> [#pcu::PcuBinding<'static>; #binding_count] {
            [#(#binding_items),*]
        }

        #vis fn #function_ident #generated_generics(
            bindings: &'a [#pcu::PcuBinding<'a>],
        ) -> ::core::result::Result<
            #pcu::model::PcuDispatchKernelBuilder<'a, #op_count>,
            #pcu::PcuError,
        > {
            let invocations: u32 = const {
                let count: usize = #invocation_expr;
                assert!(count != 0, "PCU invocation count must be nonzero");
                assert!(count <= u32::MAX as usize, "PCU invocation count exceeds u32");
                count as u32
            };
            let builder = #pcu::model::PcuDispatchKernelBuilder::<#op_count>::new(
                #kernel_id,
                "main",
                [invocations, 1, 1],
            )
            .with_bindings(bindings);
            #operations
            builder.with_control_op(#pcu::PcuDispatchControlOp::Return)
        }
    })
}

fn validate_const_generics(function: &ItemFn) -> Result<Vec<Ident>, Error> {
    if function.sig.asyncness.is_some()
        || function.sig.constness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || !matches!(function.sig.output, ReturnType::Default)
        || function.sig.generics.where_clause.is_some()
    {
        return Err(Error::new(
            function.sig.span(),
            "PCU dispatch source must be a plain function without a return type or where clause",
        ));
    }
    let mut names = Vec::new();
    for parameter in &function.sig.generics.params {
        let GenericParam::Const(parameter) = parameter else {
            return Err(Error::new(
                parameter.span(),
                "PCU dispatch does not yet support type generics: binding metadata and emitted constants currently use f32, so a `T: PcuScalar` bound alone would not make the lowered operations type-safe; use `const NAME: usize` generics for invocation shapes",
            ));
        };
        let Type::Path(ty) = &parameter.ty else {
            return Err(Error::new(
                parameter.ty.span(),
                "PCU invocation const generics must have type `usize`",
            ));
        };
        if !ty.path.is_ident("usize") || parameter.default.is_some() {
            return Err(Error::new(
                parameter.ty.span(),
                "PCU invocation const generics must have type `usize` and no default",
            ));
        }
        names.push(parameter.ident.clone());
    }
    Ok(names)
}

fn lower_invocation_expr(expr: &Expr, const_generics: &[Ident]) -> Result<TokenStream2, Error> {
    match expr {
        Expr::Lit(ExprLit {
            lit: Lit::Int(value),
            ..
        }) => {
            let parsed = value.base10_parse::<usize>()?;
            Ok(quote! { #parsed })
        }
        Expr::Path(path) if path.qself.is_none() && path.path.get_ident().is_some() => {
            let ident = path.path.get_ident().expect("guard checked the identifier");
            if !const_generics.contains(ident) {
                return Err(Error::new(
                    ident.span(),
                    "PCU invocation identifier must be a `const NAME: usize` generic",
                ));
            }
            Ok(quote! { #ident })
        }
        Expr::Paren(paren) => lower_invocation_expr(&paren.expr, const_generics),
        Expr::Group(group) => lower_invocation_expr(&group.expr, const_generics),
        Expr::Binary(binary) => {
            let lhs = lower_invocation_expr(&binary.left, const_generics)?;
            let rhs = lower_invocation_expr(&binary.right, const_generics)?;
            let (method, failure) = match binary.op {
                BinOp::Add(_) => (
                    format_ident!("checked_add"),
                    "PCU invocation addition overflow",
                ),
                BinOp::Sub(_) => (
                    format_ident!("checked_sub"),
                    "PCU invocation subtraction underflow",
                ),
                BinOp::Mul(_) => (
                    format_ident!("checked_mul"),
                    "PCU invocation multiplication overflow",
                ),
                BinOp::Div(_) => (
                    format_ident!("checked_div"),
                    "PCU invocation division by zero",
                ),
                BinOp::Rem(_) => (
                    format_ident!("checked_rem"),
                    "PCU invocation remainder by zero",
                ),
                _ => {
                    return Err(Error::new(
                        binary.op.span(),
                        "PCU invocation expression supports only +, -, *, /, and %",
                    ));
                }
            };
            Ok(quote! { (#lhs).#method(#rhs).expect(#failure) })
        }
        _ => Err(Error::new(
            expr.span(),
            "PCU invocation expression supports usize literals, const generics, parentheses, and checked + - * / %",
        )),
    }
}

fn parse_bindings(
    inputs: &syn::punctuated::Punctuated<FnArg, Token![,]>,
) -> Result<Vec<BindingSpec>, Error> {
    let mut bindings = Vec::new();
    for input in inputs {
        let FnArg::Typed(input) = input else {
            return Err(Error::new(
                input.span(),
                "PCU kernels do not support receiver arguments",
            ));
        };
        let Pat::Ident(pat) = input.pat.as_ref() else {
            return Err(Error::new(
                input.pat.span(),
                "PCU kernel bindings must be identifiers",
            ));
        };
        let access = parse_binding_type(&input.ty)?;
        let binding = u32::try_from(bindings.len())
            .map_err(|_| Error::new(input.span(), "too many PCU bindings for this macro"))?;
        bindings.push(BindingSpec {
            ident: pat.ident.clone(),
            access,
            binding,
        });
    }
    Ok(bindings)
}

fn parse_binding_type(ty: &Type) -> Result<BindingAccess, Error> {
    let Type::Reference(reference) = ty else {
        return Err(Error::new(
            ty.span(),
            "PCU binding types must be `&[f32]` or `&mut [f32]`",
        ));
    };
    if reference.lifetime.is_some() {
        return Err(Error::new(
            reference.span(),
            "explicit lifetimes on PCU dispatch bindings are unsupported",
        ));
    }
    let Type::Slice(slice) = reference.elem.as_ref() else {
        return Err(Error::new(
            reference.elem.span(),
            "PCU dispatch resources must be slices of f32",
        ));
    };
    let Type::Path(element) = slice.elem.as_ref() else {
        return Err(Error::new(
            slice.elem.span(),
            "this PCU dispatch macro currently supports only f32 elements",
        ));
    };
    if !element.path.is_ident("f32") {
        return Err(Error::new(
            element.span(),
            "this PCU dispatch macro currently supports only f32 elements",
        ));
    }
    Ok(if reference.mutability.is_some() {
        BindingAccess::ReadWrite
    } else {
        BindingAccess::ReadOnly
    })
}

enum ValidatedBody<'a> {
    Indexed {
        invocation: Ident,
        assignment: &'a ExprAssign,
    },
    GridStride {
        invocation: Ident,
        assignment: &'a ExprAssign,
        extent: &'a Expr,
    },
}

fn validate_body(function: &ItemFn) -> Result<ValidatedBody<'_>, Error> {
    let statements = &function.block.stmts;
    if statements.len() == 3
        && matches!(statements.first(), Some(Stmt::Local(local)) if matches!(local.pat, Pat::Ident(ref pat) if pat.mutability.is_some()))
    {
        return validate_grid_stride_body(statements);
    }
    if statements.len() != 2 {
        let span = statements
            .get(2)
            .map_or_else(|| function.block.span(), syn::spanned::Spanned::span);
        return Err(Error::new(
            span,
            "PCU dispatch body supports exactly one invocation binding (`context.global_invocation_id` or `pcu::context::global_invocation_id()`) followed by exactly one indexed output assignment",
        ));
    }

    let Stmt::Local(local) = &statements[0] else {
        return Err(Error::new(
            statements[0].span(),
            "first PCU dispatch statement must bind the invocation from `context.global_invocation_id` or `pcu::context::global_invocation_id()`",
        ));
    };
    if !local.attrs.is_empty() {
        return Err(Error::new(
            local.attrs[0].span(),
            "attributes on PCU dispatch local bindings are unsupported",
        ));
    }
    let Pat::Ident(pat) = &local.pat else {
        return Err(Error::new(
            local.pat.span(),
            "PCU dispatch invocation binding must be a plain identifier",
        ));
    };
    if pat.by_ref.is_some() || pat.mutability.is_some() || pat.subpat.is_some() {
        return Err(Error::new(
            pat.span(),
            "PCU dispatch invocation binding must be an immutable plain identifier",
        ));
    }
    let Some(init) = &local.init else {
        return Err(Error::new(
            local.pat.span(),
            "PCU dispatch invocation binding must initialize from `context.global_invocation_id` or `pcu::context::global_invocation_id()`",
        ));
    };
    if init.diverge.is_some() || !is_context_invocation_expr(&init.expr) {
        return Err(Error::new(
            invalid_context_expression_span(&init.expr),
            "PCU dispatch invocation binding must initialize from `context.global_invocation_id` or the zero-argument call `pcu::context::global_invocation_id()`",
        ));
    }

    let Stmt::Expr(Expr::Assign(assignment), Some(_)) = &statements[1] else {
        return Err(Error::new(
            diagnostic_statement_span(&statements[1]),
            "second PCU dispatch statement must be `output[invocation] = <expr>;`",
        ));
    };
    Ok(ValidatedBody::Indexed {
        invocation: pat.ident.clone(),
        assignment,
    })
}

/// Accepts the canonical grid-stride spelling and retains it as a structured loop region.
fn validate_grid_stride_body(statements: &[Stmt]) -> Result<ValidatedBody<'_>, Error> {
    let unsupported = |span| {
        Error::new(
            span,
            "this grid-stride form cannot be proven to execute exactly once per logical lane; use `invocations = N` with the canonical loop, or add loop-capable PCU IR support",
        )
    };
    let Stmt::Local(id_local) = &statements[0] else {
        return Err(unsupported(statements[0].span()));
    };
    let Pat::Ident(id_pat) = &id_local.pat else {
        return Err(unsupported(id_local.pat.span()));
    };
    if id_pat.mutability.is_none() || id_pat.by_ref.is_some() || id_pat.subpat.is_some() {
        return Err(unsupported(id_pat.ident.span()));
    }
    let Some(id_init) = &id_local.init else {
        return Err(unsupported(id_pat.ident.span()));
    };
    if id_init.diverge.is_some() || !is_context_invocation_expr(&id_init.expr) {
        return Err(unsupported(invalid_context_expression_span(&id_init.expr)));
    }

    let Stmt::Local(stride_local) = &statements[1] else {
        return Err(unsupported(statements[1].span()));
    };
    let Pat::Ident(stride_pat) = &stride_local.pat else {
        return Err(unsupported(stride_local.pat.span()));
    };
    if stride_pat.mutability.is_some() || stride_pat.by_ref.is_some() || stride_pat.subpat.is_some()
    {
        return Err(unsupported(stride_pat.ident.span()));
    }
    let Some(stride_init) = &stride_local.init else {
        return Err(unsupported(stride_pat.ident.span()));
    };
    if stride_init.diverge.is_some() || !is_context_invocation_count_expr(&stride_init.expr) {
        return Err(unsupported(invalid_context_expression_span(
            &stride_init.expr,
        )));
    }

    let Stmt::Expr(Expr::While(loop_expr), None) = &statements[2] else {
        return Err(unsupported(statements[2].span()));
    };
    let Expr::Binary(condition) = loop_expr.cond.as_ref() else {
        return Err(unsupported(loop_expr.cond.span()));
    };
    if !matches!(condition.op, BinOp::Lt(_)) || !is_ident_expr(&condition.left, &id_pat.ident) {
        return Err(unsupported(condition.span()));
    }
    let [
        Stmt::Expr(Expr::Assign(assignment), Some(_)),
        Stmt::Expr(Expr::Binary(increment), Some(_)),
    ] = loop_expr.body.stmts.as_slice()
    else {
        let span = loop_expr
            .body
            .stmts
            .first()
            .map_or_else(|| loop_expr.while_token.span(), grid_stride_statement_span);
        return Err(unsupported(span));
    };
    if !matches!(increment.op, BinOp::AddAssign(_)) {
        return Err(unsupported(increment.op.span()));
    }
    if !is_ident_expr(&increment.left, &id_pat.ident) {
        return Err(unsupported(increment.left.span()));
    }
    if !is_ident_expr(&increment.right, &stride_pat.ident) {
        return Err(unsupported(unsupported_expression_span(&increment.right)));
    }
    Ok(ValidatedBody::GridStride {
        invocation: id_pat.ident.clone(),
        assignment,
        extent: &condition.right,
    })
}

fn grid_stride_statement_span(statement: &Stmt) -> proc_macro2::Span {
    match statement {
        Stmt::Expr(Expr::If(expression), _) => expression.if_token.span(),
        _ => statement.span(),
    }
}

fn diagnostic_statement_span(statement: &Stmt) -> proc_macro2::Span {
    match statement {
        Stmt::Expr(Expr::If(expression), _) => expression.if_token.span(),
        _ => statement.span(),
    }
}

fn unsupported_expression_span(expression: &Expr) -> proc_macro2::Span {
    match expression {
        Expr::Binary(binary) => binary.op.span(),
        Expr::MethodCall(method_call) => method_call.method.span(),
        _ => expression.span(),
    }
}

fn invalid_context_expression_span(expression: &Expr) -> proc_macro2::Span {
    match expression {
        Expr::Call(call) => call
            .args
            .first()
            .map_or_else(|| call.func.span(), syn::spanned::Spanned::span),
        _ => expression.span(),
    }
}

fn is_context_invocation_count_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Field(ExprField { base, member, .. }) => {
            expr_ident(base).is_some_and(|ident| ident == "context")
                && member.to_token_stream().to_string() == "invocation_count"
        }
        _ => is_context_builtin_call(expr, "invocation_count"),
    }
}

fn is_ident_expr(expr: &Expr, ident: &Ident) -> bool {
    expr_ident(expr).is_some_and(|candidate| candidate == ident)
}

fn is_context_invocation_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Field(ExprField { base, member, .. }) => {
            expr_ident(base).is_some_and(|ident| ident == "context")
                && member.to_token_stream().to_string() == "global_invocation_id"
        }
        _ => is_context_builtin_call(expr, "global_invocation_id"),
    }
}

fn is_context_builtin_call(expr: &Expr, builtin: &str) -> bool {
    let Expr::Call(call) = expr else {
        return false;
    };
    if !call.args.is_empty() {
        return false;
    }
    let Expr::Path(path) = call.func.as_ref() else {
        return false;
    };
    if path.qself.is_some() || path.path.segments.len() != 3 {
        return false;
    }
    if path
        .path
        .segments
        .iter()
        .any(|segment| !matches!(segment.arguments, syn::PathArguments::None))
    {
        return false;
    }
    let mut segments = path.path.segments.iter();
    segments
        .next()
        .is_some_and(|segment| segment.ident == "pcu")
        && segments
            .next()
            .is_some_and(|segment| segment.ident == "context")
        && segments
            .next()
            .is_some_and(|segment| segment.ident == builtin)
}

fn validate_assignment_target<'a>(
    assignment: &ExprAssign,
    bindings: &'a [BindingSpec],
    invocation_ident: &Ident,
) -> Result<&'a BindingSpec, Error> {
    let Expr::Index(ExprIndex { expr, index, .. }) = assignment.left.as_ref() else {
        return Err(Error::new(
            assignment.left.span(),
            "PCU dispatch assignment target must be `output[invocation]`",
        ));
    };
    validate_invocation_index(index, invocation_ident)?;
    let Some(output_ident) = expr_ident(expr) else {
        return Err(Error::new(
            expr.span(),
            "PCU dispatch assignment target must be `output[invocation]`",
        ));
    };
    let Some(binding) = bindings
        .iter()
        .find(|binding| binding.ident == *output_ident)
    else {
        return Err(Error::new(
            output_ident.span(),
            "unknown PCU output binding",
        ));
    };
    if !matches!(binding.access, BindingAccess::ReadWrite) {
        return Err(Error::new(
            output_ident.span(),
            "PCU assignment target must be an `&mut [f32]` binding",
        ));
    }
    Ok(binding)
}

fn validate_invocation_index(expr: &Expr, invocation_ident: &Ident) -> Result<(), Error> {
    let Some(index_ident) = expr_ident(expr) else {
        return Err(Error::new(
            expr.span(),
            "PCU binding index must be the invocation identifier",
        ));
    };
    if index_ident == invocation_ident {
        Ok(())
    } else {
        Err(Error::new(
            expr.span(),
            "PCU binding index must be the invocation identifier",
        ))
    }
}

fn expr_ident(expr: &Expr) -> Option<&Ident> {
    let Expr::Path(path) = expr else {
        return None;
    };
    if path.path.segments.len() != 1 {
        return None;
    }
    path.path.segments.first().map(|segment| &segment.ident)
}

fn parse_f32_bits(float: &LitFloat) -> Result<u32, Error> {
    let value = float.base10_parse::<f32>()?;
    Ok(value.to_bits())
}

fn binding_tokens(binding: &BindingSpec, pcu: &Path) -> TokenStream2 {
    let name = binding.ident.to_string();
    let slot = binding.binding;
    let access = match binding.access {
        BindingAccess::ReadOnly => quote! { #pcu::PcuBindingAccess::ReadOnly },
        BindingAccess::ReadWrite => quote! { #pcu::PcuBindingAccess::ReadWrite },
    };
    quote! {
        #pcu::PcuBinding::scalar::<f32>(
            ::core::option::Option::Some(#name),
            0,
            #slot,
            #pcu::PcuBindingStorageClass::Storage,
            #access,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuDispatchArgs,
        expand_pcu_dispatch,
    };
    use syn::ItemFn;

    fn expand(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(input: &[f32], output: &mut [f32]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    #[test]
    fn lowers_supported_indexed_f32_map() {
        let tokens = expand("let invocation = context.global_invocation_id; output[invocation] = input[invocation] * 2.0;")
            .expect("supported map lowers");
        let generated = tokens.to_string();
        assert!(generated.contains("BindingLoad"));
        assert!(generated.contains("BindingStore"));
        assert!(generated.contains("PcuDispatchAluOp :: Mul"));
        assert!(generated.contains(":: fusion_pcu :: PcuBinding"));
    }

    #[test]
    fn lowers_multi_iteration_grid_stride_map_to_a_loop_region() {
        let tokens = expand(
            "let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < 64 { output[id] = input[id] * 2.0; id += stride; }",
        )
        .expect("canonical grid-stride loop lowers to a loop region");
        let generated = tokens.to_string();
        assert!(generated.contains("BindingLoad"));
        assert!(generated.contains("BindingStore"));
        assert!(generated.contains("GridStrideLoop"));
        assert!(generated.contains("GridStrideId"));
    }

    #[test]
    fn accepts_grid_stride_loop_with_a_smaller_dispatch_extent() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<const N: usize>(input: &[f32], output: &mut [f32]) { let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < N { output[id] = input[id]; id += stride; } }",
        )
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 32").expect("attribute parses");
        let tokens = expand_pcu_dispatch(args, &function)
            .expect("a smaller launch is represented by loop IR");
        assert!(tokens.to_string().contains("GridStrideLoop"));
    }

    #[test]
    fn accepts_invocation_spelling() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel(input: &[f32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] * 2.0; }",
        )
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("invocation spelling parses");
        let tokens = expand_pcu_dispatch(args, &function).expect("invocation spelling expands");
        assert!(tokens.to_string().contains("64"));
    }

    #[test]
    fn rejects_duplicate_invocation_arguments() {
        for source in [
            "invocations = 8, invocations = 8",
            "kernel_id = 1, kernel_id = 2, invocations = 8",
            "invocations = 8, crate_path = ::pcu, crate_path = ::other",
        ] {
            assert!(
                syn::parse_str::<PcuDispatchArgs>(source).is_err(),
                "{source}"
            );
        }
    }

    #[test]
    fn rejects_removed_thread_count_spelling() {
        let error = syn::parse_str::<PcuDispatchArgs>("threads = 8")
            .err()
            .expect("logical invocation count must use the new spelling");
        assert!(error.to_string().contains("invocations"));
    }

    #[test]
    fn rejects_invocation_expressions_outside_the_checked_const_subset() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<const R: usize>(input: &[f32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation]; }",
        )
        .expect("test function parses");
        for source in [
            "invocations = R << 1",
            "invocations = R + C",
            "invocations = size()",
        ] {
            let args = syn::parse_str::<PcuDispatchArgs>(source).expect("attribute parses");
            assert!(expand_pcu_dispatch(args, &function).is_err(), "{source}");
        }
    }

    #[test]
    fn rejects_type_generic_sources_until_element_semantics_exist() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<T: PcuScalar>(input: &[f32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation]; }",
        )
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 8").expect("attribute parses");
        let error = expand_pcu_dispatch(args, &function).expect_err("type generic is unsupported");
        let message = error.to_string();
        assert!(message.contains("does not yet support type generics"));
        assert!(message.contains("binding metadata and emitted constants currently use f32"));
        assert!(message.contains("T: PcuScalar"));
        assert!(message.contains("const NAME: usize"));
    }

    #[test]
    fn emits_all_runtime_references_through_configured_crate_path() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel(input: &[f32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] * 2.0; }",
        )
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64, crate_path = ::pcu_alias")
            .expect("renamed crate path arguments parse");
        let tokens = expand_pcu_dispatch(args, &function).expect("alias path expands");
        let generated = tokens.to_string();
        assert!(generated.contains(":: pcu_alias :: PcuBinding"));
        assert!(generated.contains(":: pcu_alias :: PcuDispatchDataOp"));
        assert!(!generated.contains("fusion_pcu"));
    }

    #[test]
    fn rejects_extra_let_instead_of_silently_dropping_it() {
        let error = expand(
            "let invocation = context.global_invocation_id; let ignored = 1.0; output[invocation] = input[invocation];",
        )
        .expect_err("extra local must be rejected");
        assert!(error.to_string().contains("exactly"));
    }

    #[test]
    fn rejects_extra_expression_statement() {
        let error =
            expand("let invocation = context.global_invocation_id; log(invocation); output[invocation] = input[invocation];")
                .expect_err("unsupported expression must be rejected");
        assert!(error.to_string().contains("exactly"));
    }

    #[test]
    fn rejects_multiple_assignments() {
        let error = expand(
            "let invocation = context.global_invocation_id; output[invocation] = input[invocation]; output[invocation] = 0.0;",
        )
        .expect_err("second assignment must be rejected");
        assert!(error.to_string().contains("exactly"));
    }

    #[test]
    fn rejects_non_assignment_body_statement() {
        let error = expand("let invocation = context.global_invocation_id; if invocation > 0 { output[invocation] = 1.0; }")
            .expect_err("control flow must be rejected in this subset");
        assert!(error.to_string().contains("second PCU dispatch statement"));
    }
}
