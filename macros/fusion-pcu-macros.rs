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
    GenericArgument,
    Ident,
    ItemFn,
    Lit,
    LitFloat,
    Pat,
    Path,
    PathArguments,
    Stmt,
    Token,
    Type,
    parse_macro_input,
};

struct PcuDispatchArgs {
    kernel_id: u32,
    invocations: u32,
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
                "kernel_id" | "invocations" => {
                    let value: syn::LitInt = input.parse()?;
                    let parsed = value.base10_parse::<u32>()?;
                    if key == "kernel_id" {
                        if kernel_id.is_some() {
                            return Err(Error::new(key.span(), "duplicate `kernel_id` argument"));
                        }
                        kernel_id = Some(parsed);
                    } else {
                        if invocations.is_some() {
                            return Err(Error::new(key.span(), "duplicate `invocations` argument"));
                        }
                        invocations = Some(parsed);
                    }
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
                        "#[pcu_dispatch] supports `kernel_id = <u32>`, `invocations = <u32>`, and `crate_path = <path>`",
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
                "#[pcu_dispatch] requires `invocations = <u32>`",
            )
        })?;
        if invocations == 0 {
            return Err(Error::new(
                input.span(),
                "#[pcu_dispatch] requires a non-zero invocation count",
            ));
        }

        Ok(Self {
            kernel_id: kernel_id.map_or(1, core::convert::identity),
            invocations,
            crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::fusion_pcu)),
        })
    }
}

#[derive(Clone, Copy)]
enum BindingAccess {
    Read,
    Write,
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
    next_value: u16,
    ops: Vec<TokenStream2>,
}

impl<'a> ExprEmitter<'a> {
    const fn new(
        bindings: &'a [BindingSpec],
        invocation_ident: &'a Ident,
        crate_path: &'a Path,
    ) -> Self {
        Self {
            bindings,
            invocation_ident,
            crate_path,
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
                expr.span(),
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
        let slot = self.binding(binding_ident, BindingAccess::Read)?.binding;
        let result = self.alloc_value(index.span())?;
        let pcu = self.crate_path;
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::BindingLoad {
                result: #pcu::PcuDispatchValueId(#result),
                binding: #pcu::PcuBindingRef::new(0, #slot),
                index: #pcu::PcuDispatchIndex::InvocationId,
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

    fn binding(&self, ident: &Ident, required: BindingAccess) -> Result<&BindingSpec, Error> {
        let Some(binding) = self.bindings.iter().find(|binding| binding.ident == *ident) else {
            return Err(Error::new(ident.span(), "unknown PCU binding"));
        };
        if !matches!(
            (binding.access, required),
            (BindingAccess::Read, BindingAccess::Read)
                | (BindingAccess::Write, BindingAccess::Write)
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

fn expand_pcu_dispatch(args: PcuDispatchArgs, function: &ItemFn) -> Result<TokenStream2, Error> {
    let vis = function.vis.clone();
    let function_ident = function.sig.ident.clone();
    let bindings_ident = format_ident!("{}_bindings", function_ident);
    let crate_path = args.crate_path;
    let binding_specs = parse_bindings(&function.sig.inputs)?;
    let (invocation_ident, assignment) = validate_body(function)?;
    let output_binding = validate_assignment_target(assignment, &binding_specs, &invocation_ident)?;
    let mut emitter = ExprEmitter::new(&binding_specs, &invocation_ident, &crate_path);
    let result_value = emitter.emit_expr(&assignment.right)?;
    let output_slot = output_binding.binding;
    let mut data_ops = emitter.ops;
    let pcu = &crate_path;
    data_ops.push(quote! {
        #pcu::PcuDispatchDataOp::BindingStore {
            binding: #pcu::PcuBindingRef::new(0, #output_slot),
            index: #pcu::PcuDispatchIndex::InvocationId,
            value: #pcu::PcuDispatchValueId(#result_value),
        }
    });

    let binding_items = binding_specs
        .iter()
        .map(|binding| binding_tokens(binding, pcu))
        .collect::<Vec<_>>();
    let binding_count = binding_items.len();
    let op_count = data_ops.len() + 1;
    let kernel_id = args.kernel_id;
    let invocations = args.invocations;

    Ok(quote! {
        #vis const fn #bindings_ident() -> [#pcu::PcuBinding<'static>; #binding_count] {
            [#(#binding_items),*]
        }

        #vis fn #function_ident<'a>(
            bindings: &'a [#pcu::PcuBinding<'a>],
        ) -> ::core::result::Result<
            #pcu::model::PcuDispatchKernelBuilder<'a, #op_count>,
            #pcu::PcuError,
        > {
            let builder = #pcu::model::PcuDispatchKernelBuilder::<#op_count>::new(
                #kernel_id,
                "main",
                [#invocations, 1, 1],
            )
            .with_bindings(bindings);
            #(let builder = builder.with_data_op(#data_ops)?;)*
            builder.with_control_op(#pcu::PcuDispatchControlOp::Return)
        }
    })
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
    let Type::Path(path) = ty else {
        return Err(Error::new(
            ty.span(),
            "PCU binding types must be read_storage<f32> or write_storage<f32>",
        ));
    };
    let Some(segment) = path.path.segments.last() else {
        return Err(Error::new(ty.span(), "invalid PCU binding type"));
    };
    let access = if segment.ident == "read_storage" {
        BindingAccess::Read
    } else if segment.ident == "write_storage" {
        BindingAccess::Write
    } else {
        return Err(Error::new(
            segment.ident.span(),
            "PCU binding types must be read_storage<f32> or write_storage<f32>",
        ));
    };
    validate_f32_generic(&segment.arguments)?;
    Ok(access)
}

fn validate_f32_generic(arguments: &PathArguments) -> Result<(), Error> {
    let PathArguments::AngleBracketed(arguments) = arguments else {
        return Err(Error::new(
            arguments.span(),
            "PCU binding type must specify `<f32>`",
        ));
    };
    if arguments.args.len() != 1 {
        return Err(Error::new(
            arguments.span(),
            "PCU binding type must specify exactly one value type",
        ));
    }
    let Some(GenericArgument::Type(Type::Path(ty))) = arguments.args.first() else {
        return Err(Error::new(
            arguments.span(),
            "PCU binding generic must be a type",
        ));
    };
    let Some(segment) = ty.path.segments.last() else {
        return Err(Error::new(ty.span(), "invalid PCU binding value type"));
    };
    if segment.ident != "f32" {
        return Err(Error::new(
            segment.ident.span(),
            "this first PCU dispatch macro cut only supports f32 bindings",
        ));
    }
    Ok(())
}

fn validate_body(function: &ItemFn) -> Result<(Ident, &ExprAssign), Error> {
    let statements = &function.block.stmts;
    if statements.len() != 2 {
        let span = statements
            .get(2)
            .map_or_else(|| function.block.span(), syn::spanned::Spanned::span);
        return Err(Error::new(
            span,
            "PCU dispatch body supports exactly `let invocation = context.global_invocation_id;` followed by one `output[invocation] = <expr>;` assignment",
        ));
    }

    let Stmt::Local(local) = &statements[0] else {
        return Err(Error::new(
            statements[0].span(),
            "first PCU dispatch statement must be `let invocation = context.global_invocation_id;`",
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
            "PCU dispatch invocation binding must initialize from `context.global_invocation_id`",
        ));
    };
    if init.diverge.is_some() || !is_context_invocation_expr(&init.expr) {
        return Err(Error::new(
            init.expr.span(),
            "PCU dispatch invocation binding must initialize from `context.global_invocation_id`",
        ));
    }

    let Stmt::Expr(Expr::Assign(assignment), Some(_)) = &statements[1] else {
        return Err(Error::new(
            statements[1].span(),
            "second PCU dispatch statement must be `output[invocation] = <expr>;`",
        ));
    };
    Ok((pat.ident.clone(), assignment))
}

fn is_context_invocation_expr(expr: &Expr) -> bool {
    let Expr::Field(ExprField { base, member, .. }) = expr else {
        return false;
    };
    let Some(base_ident) = expr_ident(base) else {
        return false;
    };
    base_ident == "context" && member.to_token_stream().to_string() == "global_invocation_id"
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
    if !matches!(binding.access, BindingAccess::Write) {
        return Err(Error::new(
            output_ident.span(),
            "PCU assignment target must be a write_storage<f32> binding",
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
        BindingAccess::Read => quote! { #pcu::PcuBindingAccess::ReadOnly },
        BindingAccess::Write => quote! { #pcu::PcuBindingAccess::WriteOnly },
    };
    quote! {
        #pcu::PcuBinding::value(
            ::core::option::Option::Some(#name),
            0,
            #slot,
            #pcu::PcuBindingStorageClass::Storage,
            #access,
            #pcu::PcuValueType::f32(),
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
            "fn kernel(input: read_storage<f32>, output: write_storage<f32>) {{ {body} }}"
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
    fn accepts_invocation_spelling() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel(input: read_storage<f32>, output: write_storage<f32>) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] * 2.0; }",
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
    fn emits_all_runtime_references_through_configured_crate_path() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel(input: read_storage<f32>, output: write_storage<f32>) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] * 2.0; }",
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
