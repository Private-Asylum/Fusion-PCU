#[path = "fusion-pcu-macros/checked_div_rem.rs"]
mod checked_div_rem;

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
    ExprMethodCall,
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum BindingAccess {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalarKind {
    F32,
    F64,
    U8,
    U16,
    U32,
    U64,
    I8,
    I16,
    I32,
    I64,
    Generic,
}

impl ScalarKind {
    fn rust_type(self) -> TokenStream2 {
        match self {
            Self::F32 => quote! { f32 },
            Self::F64 => quote! { f64 },
            Self::U8 => quote! { u8 },
            Self::U16 => quote! { u16 },
            Self::U32 => quote! { u32 },
            Self::U64 => quote! { u64 },
            Self::I8 => quote! { i8 },
            Self::I16 => quote! { i16 },
            Self::I32 => quote! { i32 },
            Self::I64 => quote! { i64 },
            Self::Generic => unreachable!("generic scalar has no concrete Rust type token"),
        }
    }
}

struct BindingSpec {
    ident: Ident,
    access: BindingAccess,
    binding: u32,
    scalar: ScalarKind,
    generic_scalar: Option<Ident>,
}

struct PcuHelper {
    ident: Ident,
    parameters: Vec<Ident>,
    body: Expr,
}

struct ExprEmitter<'a> {
    bindings: &'a [BindingSpec],
    invocation_ident: &'a Ident,
    crate_path: &'a Path,
    grid_stride: bool,
    helpers: &'a [PcuHelper],
    helper_stack: Vec<Ident>,
    values: Vec<(Ident, u16, ScalarKind)>,
    next_value: u16,
    ops: Vec<TokenStream2>,
}

impl<'a> ExprEmitter<'a> {
    const fn new(
        bindings: &'a [BindingSpec],
        invocation_ident: &'a Ident,
        crate_path: &'a Path,
        grid_stride: bool,
        helpers: &'a [PcuHelper],
    ) -> Self {
        Self {
            bindings,
            invocation_ident,
            crate_path,
            grid_stride,
            helpers,
            helper_stack: Vec::new(),
            values: Vec::new(),
            next_value: 1,
            ops: Vec::new(),
        }
    }

    fn emit_expr(&mut self, expr: &Expr) -> Result<(u16, ScalarKind), Error> {
        match expr {
            Expr::Binary(binary) => self.emit_binary(binary),
            Expr::Index(index) => self.emit_index_load(index),
            Expr::Lit(lit) => self.emit_lit(lit),
            Expr::Paren(paren) => self.emit_expr(&paren.expr),
            Expr::MethodCall(call) => self.emit_wrapping_method(call),
            Expr::Call(call) => self.emit_helper_call(call),
            Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let ident = &path.path.segments[0].ident;
                self.values
                    .iter()
                    .rev()
                    .find(|(name, _, _)| name == ident)
                    .map(|(_, value, scalar)| (*value, *scalar))
                    .ok_or_else(|| {
                        Error::new(
                            path.span(),
                            "PCU helper expressions may only reference their scalar parameters",
                        )
                    })
            }
            _ => Err(Error::new(
                unsupported_expression_span(expr),
                "unsupported PCU expression; supported subset is binding[index], f32 literals, parentheses, and + - * /",
            )),
        }
    }

    fn emit_helper_call(&mut self, call: &syn::ExprCall) -> Result<(u16, ScalarKind), Error> {
        let Expr::Path(path) = call.func.as_ref() else {
            return Err(Error::new(
                call.func.span(),
                "PCU helper calls must name a `#[pcu_fn]` sibling directly",
            ));
        };
        if path.qself.is_some() || path.path.segments.len() != 1 {
            return Err(Error::new(
                path.span(),
                "PCU helper calls must name a `#[pcu_fn]` sibling directly",
            ));
        }
        let helper_ident = &path.path.segments[0].ident;
        let Some(helper) = self
            .helpers
            .iter()
            .find(|helper| helper.ident == *helper_ident)
        else {
            return Err(Error::new(
                helper_ident.span(),
                "PCU helper calls require `#[pcu_module]` and a sibling `#[pcu_fn]` definition",
            ));
        };
        if helper.parameters.len() != call.args.len() {
            return Err(Error::new(
                call.span(),
                "PCU helper argument count does not match its definition",
            ));
        }
        if self.helper_stack.iter().any(|name| name == helper_ident) {
            return Err(Error::new(
                call.span(),
                "recursive PCU helper calls are not supported",
            ));
        }

        let mut arguments = Vec::with_capacity(call.args.len());
        for argument in &call.args {
            arguments.push(self.emit_expr(argument)?);
        }
        if arguments
            .iter()
            .any(|(_, scalar)| *scalar != ScalarKind::F32)
        {
            return Err(Error::new(
                call.span(),
                "this PCU helper profile currently accepts only f32 arguments",
            ));
        }

        let values_len = self.values.len();
        self.helper_stack.push(helper_ident.clone());
        self.values.extend(
            helper
                .parameters
                .iter()
                .cloned()
                .zip(arguments.iter().map(|(value, scalar)| (*value, *scalar)))
                .map(|(ident, (value, scalar))| (ident, value, scalar)),
        );
        let result = self.emit_expr(&helper.body);
        self.values.truncate(values_len);
        self.helper_stack.pop();
        result
    }

    fn emit_binary(&mut self, binary: &ExprBinary) -> Result<(u16, ScalarKind), Error> {
        let (lhs, lhs_type) = self.emit_expr(&binary.left)?;
        let (rhs, rhs_type) = self.emit_expr(&binary.right)?;
        if lhs_type != rhs_type || !matches!(lhs_type, ScalarKind::F32 | ScalarKind::F64) {
            return Err(Error::new(
                binary.span(),
                "PCU floating arithmetic requires matching f32 or f64 operands",
            ));
        }
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
        let value_type = match lhs_type {
            ScalarKind::F32 => quote! { #pcu::PcuValueType::f32() },
            ScalarKind::F64 => quote! { #pcu::PcuValueType::f64() },
            _ => unreachable!("floating type checked above"),
        };
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::Alu {
                value_type: #value_type,
                result: #pcu::PcuDispatchValueId(#result),
                op: #op,
                lhs: #pcu::PcuDispatchValueId(#lhs),
                rhs: #pcu::PcuDispatchValueId(#rhs),
            }
        });
        Ok((result, lhs_type))
    }

    fn emit_wrapping_method(&mut self, call: &ExprMethodCall) -> Result<(u16, ScalarKind), Error> {
        let method = call.method.to_string();
        let op = match method.as_str() {
            "wrapping_add" => quote! { Add },
            "wrapping_sub" => quote! { Sub },
            "wrapping_mul" => quote! { Mul },
            _ => {
                return Err(Error::new(
                    unsupported_expression_span(&Expr::MethodCall(call.clone())),
                    "unsupported PCU expression; supported subset is binding[index], f32 literals, parentheses, and + - * /",
                ));
            }
        };
        if call.args.len() != 1 || call.turbofish.is_some() {
            return Err(Error::new(
                call.span(),
                "wrapping arithmetic requires exactly one argument",
            ));
        }
        let rhs_expr = call.args.first().expect("argument count checked");
        let (lhs, lhs_type) = self.emit_expr(&call.receiver)?;
        let (rhs, rhs_type) = self.emit_expr(rhs_expr)?;
        if lhs_type != rhs_type
            || !matches!(
                lhs_type,
                ScalarKind::U8
                    | ScalarKind::U16
                    | ScalarKind::U32
                    | ScalarKind::U64
                    | ScalarKind::I8
                    | ScalarKind::I16
                    | ScalarKind::I32
                    | ScalarKind::I64
                    | ScalarKind::Generic
            )
        {
            return Err(Error::new(
                call.span(),
                "wrapping PCU arithmetic requires matching u8, u16, u32, u64, i8, i16, i32, or i64 operands",
            ));
        }
        let result = self.alloc_value(call.span())?;
        let pcu = self.crate_path;
        let value_type = match lhs_type {
            ScalarKind::U8 => quote! { #pcu::PcuValueType::u8() },
            ScalarKind::U16 => quote! { #pcu::PcuValueType::u16() },
            ScalarKind::U32 => quote! { #pcu::PcuValueType::u32() },
            ScalarKind::U64 => quote! { #pcu::PcuValueType::u64() },
            ScalarKind::I8 => quote! { #pcu::PcuValueType::i8() },
            ScalarKind::I16 => quote! { #pcu::PcuValueType::i16() },
            ScalarKind::I32 => quote! { #pcu::PcuValueType::i32() },
            ScalarKind::I64 => quote! { #pcu::PcuValueType::i64() },
            ScalarKind::Generic => {
                let generic = self
                    .bindings
                    .iter()
                    .find_map(|binding| binding.generic_scalar.as_ref())
                    .expect("generic arithmetic has a generic binding");
                quote! { #pcu::PcuValueType::Scalar(<#generic as #pcu::PcuScalar>::TYPE) }
            }
            ScalarKind::F32 | ScalarKind::F64 => {
                unreachable!("integer operand checked")
            }
        };
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::Alu {
                value_type: #value_type,
                result: #pcu::PcuDispatchValueId(#result),
                op: #pcu::PcuDispatchAluOp::#op,
                lhs: #pcu::PcuDispatchValueId(#lhs),
                rhs: #pcu::PcuDispatchValueId(#rhs),
            }
        });
        Ok((result, lhs_type))
    }

    fn emit_index_load(&mut self, index: &ExprIndex) -> Result<(u16, ScalarKind), Error> {
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
        Ok((
            result,
            self.binding(binding_ident, BindingAccess::ReadOnly)?.scalar,
        ))
    }

    fn emit_lit(&mut self, lit: &ExprLit) -> Result<(u16, ScalarKind), Error> {
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
        Ok((result, ScalarKind::F32))
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

/// Mark a pure, expression-bodied scalar helper owned by an enclosing `#[pcu_module]`.
#[proc_macro_attribute]
pub fn pcu_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new(
            proc_macro2::Span::call_site(),
            "`#[pcu_fn]` takes no arguments",
        )
        .into_compile_error()
        .into();
    }
    let function = parse_macro_input!(item as ItemFn);
    Error::new(
        function.sig.ident.span(),
        "`#[pcu_fn]` helpers must be declared inside an inline `#[pcu_module]`",
    )
    .into_compile_error()
    .into()
}

/// Owns a module of bounded PCU kernels and explicitly marked pure scalar helpers.
///
/// The first helper profile accepts only `f32` scalar arguments and return values, with a pure
/// arithmetic expression body. Helpers are inlined into the kernel's dispatch IR.
#[proc_macro_attribute]
pub fn pcu_module(attr: TokenStream, item: TokenStream) -> TokenStream {
    if !attr.is_empty() {
        return Error::new(
            proc_macro2::Span::call_site(),
            "`#[pcu_module]` takes no arguments",
        )
        .into_compile_error()
        .into();
    }
    let mut module = parse_macro_input!(item as syn::ItemMod);
    match expand_pcu_module(&mut module) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
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
    expand_pcu_dispatch_with_helpers(args, function, &[])
}

// Keep the shared lowering path together: generic identities and concrete kernels must pass the
// same structural/body validation before they diverge into their respective typed builders.
#[allow(clippy::too_many_lines)]
fn expand_pcu_dispatch_with_helpers(
    args: PcuDispatchArgs,
    function: &ItemFn,
    helpers: &[PcuHelper],
) -> Result<TokenStream2, Error> {
    let vis = function.vis.clone();
    let function_ident = function.sig.ident.clone();
    let bindings_ident = format_ident!("{}_bindings", function_ident);
    let crate_path = args.crate_path;
    let const_generics = validate_const_generics(function)?;
    let generic_scalar = generic_scalar_type(function)?;
    if generic_scalar.is_some() && !helpers.is_empty() {
        return Err(Error::new(
            function.sig.generics.span(),
            "generic PCU kernels currently do not support helper functions",
        ));
    }
    let invocation_expr = lower_invocation_expr(&args.invocations, &const_generics)?;
    let generated_generics = if function.sig.generics.params.is_empty() {
        quote! { <'a> }
    } else {
        let params = &function.sig.generics.params;
        quote! { <'a, #params> }
    };
    let binding_specs = parse_bindings(&function.sig.inputs, generic_scalar.as_ref())?;
    let (loop_extent, data_ops) = if let Some((extent, data_ops)) =
        checked_div_rem::lower(function, &binding_specs, &crate_path)?
    {
        (extent, data_ops)
    } else {
        let body = validate_body(function)?;
        let (invocation, assignment, extent) = match &body {
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
        let output_binding = validate_assignment_target(assignment, &binding_specs, invocation)?;
        if let Some(scalar_type) = &generic_scalar {
            if generic_scalar_wrapping(function) {
                validate_generic_wrapping_map(assignment, &binding_specs, invocation, scalar_type)?;
                let mut emitter = ExprEmitter::new(
                    &binding_specs,
                    invocation,
                    &crate_path,
                    extent.is_some(),
                    helpers,
                );
                let (result_value, result_type) = emitter.emit_expr(&assignment.right)?;
                if result_type != ScalarKind::Generic {
                    return Err(Error::new(
                        assignment.right.span(),
                        "generic wrapping result type mismatch",
                    ));
                }
                let output_slot = output_binding.binding;
                let store_index = if extent.is_some() {
                    quote! { GridStrideId }
                } else {
                    quote! { InvocationId }
                };
                emitter.ops.push(quote! {
                    #crate_path::PcuDispatchDataOp::BindingStore {
                        binding: #crate_path::PcuBindingRef::new(0, #output_slot),
                        index: #crate_path::PcuDispatchIndex::#store_index,
                        value: #crate_path::PcuDispatchValueId(#result_value),
                    }
                });
                (extent, emitter.ops)
            } else {
                validate_generic_identity(assignment, &binding_specs, invocation, scalar_type)?;
                let data_ops = lower_generic_identity(
                    &binding_specs,
                    output_binding,
                    &crate_path,
                    extent.is_some(),
                );
                (extent, data_ops)
            }
        } else {
            let mut emitter = ExprEmitter::new(
                &binding_specs,
                invocation,
                &crate_path,
                extent.is_some(),
                helpers,
            );
            let (result_value, result_type) = emitter.emit_expr(&assignment.right)?;
            if result_type != output_binding.scalar {
                return Err(Error::new(
                    assignment.right.span(),
                    "PCU store value type must match the output binding element type",
                ));
            }
            let pcu = &crate_path;
            let output_slot = output_binding.binding;
            let store_index = if extent.is_some() {
                quote! { GridStrideId }
            } else {
                quote! { InvocationId }
            };
            emitter.ops.push(quote! {
                #pcu::PcuDispatchDataOp::BindingStore {
                    binding: #pcu::PcuBindingRef::new(0, #output_slot),
                    index: #pcu::PcuDispatchIndex::#store_index,
                    value: #pcu::PcuDispatchValueId(#result_value),
                }
            });
            (extent, emitter.ops)
        }
    };
    let pcu = &crate_path;

    let generic_wrapping = generic_scalar.is_some() && generic_scalar_wrapping(function);
    let wrapping_body_ident = format_ident!("__{}_PcuWrappingBody", function_ident);
    let (operations, mut op_count, wrapping_body_item) = if generic_wrapping
        && let Some(extent) = loop_extent
    {
        let lowered_extent = lower_invocation_expr(extent, &const_generics)?;
        let body_len = data_ops.len();
        let body_ops = data_ops
            .iter()
            .map(|op| quote! { #crate_path::PcuDispatchOp::Data(#op) });
        let scalar_ident = generic_scalar
            .as_ref()
            .expect("generic wrapping has a type parameter");
        let body_type = quote! {
            #[allow(non_camel_case_types)]
            struct #wrapping_body_ident<T: #crate_path::PcuWrappingInteger>(::core::marker::PhantomData<T>);
            impl<T: #crate_path::PcuWrappingInteger> #wrapping_body_ident<T> {
                const BODY: [#crate_path::PcuDispatchOp<'static>; #body_len] = [#(#body_ops),*];
            }
        };
        let operation = quote! {
            let builder = builder.with_op(#crate_path::PcuDispatchOp::GridStrideLoop {
                extent: const { let extent: usize = #lowered_extent; assert!(extent != 0, "PCU grid-stride extent must be nonzero"); assert!(extent <= u32::MAX as usize, "PCU grid-stride extent exceeds u32"); extent as u32 },
                body: &#wrapping_body_ident::<#scalar_ident>::BODY,
            })?;
        };
        (operation, 2, body_type)
    } else {
        let (operations, count) =
            build_body_operations(&data_ops, loop_extent, &const_generics, &crate_path)?;
        (operations, count, quote! {})
    };

    let binding_items = binding_specs
        .iter()
        .map(|binding| binding_tokens(binding, pcu))
        .collect::<Vec<_>>();
    let binding_count = binding_items.len();
    let kernel_id = args.kernel_id;
    let binding_generic = generic_scalar.as_ref().map(|ident| {
        if generic_scalar_wrapping(function) {
            quote! { <#ident: #pcu::PcuWrappingInteger> }
        } else {
            quote! { <#ident: #pcu::PcuScalar> }
        }
    });
    let binding_lifetime = quote! { 'static };
    let generic_identity = generic_scalar.is_some() && !generic_scalar_wrapping(function);
    if generic_identity && loop_extent.is_some() {
        // The typed builder has a fixed three-op capacity: loop region plus terminal return.
        op_count = 3;
    }
    let normal_builder = quote! {
        let builder = #pcu::model::PcuDispatchKernelBuilder::<#op_count>::new(
            #kernel_id,
            "main",
            [invocations, 1, 1],
        )
        .with_bindings(bindings);
        #operations
        builder.with_control_op(#pcu::PcuDispatchControlOp::Return)
    };
    let builder_body = if let Some(scalar_ident) = &generic_scalar
        && generic_identity
    {
        if let Some(extent) = loop_extent {
            let extent = lower_invocation_expr(extent, &const_generics)?;
            quote! {
                #pcu::model::PcuScalarIdentityBuilder::<#scalar_ident>::build_grid_stride(
                    #kernel_id,
                    invocations,
                    const {
                        let extent: usize = #extent;
                        assert!(extent != 0, "PCU grid-stride extent must be nonzero");
                        assert!(extent <= u32::MAX as usize, "PCU grid-stride extent exceeds u32");
                        extent as u32
                    },
                    bindings,
                )
            }
        } else {
            quote! {
                #pcu::model::PcuScalarIdentityBuilder::<#scalar_ident>::build(
                    #kernel_id,
                    invocations,
                    bindings,
                )
            }
        }
    } else {
        normal_builder
    };
    let builder_result = if generic_identity {
        quote! { #pcu::model::PcuScalarIdentityBuildError }
    } else {
        quote! { #pcu::PcuError }
    };
    Ok(quote! {
        #wrapping_body_item
        #vis const fn #bindings_ident #binding_generic() -> [#pcu::PcuBinding<#binding_lifetime>; #binding_count] {
            [#(#binding_items),*]
        }

        #vis fn #function_ident #generated_generics(
            bindings: &'a [#pcu::PcuBinding<'a>],
        ) -> ::core::result::Result<#pcu::model::PcuDispatchKernelBuilder<'a, #op_count>, #builder_result> {
            let invocations: u32 = const {
                let count: usize = #invocation_expr;
                assert!(count != 0, "PCU invocation count must be nonzero");
                assert!(count <= u32::MAX as usize, "PCU invocation count exceeds u32");
                count as u32
            };
            #builder_body
        }
    })
}

fn lower_generic_identity(
    bindings: &[BindingSpec],
    output: &BindingSpec,
    crate_path: &Path,
    grid_stride: bool,
) -> Vec<TokenStream2> {
    let input = bindings
        .iter()
        .find(|binding| binding.access == BindingAccess::ReadOnly)
        .expect("validated generic identity signature");
    let output_slot = output.binding;
    let input_slot = input.binding;
    let index = if grid_stride {
        quote! { #crate_path::PcuDispatchIndex::GridStrideId }
    } else {
        quote! { #crate_path::PcuDispatchIndex::InvocationId }
    };
    vec![
        quote! {
            #crate_path::PcuDispatchDataOp::BindingLoad {
                result: #crate_path::PcuDispatchValueId(1),
                binding: #crate_path::PcuBindingRef::new(0, #input_slot),
                index: #index,
            }
        },
        quote! {
            #crate_path::PcuDispatchDataOp::BindingStore {
                binding: #crate_path::PcuBindingRef::new(0, #output_slot),
                index: #index,
                value: #crate_path::PcuDispatchValueId(1),
            }
        },
    ]
}

fn expand_pcu_module(module: &mut syn::ItemMod) -> Result<TokenStream2, Error> {
    let Some((_, items)) = module.content.take() else {
        return Err(Error::new_spanned(
            module,
            "`#[pcu_module]` requires an inline module so its helpers are visible to the frontend",
        ));
    };

    let helpers = items
        .iter()
        .filter_map(|item| match item {
            syn::Item::Fn(function) if has_attribute(&function.attrs, "pcu_fn") => {
                Some(parse_pcu_helper(function))
            }
            _ => None,
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_pcu_helpers(&helpers)?;

    let mut expanded = Vec::with_capacity(items.len());
    let mut kernel_count = 0_usize;
    for item in items {
        match item {
            syn::Item::Fn(function) if has_attribute(&function.attrs, "pcu_fn") => {}
            syn::Item::Fn(mut function) => {
                let pcu_attributes = function
                    .attrs
                    .iter()
                    .filter(|attribute| {
                        attribute_ends_with(attribute, "pcu")
                            || attribute_ends_with(attribute, "pcu_dispatch")
                    })
                    .cloned()
                    .collect::<Vec<_>>();
                if pcu_attributes.is_empty() {
                    expanded.push(quote! { #function });
                    continue;
                }
                if pcu_attributes.len() != 1 {
                    return Err(Error::new_spanned(
                        &function.sig.ident,
                        "a PCU module function must have exactly one `#[pcu(...)]` or `#[pcu_dispatch(...)]` attribute",
                    ));
                }
                if helpers
                    .iter()
                    .any(|helper| helper.ident == function.sig.ident)
                {
                    return Err(Error::new_spanned(
                        &function.sig.ident,
                        "a PCU kernel and a `#[pcu_fn]` helper cannot share a name",
                    ));
                }
                let attribute = &pcu_attributes[0];
                let args = attribute.parse_args::<PcuDispatchArgs>()?;
                function.attrs.retain(|candidate| {
                    !attribute_ends_with(candidate, "pcu")
                        && !attribute_ends_with(candidate, "pcu_dispatch")
                });
                if let Some(attribute) = function.attrs.first() {
                    return Err(Error::new_spanned(
                        attribute,
                        "`#[pcu_module]` kernels currently accept only their PCU attribute",
                    ));
                }
                kernel_count += 1;
                expanded.push(expand_pcu_dispatch_with_helpers(args, &function, &helpers)?);
            }
            other => {
                if let Some(attribute) = item_attribute(&other, "pcu_fn") {
                    return Err(Error::new_spanned(
                        attribute,
                        "`#[pcu_fn]` may only mark a free function inside `#[pcu_module]`",
                    ));
                }
                expanded.push(quote! { #other });
            }
        }
    }
    if kernel_count == 0 {
        return Err(Error::new_spanned(
            &module.ident,
            "`#[pcu_module]` requires at least one `#[pcu(...)]` or `#[pcu_dispatch(...)]` kernel",
        ));
    }

    let attrs = &module.attrs;
    let vis = &module.vis;
    let unsafety = &module.unsafety;
    let ident = &module.ident;
    Ok(quote! {
        #(#attrs)*
        #vis #unsafety mod #ident {
            #(#expanded)*
        }
    })
}

fn item_attribute<'a>(item: &'a syn::Item, name: &str) -> Option<&'a syn::Attribute> {
    let attrs = match item {
        syn::Item::Const(item) => &item.attrs,
        syn::Item::Enum(item) => &item.attrs,
        syn::Item::ExternCrate(item) => &item.attrs,
        syn::Item::Fn(item) => &item.attrs,
        syn::Item::ForeignMod(item) => &item.attrs,
        syn::Item::Impl(item) => &item.attrs,
        syn::Item::Macro(item) => &item.attrs,
        syn::Item::Mod(item) => &item.attrs,
        syn::Item::Static(item) => &item.attrs,
        syn::Item::Struct(item) => &item.attrs,
        syn::Item::Trait(item) => &item.attrs,
        syn::Item::TraitAlias(item) => &item.attrs,
        syn::Item::Type(item) => &item.attrs,
        syn::Item::Union(item) => &item.attrs,
        syn::Item::Use(item) => &item.attrs,
        _ => return None,
    };
    attrs
        .iter()
        .find(|attribute| attribute_ends_with(attribute, name))
}

fn has_attribute(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs
        .iter()
        .any(|attribute| attribute_ends_with(attribute, name))
}

fn attribute_ends_with(attribute: &syn::Attribute, name: &str) -> bool {
    attribute
        .path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == name)
}

fn parse_pcu_helper(function: &ItemFn) -> Result<PcuHelper, Error> {
    let marker_count = function
        .attrs
        .iter()
        .filter(|attribute| attribute_ends_with(attribute, "pcu_fn"))
        .count();
    if marker_count != 1 {
        return Err(Error::new_spanned(
            &function.sig.ident,
            "a PCU helper must have exactly one `#[pcu_fn]` marker",
        ));
    }
    if let Some(marker) = function
        .attrs
        .iter()
        .find(|attribute| attribute_ends_with(attribute, "pcu_fn"))
        && !matches!(marker.meta, syn::Meta::Path(_))
    {
        return Err(Error::new_spanned(marker, "`#[pcu_fn]` takes no arguments"));
    }
    if let Some(attribute) = function
        .attrs
        .iter()
        .find(|attribute| !attribute_ends_with(attribute, "pcu_fn"))
    {
        return Err(Error::new_spanned(
            attribute,
            "`#[pcu_module]` helpers currently accept only the `#[pcu_fn]` marker",
        ));
    }
    if function.sig.asyncness.is_some()
        || function.sig.constness.is_some()
        || function.sig.unsafety.is_some()
        || function.sig.abi.is_some()
        || !function.sig.generics.params.is_empty()
        || function.sig.generics.where_clause.is_some()
        || function.sig.variadic.is_some()
    {
        return Err(Error::new_spanned(
            &function.sig,
            "PCU helpers must be plain, non-generic, safe Rust functions",
        ));
    }
    let ReturnType::Type(_, return_type) = &function.sig.output else {
        return Err(Error::new_spanned(
            &function.sig.output,
            "PCU helpers must declare an explicit `-> f32` return type",
        ));
    };
    if !matches!(return_type.as_ref(), Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32"))
    {
        return Err(Error::new_spanned(
            return_type,
            "this PCU helper profile supports only the `f32` scalar type",
        ));
    }
    let mut parameters = Vec::with_capacity(function.sig.inputs.len());
    for input in &function.sig.inputs {
        let FnArg::Typed(argument) = input else {
            return Err(Error::new_spanned(
                input,
                "PCU helpers cannot have a receiver",
            ));
        };
        let Pat::Ident(pattern) = argument.pat.as_ref() else {
            return Err(Error::new_spanned(
                &argument.pat,
                "PCU helper parameters must be simple identifiers",
            ));
        };
        if !matches!(argument.ty.as_ref(), Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32"))
        {
            return Err(Error::new_spanned(
                &argument.ty,
                "this PCU helper profile supports only `f32` scalar parameters",
            ));
        }
        if parameters
            .iter()
            .any(|parameter: &Ident| parameter == &pattern.ident)
        {
            return Err(Error::new_spanned(
                &pattern.ident,
                "PCU helper parameter names must be unique",
            ));
        }
        parameters.push(pattern.ident.clone());
    }
    let [Stmt::Expr(body, None)] = function.block.stmts.as_slice() else {
        return Err(Error::new_spanned(
            &function.block,
            "PCU helper bodies must be one pure scalar expression without statements or control flow",
        ));
    };
    Ok(PcuHelper {
        ident: function.sig.ident.clone(),
        parameters,
        body: body.clone(),
    })
}

fn validate_pcu_helpers(helpers: &[PcuHelper]) -> Result<(), Error> {
    for (index, helper) in helpers.iter().enumerate() {
        if helpers[..index]
            .iter()
            .any(|prior| prior.ident == helper.ident)
        {
            return Err(Error::new(
                helper.ident.span(),
                "duplicate PCU helper name in this module",
            ));
        }
    }
    let mut call_graph = Vec::with_capacity(helpers.len());
    for helper in helpers {
        let mut calls = Vec::new();
        validate_helper_expression(&helper.body, &helper.parameters, helpers, &mut calls)?;
        call_graph.push(calls);
    }
    let mut state = vec![0_u8; helpers.len()];
    for index in 0..helpers.len() {
        validate_helper_acyclic(index, helpers, &call_graph, &mut state)?;
    }
    Ok(())
}

fn validate_helper_expression(
    expression: &Expr,
    parameters: &[Ident],
    helpers: &[PcuHelper],
    calls: &mut Vec<usize>,
) -> Result<(), Error> {
    match expression {
        Expr::Binary(binary) => {
            if !matches!(
                binary.op,
                BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_) | BinOp::Div(_)
            ) {
                return Err(Error::new(
                    binary.op.span(),
                    "PCU helper arithmetic supports only `+`, `-`, `*`, and `/`",
                ));
            }
            validate_helper_expression(&binary.left, parameters, helpers, calls)?;
            validate_helper_expression(&binary.right, parameters, helpers, calls)
        }
        Expr::Call(call) => {
            let Expr::Path(path) = call.func.as_ref() else {
                return Err(Error::new(
                    call.func.span(),
                    "PCU helpers may call only direct sibling `#[pcu_fn]` helpers",
                ));
            };
            if path.qself.is_some() || path.path.segments.len() != 1 {
                return Err(Error::new(
                    path.span(),
                    "PCU helpers may call only direct sibling `#[pcu_fn]` helpers",
                ));
            }
            let name = &path.path.segments[0].ident;
            let Some((helper_index, target)) = helpers
                .iter()
                .enumerate()
                .find(|(_, candidate)| candidate.ident == *name)
            else {
                return Err(Error::new(
                    name.span(),
                    "PCU helper calls must resolve to a sibling `#[pcu_fn]` helper",
                ));
            };
            if target.parameters.len() != call.args.len() {
                return Err(Error::new(
                    call.span(),
                    "PCU helper argument count does not match its definition",
                ));
            }
            calls.push(helper_index);
            for argument in &call.args {
                validate_helper_expression(argument, parameters, helpers, calls)?;
            }
            Ok(())
        }
        Expr::Lit(literal) => match &literal.lit {
            Lit::Float(value) if matches!(value.suffix(), "" | "f32") => Ok(()),
            Lit::Float(value) => Err(Error::new(
                value.span(),
                "PCU helper constants must be unsuffixed or explicitly suffixed `f32`",
            )),
            _ => Err(Error::new(
                literal.span(),
                "PCU helper constants must be floating point literals",
            )),
        },
        Expr::Paren(paren) => validate_helper_expression(&paren.expr, parameters, helpers, calls),
        Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
            let name = &path.path.segments[0].ident;
            if parameters.iter().any(|parameter| parameter == name) {
                Ok(())
            } else {
                Err(Error::new(
                    name.span(),
                    "PCU helper expressions may reference only their scalar parameters",
                ))
            }
        }
        _ => Err(Error::new(
            expression.span(),
            "PCU helper bodies support only scalar parameters, f32 literals, parentheses, arithmetic, and sibling helper calls",
        )),
    }
}

fn validate_helper_acyclic(
    index: usize,
    helpers: &[PcuHelper],
    call_graph: &[Vec<usize>],
    state: &mut [u8],
) -> Result<(), Error> {
    match state[index] {
        1 => {
            return Err(Error::new(
                helpers[index].ident.span(),
                "recursive PCU helper calls are not supported",
            ));
        }
        2 => return Ok(()),
        _ => {}
    }
    state[index] = 1;
    for &called in &call_graph[index] {
        validate_helper_acyclic(called, helpers, call_graph, state)?;
    }
    state[index] = 2;
    Ok(())
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
            continue;
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

fn generic_scalar_type(function: &ItemFn) -> Result<Option<Ident>, Error> {
    let types = function
        .sig
        .generics
        .params
        .iter()
        .filter_map(|parameter| match parameter {
            GenericParam::Type(parameter) => Some(parameter),
            _ => None,
        })
        .collect::<Vec<_>>();
    if types.is_empty() {
        return Ok(None);
    }
    if types.len() != 1 {
        return Err(Error::new(
            function.sig.generics.span(),
            "generic PCU kernels support exactly one sealed scalar type parameter",
        ));
    }
    let parameter = types[0];
    let has_supported_bound = parameter.bounds.len() == 1
        && parameter.bounds.iter().any(|bound| match bound {
            syn::TypeParamBound::Trait(bound) => {
                bound.path.segments.last().is_some_and(|segment| {
                    segment.ident == "PcuScalar" || segment.ident == "PcuWrappingInteger"
                })
            }
            _ => false,
        });
    if !has_supported_bound || parameter.default.is_some() || !parameter.attrs.is_empty() {
        return Err(Error::new(
            parameter.span(),
            "generic PCU kernels require exactly `T: PcuScalar` or `T: PcuWrappingInteger`; both traits are sealed",
        ));
    }
    Ok(Some(parameter.ident.clone()))
}

fn generic_scalar_wrapping(function: &ItemFn) -> bool {
    function
        .sig
        .generics
        .params
        .iter()
        .find_map(|parameter| match parameter {
            GenericParam::Type(parameter) => {
                parameter.bounds.iter().find_map(|bound| match bound {
                    syn::TypeParamBound::Trait(bound) => bound
                        .path
                        .segments
                        .last()
                        .map(|segment| segment.ident == "PcuWrappingInteger"),
                    _ => None,
                })
            }
            _ => None,
        })
        .unwrap_or(false)
}

fn validate_generic_wrapping_map(
    assignment: &ExprAssign,
    bindings: &[BindingSpec],
    invocation: &Ident,
    scalar_type: &Ident,
) -> Result<(), Error> {
    if bindings.len() != 3
        || bindings.iter().any(|binding| {
            binding.scalar != ScalarKind::Generic
                || binding.generic_scalar.as_ref() != Some(scalar_type)
        })
    {
        return Err(Error::new(
            assignment.span(),
            "generic `T: PcuWrappingInteger` maps require exactly two `&[T]` inputs and one `&mut [T]` output",
        ));
    }
    let Expr::MethodCall(call) = assignment.right.as_ref() else {
        return Err(Error::new(
            assignment.right.span(),
            "generic wrapping maps currently support only `output[id] = left[id].wrapping_add/sub/mul(right[id])`",
        ));
    };
    if !matches!(
        call.method.to_string().as_str(),
        "wrapping_add" | "wrapping_sub" | "wrapping_mul"
    ) || call.args.len() != 1
        || call.turbofish.is_some()
    {
        return Err(Error::new(
            call.span(),
            "generic wrapping maps support only one-argument wrapping_add, wrapping_sub, or wrapping_mul",
        ));
    }
    let mut input_names = Vec::with_capacity(2);
    for operand in [
        call.receiver.as_ref(),
        call.args.first().expect("argument count checked"),
    ] {
        let Expr::Index(index) = operand else {
            return Err(Error::new(
                operand.span(),
                "generic wrapping operands must be indexed bindings",
            ));
        };
        validate_invocation_index(&index.index, invocation)?;
        let Some(name) = expr_ident(&index.expr) else {
            return Err(Error::new(
                index.expr.span(),
                "generic wrapping operands must name input bindings",
            ));
        };
        if bindings
            .iter()
            .find(|binding| binding.ident == *name)
            .is_none_or(|binding| binding.access != BindingAccess::ReadOnly)
        {
            return Err(Error::new(
                index.expr.span(),
                "generic wrapping operands must use read-only inputs",
            ));
        }
        input_names.push(name.clone());
    }
    let Expr::Index(output) = assignment.left.as_ref() else {
        return Err(Error::new(
            assignment.left.span(),
            "generic wrapping destination must be indexed",
        ));
    };
    validate_invocation_index(&output.index, invocation)?;
    let Some(name) = expr_ident(&output.expr) else {
        return Err(Error::new(
            output.expr.span(),
            "generic wrapping destination must name the output binding",
        ));
    };
    if bindings
        .iter()
        .find(|binding| binding.ident == *name)
        .is_none_or(|binding| binding.access != BindingAccess::ReadWrite)
    {
        return Err(Error::new(
            output.expr.span(),
            "generic wrapping destination must use the writable output",
        ));
    }
    if input_names[0] == input_names[1] || input_names.iter().any(|input| input == name) {
        return Err(Error::new(
            assignment.span(),
            "generic wrapping maps require two distinct read-only inputs and a distinct writable output",
        ));
    }
    Ok(())
}

fn validate_generic_identity(
    assignment: &ExprAssign,
    bindings: &[BindingSpec],
    invocation: &Ident,
    scalar_type: &Ident,
) -> Result<(), Error> {
    if bindings.len() != 2
        || bindings.iter().any(|binding| {
            binding.scalar != ScalarKind::Generic
                || binding.generic_scalar.as_ref() != Some(scalar_type)
        })
    {
        return Err(Error::new(
            assignment.span(),
            "generic `T: PcuScalar` kernels require exactly `input: &[T]` and `output: &mut [T]`",
        ));
    }
    let Expr::Index(source_index) = assignment.right.as_ref() else {
        return Err(Error::new(
            assignment.right.span(),
            "generic `T: PcuScalar` kernels currently support only `output[id] = input[id]`",
        ));
    };
    let Some(source) = expr_ident(&source_index.expr) else {
        return Err(Error::new(
            source_index.expr.span(),
            "identity source must be a binding",
        ));
    };
    validate_invocation_index(&source_index.index, invocation)?;
    let Expr::Index(target_index) = assignment.left.as_ref() else {
        unreachable!("validated generic output binding is indexed")
    };
    let Some(target) = expr_ident(&target_index.expr) else {
        unreachable!("validated generic output binding is named")
    };
    validate_invocation_index(&target_index.index, invocation)?;
    if source == target
        || bindings
            .iter()
            .find(|binding| binding.ident == *source)
            .is_none_or(|binding| binding.access != BindingAccess::ReadOnly)
        || bindings
            .iter()
            .find(|binding| binding.ident == *target)
            .is_none_or(|binding| binding.access != BindingAccess::ReadWrite)
    {
        return Err(Error::new(
            assignment.span(),
            "generic identity copy requires a distinct read-only input and writable output",
        ));
    }
    Ok(())
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
    generic_scalar: Option<&Ident>,
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
        let (access, scalar, binding_generic) = parse_binding_type(&input.ty, generic_scalar)?;
        let binding = u32::try_from(bindings.len())
            .map_err(|_| Error::new(input.span(), "too many PCU bindings for this macro"))?;
        bindings.push(BindingSpec {
            ident: pat.ident.clone(),
            access,
            binding,
            scalar,
            generic_scalar: binding_generic,
        });
    }
    Ok(bindings)
}

fn parse_binding_type(
    ty: &Type,
    generic_scalar: Option<&Ident>,
) -> Result<(BindingAccess, ScalarKind, Option<Ident>), Error> {
    let Type::Reference(reference) = ty else {
        return Err(Error::new(
            ty.span(),
            "PCU binding types must be slices of f32, f64, u8, u16, u32, u64, i8, i16, i32, or i64",
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
            "PCU dispatch resources must be slices of f32, f64, u8, u16, u32, u64, i8, i16, i32, or i64",
        ));
    };
    let Type::Path(element) = slice.elem.as_ref() else {
        return Err(Error::new(
            slice.elem.span(),
            "this PCU dispatch macro currently supports only f32, f64, u8, u16, u32, u64, i8, i16, i32, and i64 elements",
        ));
    };
    if let Some(generic) = generic_scalar
        && element.path.is_ident(generic)
    {
        return Ok((
            if reference.mutability.is_some() {
                BindingAccess::ReadWrite
            } else {
                BindingAccess::ReadOnly
            },
            ScalarKind::Generic,
            Some(generic.clone()),
        ));
    }
    let scalar = if element.path.is_ident("f32") {
        ScalarKind::F32
    } else if element.path.is_ident("f64") {
        ScalarKind::F64
    } else if element.path.is_ident("u8") {
        ScalarKind::U8
    } else if element.path.is_ident("u16") {
        ScalarKind::U16
    } else if element.path.is_ident("u32") {
        ScalarKind::U32
    } else if element.path.is_ident("u64") {
        ScalarKind::U64
    } else if element.path.is_ident("i8") {
        ScalarKind::I8
    } else if element.path.is_ident("i16") {
        ScalarKind::I16
    } else if element.path.is_ident("i32") {
        ScalarKind::I32
    } else if element.path.is_ident("i64") {
        ScalarKind::I64
    } else {
        return Err(Error::new(
            element.span(),
            "this PCU dispatch macro currently supports only f32, f64, u8, u16, u32, u64, i8, i16, i32, and i64 elements",
        ));
    };
    Ok((
        if reference.mutability.is_some() {
            BindingAccess::ReadWrite
        } else {
            BindingAccess::ReadOnly
        },
        scalar,
        None,
    ))
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
            "PCU assignment target must be a mutable f32, u32, u64, i32, or i64 slice binding",
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
    if let Some(generic) = &binding.generic_scalar {
        return quote! {
            #pcu::PcuBinding::scalar::<#generic>(
                ::core::option::Option::Some(#name),
                0,
                #slot,
                #pcu::PcuBindingStorageClass::Storage,
                #access,
            )
        };
    }
    let scalar = binding.scalar.rust_type();
    quote! {
        #pcu::PcuBinding::scalar::<#scalar>(
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
    fn lowers_supported_indexed_f64_map() {
        let function = syn::parse_str::<ItemFn>("fn kernel(input: &[f64], rhs: &[f64], output: &mut [f64]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] + rhs[invocation]; }")
            .expect("f64 function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64").expect("arguments parse");
        let generated = expand_pcu_dispatch(args, &function)
            .expect("f64 map lowers")
            .to_string();
        assert!(generated.contains("PcuValueType :: f64"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
    }

    fn expand_u32(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(input: &[u32], rhs: &[u32], output: &mut [u32]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    fn expand_u32_div_rem(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(a: &[u32], b: &[u32], quotient: &mut [u32], remainder: &mut [u32]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    #[test]
    fn lowers_checked_u32_div_rem_direct_and_grid_stride() {
        let direct = expand_u32_div_rem(
            "let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id], b[id]); quotient[id] = q; remainder[id] = r;",
        )
        .expect("direct checked DivRem lowers")
        .to_string();
        assert!(
            direct.contains("PcuDispatchDataOp :: CheckedDivRem"),
            "{direct}"
        );
        assert!(direct.contains("PcuIntegerDivFlags :: CHECKED"), "{direct}");
        assert!(direct.contains("PcuValueType :: u32"), "{direct}");
        assert_eq!(
            direct.matches("PcuDispatchDataOp :: BindingStore").count(),
            2
        );

        let grid = expand_u32_div_rem(
            "let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < 64 { let (q, r) = pcu::checked_div_rem(a[id], b[id]); quotient[id] = q; remainder[id] = r; id += stride; }",
        );
        let grid = grid.expect("grid-stride checked DivRem lowers").to_string();
        assert!(grid.contains("PcuDispatchOp :: GridStrideLoop"), "{grid}");
        assert!(
            grid.contains("PcuDispatchDataOp :: CheckedDivRem"),
            "{grid}"
        );
        assert_eq!(grid.matches("PcuDispatchDataOp :: BindingStore").count(), 2);
    }

    #[test]
    fn rejects_invalid_checked_u32_div_rem_shapes_and_plain_operators() {
        for body in [
            "let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id], b[id]); quotient[id] = q; quotient[id] = r;",
            "let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id], b[id]); quotient[id] = q; remainder[id] = q;",
            "let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id] / b[id], b[id]); quotient[id] = q; remainder[id] = r;",
            "let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id] % b[id], b[id]); quotient[id] = q; remainder[id] = r;",
            "let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id], b[id]); quotient[id] = q;",
        ] {
            assert!(expand_u32_div_rem(body).is_err(), "accepted `{body}`");
        }
        let mixed = syn::parse_str::<ItemFn>("fn kernel(a: &[u32], b: &[i32], quotient: &mut [u32], remainder: &mut [u32]) { let id = context.global_invocation_id; let (q, r) = pcu::checked_div_rem(a[id], b[id]); quotient[id] = q; remainder[id] = r; }").expect("mixed function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64").expect("arguments parse");
        assert!(expand_pcu_dispatch(args, &mixed).is_err());
    }

    fn expand_u16(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(input: &[u16], rhs: &[u16], output: &mut [u16]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    #[test]
    fn lowers_u16_wrapping_arithmetic_and_rejects_plain_operators() {
        for (method, op) in [
            ("wrapping_add", "Add"),
            ("wrapping_sub", "Sub"),
            ("wrapping_mul", "Mul"),
        ] {
            let body = format!(
                "let invocation = context.global_invocation_id; output[invocation] = input[invocation].{method}(rhs[invocation]);"
            );
            let generated = expand_u16(&body)
                .expect("u16 wrapping map lowers")
                .to_string();
            assert!(
                generated.contains(&format!("PcuDispatchAluOp :: {op}")),
                "{generated}"
            );
            assert!(generated.contains("PcuValueType :: u16"), "{generated}");
            assert!(
                generated.contains("PcuBinding :: scalar :: < u16 >"),
                "{generated}"
            );
        }
        assert!(expand_u16(
            "let invocation = context.global_invocation_id; output[invocation] = input[invocation] + rhs[invocation];"
        )
        .is_err());
    }

    #[test]
    fn lowers_u16_wrapping_chain_inside_grid_stride_loop() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<const N: usize>(input: &[u16], rhs: &[u16], output: &mut [u16]) { let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < N { output[id] = input[id].wrapping_add(rhs[id]).wrapping_mul(rhs[id]); id += stride; } }",
        )
        .expect("grid-stride function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 4").expect("arguments parse");
        let generated = expand_pcu_dispatch(args, &function)
            .expect("u16 grid-stride map lowers")
            .to_string();
        assert!(
            generated.contains("PcuDispatchOp :: GridStrideLoop"),
            "{generated}"
        );
        assert!(generated.contains("PcuValueType :: u16"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Mul"), "{generated}");
    }

    fn expand_u8(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(input: &[u8], rhs: &[u8], output: &mut [u8]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    #[test]
    fn lowers_u8_wrapping_arithmetic_and_rejects_plain_operators() {
        for (method, op) in [
            ("wrapping_add", "Add"),
            ("wrapping_sub", "Sub"),
            ("wrapping_mul", "Mul"),
        ] {
            let body = format!(
                "let invocation = context.global_invocation_id; output[invocation] = input[invocation].{method}(rhs[invocation]);"
            );
            let generated = expand_u8(&body)
                .expect("u8 wrapping map lowers")
                .to_string();
            assert!(
                generated.contains(&format!("PcuDispatchAluOp :: {op}")),
                "{generated}"
            );
            assert!(generated.contains("PcuValueType :: u8"), "{generated}");
            assert!(
                generated.contains("PcuBinding :: scalar :: < u8 >"),
                "{generated}"
            );
        }
        assert!(expand_u8(
            "let invocation = context.global_invocation_id; output[invocation] = input[invocation] + rhs[invocation];"
        )
        .is_err());
    }

    #[test]
    fn lowers_u8_wrapping_chain_inside_grid_stride_loop() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<const N: usize>(input: &[u8], rhs: &[u8], output: &mut [u8]) { let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < N { output[id] = input[id].wrapping_add(rhs[id]).wrapping_mul(rhs[id]); id += stride; } }",
        )
        .expect("grid-stride function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 4").expect("arguments parse");
        let generated = expand_pcu_dispatch(args, &function)
            .expect("u8 grid-stride map lowers")
            .to_string();
        assert!(
            generated.contains("PcuDispatchOp :: GridStrideLoop"),
            "{generated}"
        );
        assert!(generated.contains("PcuValueType :: u8"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Mul"), "{generated}");
    }

    fn expand_i8(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(input: &[i8], rhs: &[i8], output: &mut [i8]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    #[test]
    fn lowers_i8_wrapping_arithmetic_and_rejects_plain_operators() {
        for (method, op) in [
            ("wrapping_add", "Add"),
            ("wrapping_sub", "Sub"),
            ("wrapping_mul", "Mul"),
        ] {
            let body = format!(
                "let invocation = context.global_invocation_id; output[invocation] = input[invocation].{method}(rhs[invocation]);"
            );
            let generated = expand_i8(&body)
                .expect("i8 wrapping map lowers")
                .to_string();
            assert!(
                generated.contains(&format!("PcuDispatchAluOp :: {op}")),
                "{generated}"
            );
            assert!(generated.contains("PcuValueType :: i8"), "{generated}");
            assert!(
                generated.contains("PcuBinding :: scalar :: < i8 >"),
                "{generated}"
            );
        }
        assert!(expand_i8(
            "let invocation = context.global_invocation_id; output[invocation] = input[invocation] + rhs[invocation];"
        )
        .is_err());
    }

    #[test]
    fn lowers_i8_wrapping_chain_inside_grid_stride_loop() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<const N: usize>(input: &[i8], rhs: &[i8], output: &mut [i8]) { let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < N { output[id] = input[id].wrapping_add(rhs[id]).wrapping_mul(rhs[id]); id += stride; } }",
        )
        .expect("grid-stride function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 4").expect("arguments parse");
        let generated = expand_pcu_dispatch(args, &function)
            .expect("i8 grid-stride map lowers")
            .to_string();
        assert!(
            generated.contains("PcuDispatchOp :: GridStrideLoop"),
            "{generated}"
        );
        assert!(generated.contains("PcuValueType :: i8"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Mul"), "{generated}");
    }

    #[test]
    fn lowers_u32_wrapping_arithmetic_with_typed_alu_ops() {
        for (method, op) in [
            ("wrapping_add", "Add"),
            ("wrapping_sub", "Sub"),
            ("wrapping_mul", "Mul"),
        ] {
            let body = format!(
                "let invocation = context.global_invocation_id; output[invocation] = (input[invocation]).{method}((rhs[invocation]));"
            );
            let generated = expand_u32(&body)
                .expect("supported wrapping map lowers")
                .to_string();
            assert!(
                generated.contains(&format!("PcuDispatchAluOp :: {op}")),
                "{generated}"
            );
            assert!(generated.contains("PcuValueType :: u32"), "{generated}");
        }
    }

    #[test]
    fn lowers_chained_u32_wrapping_arithmetic_in_ssa_order() {
        let generated = expand_u32(
            "let invocation = context.global_invocation_id; output[invocation] = (input[invocation]).wrapping_add(rhs[invocation]).wrapping_mul(rhs[invocation]);",
        )
        .expect("chained wrapping arithmetic lowers")
        .to_string();
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Mul"), "{generated}");
        assert!(
            generated.contains("lhs : :: fusion_pcu :: PcuDispatchValueId (3u16)"),
            "{generated}"
        );
        assert!(
            generated.contains("value : :: fusion_pcu :: PcuDispatchValueId (5u16)"),
            "{generated}"
        );
    }

    fn expand_u64(body: &str) -> Result<proc_macro2::TokenStream, syn::Error> {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn kernel(input: &[u64], rhs: &[u64], output: &mut [u64]) {{ {body} }}"
        ))
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        expand_pcu_dispatch(args, &function)
    }

    #[test]
    fn lowers_u64_wrapping_arithmetic_with_typed_alu_ops() {
        for (method, op) in [
            ("wrapping_add", "Add"),
            ("wrapping_sub", "Sub"),
            ("wrapping_mul", "Mul"),
        ] {
            let body = format!(
                "let invocation = context.global_invocation_id; output[invocation] = (input[invocation]).{method}(rhs[invocation]);"
            );
            let generated = expand_u64(&body)
                .expect("u64 wrapping map lowers")
                .to_string();
            assert!(
                generated.contains(&format!("PcuDispatchAluOp :: {op}")),
                "{generated}"
            );
            assert!(generated.contains("PcuValueType :: u64"), "{generated}");
            assert!(
                generated.contains("PcuBinding :: scalar :: < u64 >"),
                "{generated}"
            );
        }
    }

    #[test]
    fn lowers_i32_wrapping_arithmetic_with_typed_alu_ops() {
        let function = syn::parse_str::<ItemFn>("fn kernel(input: &[i32], rhs: &[i32], output: &mut [i32]) { let invocation = context.global_invocation_id; output[invocation] = (input[invocation]).wrapping_add(rhs[invocation]); }").expect("function parses");
        let generated = expand_pcu_dispatch(
            PcuDispatchArgs {
                kernel_id: 1,
                invocations: syn::parse_quote!(8),
                crate_path: syn::parse_quote!(::fusion_pcu),
            },
            &function,
        )
        .expect("i32 wrapping map lowers")
        .to_string();
        assert!(generated.contains("PcuValueType :: i32 ()"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
    }

    #[test]
    fn lowers_i16_wrapping_arithmetic_with_typed_alu_ops() {
        let function = syn::parse_str::<ItemFn>("fn kernel(input: &[i16], rhs: &[i16], output: &mut [i16]) { let invocation = context.global_invocation_id; output[invocation] = (input[invocation]).wrapping_add(rhs[invocation]); }").expect("function parses");
        let generated = expand_pcu_dispatch(
            PcuDispatchArgs {
                kernel_id: 1,
                invocations: syn::parse_quote!(8),
                crate_path: syn::parse_quote!(::fusion_pcu),
            },
            &function,
        )
        .expect("i16 wrapping map lowers")
        .to_string();
        assert!(generated.contains("PcuValueType :: i16 ()"), "{generated}");
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
    }

    #[test]
    fn lowers_i64_wrapping_arithmetic_with_typed_alu_ops() {
        let function = syn::parse_str::<ItemFn>("fn kernel(input: &[i64], rhs: &[i64], output: &mut [i64]) { let invocation = context.global_invocation_id; output[invocation] = (input[invocation]).wrapping_add(rhs[invocation]); }")
            .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default crate path arguments parse");
        let generated = expand_pcu_dispatch(args, &function).expect("i64 wrapping map lowers");
        let generated = generated.to_string();
        assert!(generated.contains("PcuValueType :: i64"), "{generated}");
        assert!(
            generated.contains("PcuBinding :: scalar :: < i64 >"),
            "{generated}"
        );
        assert!(generated.contains("PcuDispatchAluOp :: Add"), "{generated}");
    }

    #[test]
    fn rejects_plain_u32_arithmetic_as_overflow_mode_dependent() {
        for operator in ["+", "-", "*"] {
            let body = format!(
                "let invocation = context.global_invocation_id; output[invocation] = input[invocation] {operator} rhs[invocation];"
            );
            let error = expand_u32(&body).expect_err("plain u32 arithmetic is rejected");
            assert!(error.to_string().contains("matching f32 or f64 operands"));
        }
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
    fn rejects_generic_scalar_arithmetic_outside_identity_profile() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] + input[invocation]; }",
        )
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 8").expect("attribute parses");
        let error =
            expand_pcu_dispatch(args, &function).expect_err("generic arithmetic is unsupported");
        let message = error.to_string();
        assert!(message.contains("currently support only `output[id] = input[id]`"));
    }

    #[test]
    fn lowers_generic_scalar_grid_stride_through_typed_builder() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) { let mut id = context.global_invocation_id; let stride = context.invocation_count; while id < N { output[id] = input[id]; id += stride; } }",
        )
        .expect("function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 4").expect("attribute parses");
        let tokens = expand_pcu_dispatch(args, &function)
            .expect("the typed identity builder supports bounded grid stride");
        assert!(tokens.to_string().contains("build_grid_stride"));
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
