#[path = "fusion-pcu-macros/checked_div_rem.rs"]
mod checked_div_rem;
#[path = "fusion-pcu-macros/hosted.rs"]
mod hosted;
#[path = "fusion-pcu-macros/owned.rs"]
mod owned;
#[path = "fusion-pcu-macros/prepared.rs"]
mod prepared;
#[path = "fusion-pcu-macros/shape.rs"]
mod shape;

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
#[rustfmt::skip]
use quote::{
    format_ident,
    quote,
    ToTokens,
};
#[rustfmt::skip]
use syn::parse::{
    Parse,
    ParseStream,
    Parser,
};
use syn::spanned::Spanned;
use syn::visit_mut::{self, VisitMut};
#[rustfmt::skip]
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
    underflow_flag: Option<PcuOwnedFlag>,
    clamp_range: bool,
    numerical_mode: Option<bool>,
}

struct PcuScalarHelperArgs {
    crate_path: Path,
    underflow_flag: Option<PcuOwnedFlag>,
    clamp_range: bool,
    numerical_mode: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PcuOwnedFlag {
    IeeeUnderflow,
    AllowGradualUnderflow,
    RejectSubnormalResult,
}

impl Parse for PcuScalarHelperArgs {
    fn parse(input: ParseStream<'_>) -> Result<Self, Error> {
        let mut crate_path = None;
        let mut underflow_flag = None;
        let mut clamp_range = false;
        let mut numerical_mode = None;
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "flag" {
                let content;
                syn::parenthesized!(content in input);
                let flag: Ident = content.parse()?;
                if !content.is_empty() {
                    return Err(content.error("a `pcu` float flag takes exactly one name"));
                }
                let parsed = match flag.to_string().as_str() {
                    "strict" | "non_strict" => {
                        let strict = flag == "strict";
                        if let Some(previous) = numerical_mode {
                            return Err(Error::new(
                                flag.span(),
                                if previous == strict {
                                    "duplicate `pcu` numerical mode flag"
                                } else {
                                    "conflicting `pcu` numerical mode flags"
                                },
                            ));
                        }
                        numerical_mode = Some(strict);
                        if input.is_empty() {
                            break;
                        }
                        let _: Token![,] = input.parse()?;
                        continue;
                    }
                    "clamp_range" => {
                        if clamp_range {
                            return Err(Error::new(flag.span(), "duplicate `pcu` clamp flag"));
                        }
                        clamp_range = true;
                        if input.is_empty() {
                            break;
                        }
                        let _: Token![,] = input.parse()?;
                        continue;
                    }
                    "ieee_underflow" => PcuOwnedFlag::IeeeUnderflow,
                    "allow_gradual_underflow" => PcuOwnedFlag::AllowGradualUnderflow,
                    "reject_subnormal_result" => PcuOwnedFlag::RejectSubnormalResult,
                    _ => return Err(Error::new(flag.span(), "unknown `pcu` float flag")),
                };
                if let Some(previous) = underflow_flag {
                    return Err(Error::new(
                        flag.span(),
                        if previous == parsed {
                            "duplicate `pcu` float flag"
                        } else {
                            "conflicting `pcu` float flags; choose one underflow policy"
                        },
                    ));
                }
                underflow_flag = Some(parsed);
            } else {
                let _: Token![=] = input.parse()?;
                if key != "crate_path" {
                    return Err(Error::new(
                        key.span(),
                        "`#[pcu]` helpers accept `crate_path = <path>` and one `flag(...)`",
                    ));
                }
                if crate_path.is_some() {
                    return Err(Error::new(key.span(), "duplicate `crate_path` argument"));
                }
                crate_path = Some(input.parse()?);
            }
            if input.is_empty() {
                break;
            }
            let _: Token![,] = input.parse()?;
        }
        Ok(Self {
            crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::fusion_pcu)),
            underflow_flag,
            clamp_range,
            numerical_mode,
        })
    }
}

// Argument parsing keeps independent numerical, range, and underflow diagnostics together.
impl Parse for PcuDispatchArgs {
    #[allow(clippy::too_many_lines)]
    fn parse(input: ParseStream<'_>) -> Result<Self, Error> {
        let mut kernel_id = None;
        let mut invocations = None;
        let mut crate_path = None;
        let mut underflow_flag = None;
        let mut clamp_range = false;
        let mut numerical_mode = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "flag" {
                let content;
                syn::parenthesized!(content in input);
                let flag: Ident = content.parse()?;
                if !content.is_empty() {
                    return Err(content.error("a `pcu` float flag takes exactly one name"));
                }
                let parsed = match flag.to_string().as_str() {
                    "strict" | "non_strict" => {
                        let strict = flag == "strict";
                        if let Some(previous) = numerical_mode {
                            return Err(Error::new(
                                flag.span(),
                                if previous == strict {
                                    "duplicate `pcu` numerical mode flag"
                                } else {
                                    "conflicting `pcu` numerical mode flags"
                                },
                            ));
                        }
                        numerical_mode = Some(strict);
                        if input.is_empty() {
                            break;
                        }
                        let _: Token![,] = input.parse()?;
                        continue;
                    }
                    "clamp_range" => {
                        if clamp_range {
                            return Err(Error::new(flag.span(), "duplicate `pcu` clamp flag"));
                        }
                        clamp_range = true;
                        if input.is_empty() {
                            break;
                        }
                        let _: Token![,] = input.parse()?;
                        continue;
                    }
                    "ieee_underflow" => PcuOwnedFlag::IeeeUnderflow,
                    "allow_gradual_underflow" => PcuOwnedFlag::AllowGradualUnderflow,
                    "reject_subnormal_result" => PcuOwnedFlag::RejectSubnormalResult,
                    _ => return Err(Error::new(flag.span(), "unknown `pcu` float flag")),
                };
                if let Some(previous) = underflow_flag {
                    return Err(Error::new(
                        flag.span(),
                        if previous == parsed {
                            "duplicate `pcu` float flag"
                        } else {
                            "conflicting `pcu` float flags; choose one underflow policy"
                        },
                    ));
                }
                underflow_flag = Some(parsed);
                if input.is_empty() {
                    break;
                }
                let _: Token![,] = input.parse()?;
                continue;
            }
            if key == "invocations" && input.peek(Token![:]) {
                let _: Token![:] = input.parse()?;
            } else {
                let _: Token![=] = input.parse()?;
            }
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
            kernel_id: kernel_id.unwrap_or(1),
            invocations,
            crate_path: crate_path.unwrap_or_else(|| syn::parse_quote!(::fusion_pcu)),
            underflow_flag,
            clamp_range,
            numerical_mode,
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
    scalar_reference: bool,
    matrix: Option<shape::FixedMatrixShape>,
}

#[derive(Clone)]
struct MatrixLocals {
    row: Ident,
    column: Ident,
    columns: Expr,
    invocation: Ident,
    stride: Option<Ident>,
}

struct PcuHelper {
    ident: Ident,
    parameters: Vec<Ident>,
    scalar: ScalarKind,
    body: Expr,
}

fn transparent_type(ty: &Type) -> &Type {
    match ty {
        Type::Group(group) => transparent_type(&group.elem),
        Type::Paren(paren) => transparent_type(&paren.elem),
        _ => ty,
    }
}

fn scalar_kind(ty: &Type) -> Option<ScalarKind> {
    match transparent_type(ty) {
        Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32") => {
            Some(ScalarKind::F32)
        }
        Type::Path(path) if path.qself.is_none() && path.path.is_ident("f64") => {
            Some(ScalarKind::F64)
        }
        _ => None,
    }
}

struct ExprEmitter<'a> {
    bindings: &'a [BindingSpec],
    invocation_ident: &'a Ident,
    crate_path: &'a Path,
    grid_stride: bool,
    expected_scalar: ScalarKind,
    values: Vec<(Ident, u16, ScalarKind)>,
    next_value: u16,
    ops: Vec<TokenStream2>,
}

/// Emits ordered runtime builder statements for expressions that cross a generated helper
/// companion. The regular no-helper path stays const-sized and proc-macro-lowered.
struct RuntimeExprEmitter<'a> {
    bindings: &'a [BindingSpec],
    invocation_ident: &'a Ident,
    crate_path: &'a Path,
    grid_stride: bool,
    expected_scalar: ScalarKind,
    context_is_owned: bool,
    values: Vec<(Ident, TokenStream2, ScalarKind)>,
    statements: Vec<TokenStream2>,
    next_local: usize,
}

impl<'a> RuntimeExprEmitter<'a> {
    const fn new(
        bindings: &'a [BindingSpec],
        invocation_ident: &'a Ident,
        crate_path: &'a Path,
        grid_stride: bool,
        expected_scalar: ScalarKind,
        context_is_owned: bool,
    ) -> Self {
        Self {
            bindings,
            invocation_ident,
            crate_path,
            grid_stride,
            expected_scalar,
            context_is_owned,
            values: Vec::new(),
            statements: Vec::new(),
            next_local: 0,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn emit_expr(&mut self, expr: &Expr) -> Result<(TokenStream2, ScalarKind), Error> {
        match expr {
            Expr::Cast(cast) => {
                let target_kind = match transparent_type(&cast.ty) {
                    Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32") => {
                        ScalarKind::F32
                    }
                    Type::Path(path) if path.qself.is_none() && path.path.is_ident("f64") => {
                        ScalarKind::F64
                    }
                    _ => {
                        return Err(Error::new(
                            cast.ty.span(),
                            "PCU checked casts support only explicit `f64 as f32` and `f32 as f64` conversions",
                        ));
                    }
                };
                if target_kind == ScalarKind::F64 && !self.has_f32_source_evidence(&cast.expr) {
                    return Err(Error::new(
                        cast.expr.span(),
                        "PCU checked cast to f64 requires an explicitly typed f32 source; unsuffixed float literals default to f64",
                    ));
                }
                let (source_kind, conversion_method) = match target_kind {
                    ScalarKind::F32 => (ScalarKind::F64, quote! { checked_f64_to_f32_value }),
                    ScalarKind::F64 => (ScalarKind::F32, quote! { checked_f32_to_f64_value }),
                    _ => unreachable!("cast target kinds are concrete floats"),
                };
                // Infer an unsuffixed literal from the explicit conversion direction, then
                // restore the enclosing result kind before returning from this cast node.
                let enclosing_scalar = self.expected_scalar;
                self.expected_scalar = source_kind;
                let source_result = self.emit_expr(&cast.expr);
                self.expected_scalar = enclosing_scalar;
                let (source, actual_source_kind) = source_result?;
                if actual_source_kind != source_kind {
                    let expectation = match target_kind {
                        ScalarKind::F32 => "PCU checked cast to f32 requires a concrete f64 source",
                        ScalarKind::F64 => "PCU checked cast to f64 requires a concrete f32 source",
                        _ => unreachable!("cast target kinds are concrete floats"),
                    };
                    return Err(Error::new(cast.expr.span(), expectation));
                }
                let result = self.fresh_local();
                self.statements.push(quote! {
                    let #result = __pcu_context.#conversion_method(#source)?;
                });
                Ok((quote! { #result }, target_kind))
            }
            Expr::Binary(binary) => {
                let (lhs, lhs_type) = self.emit_expr(&binary.left)?;
                let (rhs, rhs_type) = self.emit_expr(&binary.right)?;
                if lhs_type != rhs_type || !matches!(lhs_type, ScalarKind::F32 | ScalarKind::F64) {
                    return Err(Error::new(
                        binary.span(),
                        "PCU floating arithmetic requires matching f32 or f64 operands",
                    ));
                }
                let op = match &binary.op {
                    BinOp::Add(_) => quote! { Add },
                    BinOp::Sub(_) => quote! { Sub },
                    BinOp::Mul(_) => quote! { Mul },
                    BinOp::Div(_) => quote! { Div },
                    _ => {
                        return Err(Error::new(
                            binary.op.span(),
                            "unsupported PCU binary operator; supported operators are + - * /",
                        ));
                    }
                };
                let result = self.fresh_local();
                let pcu = self.crate_path;
                self.statements.push(quote! {
                    let #result = __pcu_context.checked_binary_value(#pcu::PcuDispatchFloatBinaryOp::#op, #lhs, #rhs)?;
                });
                Ok((quote! { #result }, lhs_type))
            }
            Expr::Index(index) => {
                let matrix_binding =
                    shape::matrix_base(index)
                        .and_then(expr_ident)
                        .and_then(|ident| {
                            self.bindings
                                .iter()
                                .find(|binding| binding.ident == *ident && binding.matrix.is_some())
                        });
                let base = if let Some(binding) = matrix_binding {
                    let shape = binding.matrix.as_ref().expect("matrix binding checked");
                    if !shape::is_canonical_matrix_index(
                        index,
                        self.invocation_ident,
                        &shape.columns,
                    ) {
                        return Err(Error::new(
                            index.span(),
                            "matrix access must use `matrix[id / C][id % C]`",
                        ));
                    }
                    shape::matrix_base(index).expect("canonical matrix index has a base")
                } else {
                    validate_invocation_index(&index.index, self.invocation_ident)?;
                    &index.expr
                };
                let Some(binding_ident) = expr_ident(base) else {
                    return Err(Error::new(
                        base.span(),
                        "PCU binding load must use `binding[invocation]`",
                    ));
                };
                let (slot, scalar) = {
                    let binding = self.binding(binding_ident, BindingAccess::ReadOnly)?;
                    (binding.binding, binding.scalar)
                };
                if !matches!(scalar, ScalarKind::F32 | ScalarKind::F64) {
                    return Err(Error::new(
                        index.span(),
                        "floating scalar helper calls support only f32 or f64 bindings",
                    ));
                }
                let result = self.fresh_local();
                let index_kind = if self.grid_stride {
                    quote! { GridStrideId }
                } else {
                    quote! { InvocationId }
                };
                let pcu = self.crate_path;
                let scalar_ty = scalar.rust_type();
                self.statements.push(quote! {
                    let #result = __pcu_context.load_value::<#scalar_ty>(
                        #pcu::PcuBindingRef::new(0, #slot),
                        #pcu::PcuDispatchIndex::#index_kind,
                    )?;
                });
                Ok((quote! { #result }, scalar))
            }
            Expr::Lit(literal) => {
                let Lit::Float(float) = &literal.lit else {
                    return Err(Error::new(
                        literal.span(),
                        "PCU constants support f32 float literals in this profile",
                    ));
                };
                let (scalar, bits) = parse_float_bits(float, self.expected_scalar)?;
                let result = self.fresh_local();
                let method = if scalar == ScalarKind::F32 {
                    quote! { constant_f32_value }
                } else {
                    quote! { constant_f64_value }
                };
                let bits = if scalar == ScalarKind::F32 {
                    quote! { u32::try_from(#bits).expect("f32 bits fit u32") }
                } else {
                    quote! { #bits }
                };
                self.statements.push(quote! {
                    let #result = __pcu_context.#method(#bits)?;
                });
                Ok((quote! { #result }, scalar))
            }
            Expr::Paren(paren) => self.emit_expr(&paren.expr),
            Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                let Expr::Path(path) = unary.expr.as_ref() else {
                    return Err(Error::new(
                        unary.expr.span(),
                        "PCU scalar dereference must refer directly to a read-only scalar binding",
                    ));
                };
                let Some(ident) = path.path.get_ident() else {
                    return Err(Error::new(
                        path.span(),
                        "PCU scalar dereference must refer directly to a read-only scalar binding",
                    ));
                };
                if !self
                    .binding(ident, BindingAccess::ReadOnly)?
                    .scalar_reference
                {
                    return Err(Error::new(
                        path.span(),
                        "PCU dereference is supported only for read-only scalar bindings",
                    ));
                }
                self.emit_expr(&unary.expr)
            }
            Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let ident = &path.path.segments[0].ident;
                if let Some(value) = self
                    .values
                    .iter()
                    .rev()
                    .find(|(name, _, _)| name == ident)
                    .map(|(_, value, scalar)| (value.clone(), *scalar))
                {
                    return Ok(value);
                }
                let (scalar_reference, slot, scalar) = {
                    let binding = self.binding(ident, BindingAccess::ReadOnly)?;
                    (binding.scalar_reference, binding.binding, binding.scalar)
                };
                if !scalar_reference || !matches!(scalar, ScalarKind::F32 | ScalarKind::F64) {
                    return Err(Error::new(
                        path.span(),
                        "PCU helper expressions may only reference scalar parameters or f32/f64 scalar bindings",
                    ));
                }
                let result = self.fresh_local();
                let pcu = self.crate_path;
                let scalar_ty = scalar.rust_type();
                self.statements.push(quote! {
                    let #result = __pcu_context.load_value::<#scalar_ty>(
                        #pcu::PcuBindingRef::new(0, #slot),
                        #pcu::PcuDispatchIndex::BindingElementZero,
                    )?;
                });
                Ok((quote! { #result }, scalar))
            }
            Expr::Call(call) => {
                let Expr::Path(path) = call.func.as_ref() else {
                    return Err(Error::new(
                        call.func.span(),
                        "PCU helper calls must name a `#[pcu]` scalar function",
                    ));
                };
                if path.qself.is_some() {
                    return Err(Error::new(
                        path.span(),
                        "PCU helper calls must use a Rust-resolved function path",
                    ));
                }
                let mut companion = path.path.clone();
                if !self.context_is_owned {
                    companion = rebase_companion_path(companion);
                }
                let mut args = Vec::with_capacity(call.args.len());
                for arg in &call.args {
                    args.push(self.emit_expr(arg)?);
                }
                let known_kinds = args
                    .iter()
                    .map(|(_, kind)| *kind)
                    .filter(|kind| matches!(kind, ScalarKind::F32 | ScalarKind::F64))
                    .collect::<Vec<_>>();
                if known_kinds.windows(2).any(|kinds| kinds[0] != kinds[1]) {
                    return Err(Error::new(
                        call.span(),
                        "PCU helper calls cannot mix f32 and f64 arguments",
                    ));
                }
                let values = args.iter().map(|(value, _)| value).collect::<Vec<_>>();
                let result = self.fresh_local();
                let context = if self.context_is_owned {
                    quote! { &mut __pcu_context }
                } else {
                    quote! { __pcu_context }
                };
                self.statements.push(quote! {
                    let #result = #companion::__pcu_lower(#context, [#(#values),*])?;
                });
                let result_kind = known_kinds.first().copied().unwrap_or(self.expected_scalar);
                if result_kind != self.expected_scalar {
                    return Err(Error::new(
                        call.span(),
                        "PCU helper call scalar type does not match the enclosing scalar profile",
                    ));
                }
                Ok((quote! { #result }, result_kind))
            }
            _ => Err(Error::new(
                unsupported_expression_span(expr),
                "unsupported PCU expression; supported subset is binding[index], f32/f64 literals, parentheses, arithmetic, and #[pcu] scalar helper calls",
            )),
        }
    }

    fn has_f32_source_evidence(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Lit(literal) => matches!(
                &literal.lit,
                Lit::Float(float) if float.suffix() == "f32"
            ),
            Expr::Paren(paren) => self.has_f32_source_evidence(&paren.expr),
            Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                self.has_f32_source_evidence(&unary.expr)
            }
            Expr::Binary(binary) => {
                self.has_f32_source_evidence(&binary.left)
                    || self.has_f32_source_evidence(&binary.right)
            }
            Expr::Cast(cast) => matches!(
                transparent_type(&cast.ty),
                Type::Path(path) if path.qself.is_none() && path.path.is_ident("f32")
            ),
            Expr::Index(index) => {
                let base = shape::matrix_base(index).unwrap_or(&index.expr);
                expr_ident(base)
                    .and_then(|ident| self.bindings.iter().find(|binding| binding.ident == *ident))
                    .is_some_and(|binding| binding.scalar == ScalarKind::F32)
            }
            Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let ident = &path.path.segments[0].ident;
                self.values
                    .iter()
                    .rev()
                    .find(|(name, _, _)| name == ident)
                    .is_some_and(|(_, _, kind)| *kind == ScalarKind::F32)
                    || self
                        .bindings
                        .iter()
                        .find(|binding| binding.ident == *ident)
                        .is_some_and(|binding| {
                            binding.scalar_reference && binding.scalar == ScalarKind::F32
                        })
            }
            Expr::Call(call) => call
                .args
                .iter()
                .any(|argument| self.has_f32_source_evidence(argument)),
            _ => false,
        }
    }

    fn binding(&self, ident: &Ident, required: BindingAccess) -> Result<&BindingSpec, Error> {
        let Some(binding) = self.bindings.iter().find(|binding| binding.ident == *ident) else {
            return Err(Error::new(ident.span(), "unknown PCU binding"));
        };
        if required == BindingAccess::ReadWrite && binding.access != BindingAccess::ReadWrite {
            return Err(Error::new(
                ident.span(),
                "PCU binding access does not match the expression context",
            ));
        }
        Ok(binding)
    }

    fn fresh_local(&mut self) -> Ident {
        let ident = format_ident!("__pcu_helper_value_{}", self.next_local);
        self.next_local += 1;
        ident
    }
}

fn rebase_companion_path(mut path: Path) -> Path {
    if let Some(first) = path.segments.first_mut() {
        match first.ident.to_string().as_str() {
            "self" => first.ident = format_ident!("super"),
            "super" => path.segments.insert(0, syn::parse_quote!(super)),
            _ => {}
        }
    }
    path
}

impl<'a> ExprEmitter<'a> {
    const fn new(
        bindings: &'a [BindingSpec],
        invocation_ident: &'a Ident,
        crate_path: &'a Path,
        grid_stride: bool,
        expected_scalar: ScalarKind,
    ) -> Self {
        Self {
            bindings,
            invocation_ident,
            crate_path,
            grid_stride,
            expected_scalar,
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
            Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
                let Expr::Path(path) = unary.expr.as_ref() else {
                    return Err(Error::new(
                        unary.expr.span(),
                        "PCU scalar dereference must refer directly to a read-only scalar binding",
                    ));
                };
                let Some(ident) = path.path.get_ident() else {
                    return Err(Error::new(
                        path.span(),
                        "PCU scalar dereference must refer directly to a read-only scalar binding",
                    ));
                };
                if !self
                    .binding(ident, BindingAccess::ReadOnly)?
                    .scalar_reference
                {
                    return Err(Error::new(
                        path.span(),
                        "PCU dereference is supported only for read-only scalar bindings",
                    ));
                }
                self.emit_expr(&unary.expr)
            }
            Expr::MethodCall(call) => self.emit_wrapping_method(call),
            Expr::Call(call) => Err(Error::new(
                call.func.span(),
                "PCU helper calls require `#[pcu]` scalar helpers and `#[pcu(...)]` kernels",
            )),
            Expr::Path(path) if path.qself.is_none() && path.path.segments.len() == 1 => {
                let ident = &path.path.segments[0].ident;
                if let Some(value) = self
                    .values
                    .iter()
                    .rev()
                    .find(|(name, _, _)| name == ident)
                    .map(|(_, value, scalar)| (*value, *scalar))
                {
                    return Ok(value);
                }
                let (scalar_reference, slot, scalar) = {
                    let binding = self.binding(ident, BindingAccess::ReadOnly)?;
                    (binding.scalar_reference, binding.binding, binding.scalar)
                };
                if !scalar_reference {
                    return Err(Error::new(
                        path.span(),
                        "PCU expressions may only use scalar bindings as bare names",
                    ));
                }
                if !matches!(scalar, ScalarKind::F32 | ScalarKind::F64) {
                    return Err(Error::new(
                        path.span(),
                        "read-only scalar parameters currently support only f32 and f64",
                    ));
                }
                let result = self.alloc_value(path.span())?;
                let pcu = self.crate_path;
                self.ops.push(quote! {
                    #pcu::PcuDispatchDataOp::BindingLoad {
                        result: #pcu::PcuDispatchValueId(#result),
                        binding: #pcu::PcuBindingRef::new(0, #slot),
                        index: #pcu::PcuDispatchIndex::BindingElementZero,
                    }
                });
                self.values.push((ident.clone(), result, scalar));
                Ok((result, scalar))
            }
            _ => Err(Error::new(
                unsupported_expression_span(expr),
                "unsupported PCU expression; supported subset is binding[index], f32/f64 literals, parentheses, and + - * /",
            )),
        }
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
                    "unsupported PCU expression; supported subset is binding[index], f32/f64 literals, parentheses, and + - * /",
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
        let (base, matrix) =
            shape::matrix_base(index).map_or((&*index.expr, false), |base| (base, true));
        let Some(binding_ident) = expr_ident(base) else {
            return Err(Error::new(
                base.span(),
                "PCU binding load must use `binding[invocation]` or the canonical `matrix[invocation / C][invocation % C]` form",
            ));
        };
        let binding = self.binding(binding_ident, BindingAccess::ReadOnly)?;
        if let Some(dimensions) = &binding.matrix {
            if !matrix
                || !shape::is_canonical_matrix_index(
                    index,
                    self.invocation_ident,
                    &dimensions.columns,
                )
            {
                return Err(Error::new(
                    index.span(),
                    "rank-two PCU binding loads must use `matrix[invocation / C][invocation % C]` with the declared column extent",
                ));
            }
        } else if matrix {
            return Err(Error::new(
                index.span(),
                "nested PCU indexing is supported only for fixed rank-two array bindings",
            ));
        } else {
            validate_invocation_index(&index.index, self.invocation_ident)?;
        }
        let slot = binding.binding;
        let scalar = binding.scalar;
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
        Ok((result, scalar))
    }

    fn emit_lit(&mut self, lit: &ExprLit) -> Result<(u16, ScalarKind), Error> {
        let Lit::Float(float) = &lit.lit else {
            return Err(Error::new(
                lit.span(),
                "PCU constants support floating-point literals in this profile",
            ));
        };
        let (scalar, bits) = parse_float_bits(float, self.expected_scalar)?;
        let result = self.alloc_value(lit.span())?;
        let pcu = self.crate_path;
        let value = if scalar == ScalarKind::F32 {
            let bits = u32::try_from(bits).expect("f32 bits fit u32");
            quote! { #pcu::PcuParameterValue::F32(#bits) }
        } else {
            quote! { #pcu::PcuParameterValue::F64(#bits) }
        };
        self.ops.push(quote! {
            #pcu::PcuDispatchDataOp::Constant {
                result: #pcu::PcuDispatchValueId(#result),
                value: #value,
            }
        });
        Ok((result, scalar))
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

/// Marks a scalar helper or defines a bounded PCU kernel.
///
/// Bare `#[pcu]` marks a pure, expression-bodied f32 or f64 helper. The source function remains
/// callable on the CPU, while its hidden companion lowers the same expression into kernel IR.
/// Helpers may call nested or module-qualified helpers, including through import aliases.
/// Helper recursion and total IR size are bounded and report `PcuError` during IR construction.
/// An explicit `Result<PcuTensor<T>, PcuExecutionError>` return selects the bounded owned-tensor
/// profile: homogeneous builtin operations, immutable graph temporaries and marked helper
/// composition. Scalars may be concrete or generic under `T: PcuScalar`; operation coverage is
/// checked by the backend. Borrowed source signatures accept RAM or resident storage.
/// A sole by-value resident input supports builtin composition with lexical moves and borrows;
/// references may precede its final bare use, but any use after that move is rejected. Consuming
/// marked helpers and owner/reference alias locals are not yet supported. Internal graph values
/// describe non-escaping SSA temporaries, rather than independent physical Rust owners.
///
/// `#[pcu(invocations = N)]` or `#[pcu(invocations: N)]` defines a kernel. It keeps
/// `<name>_bindings`, `<name>_ir`, and typed `<name>_prepare` / `<name>_prepare_device` APIs, and
/// generates a direct function with the source name and signature. The direct function uses the
/// facade's selected global backend and returns `Result<(), PcuExecutionError>`; it never selects
/// a CPU fallback. The prepare functions build IR and prepare the backend once, then return a
/// reusable `FnMut` that calls that prepared executable. Host and resident-device calls preserve
/// the source slice and [`PcuDeviceBuffer`](https://docs.rs/fusion-pcu/latest/fusion_pcu/struct.PcuDeviceBuffer.html)
/// argument types respectively.
///
/// Kernel bodies are limited to the documented bounded indexed or canonical grid-stride maps and
/// supported scalar profiles. Helper bodies support homogeneous f32 or f64 arithmetic expressions.
/// `flag(strict)` requests checked internal compound operations; `flag(non_strict)` explicitly
/// selects boundary handling. Unannotated owned helpers inherit the caller's active mode, with
/// the global numerical mode providing the outer default. Scalar helper and invocation operators
/// are already checked in either mode. Raw `_ir` and `_prepare` builders use their documented
/// fixed defaults and do not consult the process policy.
/// Invocation kernels may select one underflow policy with `flag(ieee_underflow)`,
/// `flag(allow_gradual_underflow)`, or `flag(reject_subnormal_result)`. The selected policy
/// applies to every checked floating operation in that compiled kernel, including nested scalar
/// helper expansions. Concrete F32/F64 invocation kernels may independently select
/// `flag(clamp_range)` to report and recover from finite range faults; it can be combined with an
/// underflow flag. Scalar helper declarations and owned tensor compositions do not accept this
/// invocation-only range flag.
#[proc_macro_attribute]
pub fn pcu(attr: TokenStream, item: TokenStream) -> TokenStream {
    if let Ok(helper_args) = syn::parse::<PcuScalarHelperArgs>(attr.clone()) {
        let function = parse_macro_input!(item as ItemFn);
        if owned::declares_owned_tensor_return(&function.sig.output) {
            if helper_args.clamp_range {
                return Error::new_spanned(
                    &function.sig,
                    "`flag(clamp_range)` is not supported on owned tensor composition helpers yet",
                )
                .into_compile_error()
                .into();
            }
            return match owned::expand_owned_return_with_policies(
                &function,
                &helper_args.crate_path,
                helper_args.underflow_flag,
                helper_args.numerical_mode,
            ) {
                Ok(tokens) => tokens.into(),
                Err(error) => error.into_compile_error().into(),
            };
        }
        if helper_args.underflow_flag.is_some() || helper_args.clamp_range {
            return Error::new_spanned(
                &function.sig,
                "float policy flags are supported only on owned tensor composition helpers; `clamp_range` is currently invocation-only",
            )
            .into_compile_error()
            .into();
        }
        return match expand_pcu_scalar_helper(function, &helper_args.crate_path) {
            Ok(tokens) => tokens.into(),
            Err(error) => error.into_compile_error().into(),
        };
    }
    let args = parse_macro_input!(attr as PcuDispatchArgs);
    let function = parse_macro_input!(item as ItemFn);
    match expand_pcu_direct(args, &function) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.into_compile_error().into(),
    }
}

fn expand_pcu_scalar_helper(
    mut function: ItemFn,
    crate_path: &Path,
) -> Result<TokenStream2, Error> {
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
    let helper = parse_pcu_helper(&function)?;
    let companion_crate_path = rebase_companion_path(crate_path.clone());
    let mut emitter = RuntimeExprEmitter::new(
        &[],
        &helper.ident,
        &companion_crate_path,
        false,
        helper.scalar,
        false,
    );
    emitter
        .values
        .extend(helper.parameters.iter().enumerate().map(|(index, ident)| {
            (
                ident.clone(),
                quote! { __pcu_arguments[#index] },
                helper.scalar,
            )
        }));
    let (result, result_kind) = emitter.emit_expr(&helper.body)?;
    if result_kind != helper.scalar {
        return Err(Error::new(
            helper.body.span(),
            "PCU helper result must match its f32 or f64 parameter profile",
        ));
    }
    let statements = emitter.statements;
    let helper_ident = &function.sig.ident;
    let companion_ident = helper_ident;
    let scalar_ty = helper.scalar.rust_type();
    let vis = &function.vis;
    let companion_cfg = function
        .attrs
        .iter()
        .flat_map(project_companion_cfg)
        .collect::<Vec<_>>();
    function
        .attrs
        .retain(|attribute| !attribute_ends_with(attribute, "pcu"));
    function.attrs.push(syn::parse_quote!(#[allow(dead_code)]));
    let argument_count = helper.parameters.len();
    Ok(quote! {
        #function

        #[doc(hidden)]
        #(#companion_cfg)*
        #vis mod #companion_ident {
            #[allow(unused_imports)]
            use super::*;

            pub fn __pcu_lower<'a, const __PCU_MAX_OPS: usize>(
                __pcu_context: &mut #companion_crate_path::PcuScalarLowering<'a, __PCU_MAX_OPS>,
                __pcu_arguments: [#companion_crate_path::PcuScalarValue<#scalar_ty>; #argument_count],
            ) -> ::core::result::Result<#companion_crate_path::PcuScalarValue<#scalar_ty>, #companion_crate_path::PcuError> {
                __pcu_context.enter_helper()?;
                let __pcu_result = (|| -> ::core::result::Result<
                    #companion_crate_path::PcuScalarValue<#scalar_ty>,
                    #companion_crate_path::PcuError,
                > {
                    #(#statements)*
                    ::core::result::Result::Ok(#result)
                })();
                __pcu_context.leave_helper();
                __pcu_result
            }
        }
    })
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
    let function = normalize_function_where(function)?;
    expand_pcu_dispatch_inner(args, &function, false)
}

fn expand_pcu_direct(args: PcuDispatchArgs, function: &ItemFn) -> Result<TokenStream2, Error> {
    let function = normalize_function_where(function)?;
    expand_pcu_dispatch_inner(args, &function, true)
}

fn normalize_function_where(function: &ItemFn) -> Result<ItemFn, Error> {
    let mut normalized = function.clone();
    let Some(where_clause) = normalized.sig.generics.where_clause.take() else {
        return Ok(normalized);
    };
    if where_clause.predicates.len() != 1 {
        return Err(Error::new_spanned(
            where_clause,
            "PCU generic kernels support one sealed scalar bound in the where clause",
        ));
    }
    let predicate = where_clause
        .predicates
        .first()
        .expect("one where predicate was checked");
    let syn::WherePredicate::Type(predicate) = predicate else {
        return Err(Error::new_spanned(
            predicate,
            "PCU generic kernels support only a sealed scalar type bound",
        ));
    };
    let Type::Path(bounded_type) = &predicate.bounded_ty else {
        return Err(Error::new_spanned(
            &predicate.bounded_ty,
            "PCU generic kernels support only a sealed scalar type bound",
        ));
    };
    if bounded_type.qself.is_some() || bounded_type.path.segments.len() != 1 {
        return Err(Error::new_spanned(
            &predicate.bounded_ty,
            "PCU generic kernels support only a sealed scalar type bound",
        ));
    }
    let Some(GenericParam::Type(parameter)) = normalized
        .sig
        .generics
        .params
        .iter_mut()
        .find(|parameter| matches!(parameter, GenericParam::Type(parameter) if parameter.ident == bounded_type.path.segments[0].ident))
    else {
        return Err(Error::new_spanned(
            &predicate.bounded_ty,
            "where-clause scalar bound must name the kernel's type parameter",
        ));
    };
    if !parameter.bounds.is_empty() {
        return Err(Error::new_spanned(
            &parameter.bounds,
            "PCU generic kernels do not combine inline and where-clause bounds",
        ));
    }
    if predicate.bounds.len() != 1
        || !predicate.bounds.iter().any(|bound| match bound {
            syn::TypeParamBound::Trait(bound) => {
                bound.path.segments.last().is_some_and(|segment| {
                    segment.ident == "PcuScalar" || segment.ident == "PcuWrappingInteger"
                })
            }
            _ => false,
        })
    {
        return Err(Error::new_spanned(
            predicate,
            "where-clause bounds must be exactly `T: PcuScalar` or `T: PcuWrappingInteger`",
        ));
    }
    parameter.bounds.clone_from(&predicate.bounds);
    Ok(normalized)
}

fn unique_builder_lifetime(function: &ItemFn) -> syn::Lifetime {
    let mut name = String::from("__pcu_builder");
    while function.sig.generics.params.iter().any(|parameter| {
        matches!(parameter, GenericParam::Lifetime(parameter) if parameter.lifetime.ident == name)
    }) {
        name.insert(0, '_');
    }
    syn::Lifetime::new(&format!("'{name}"), function.sig.ident.span())
}

// Keep the shared lowering path together: generic identities and concrete kernels must pass the
// same structural/body validation before they diverge into their respective typed builders.
#[allow(clippy::too_many_lines)]
fn expand_pcu_dispatch_inner(
    args: PcuDispatchArgs,
    function: &ItemFn,
    direct_entry: bool,
) -> Result<TokenStream2, Error> {
    let vis = function.vis.clone();
    let function_ident = function.sig.ident.clone();
    let builder_ident = if direct_entry {
        format_ident!("{}_ir", function_ident)
    } else {
        function_ident.clone()
    };
    let policy_builder_ident = format_ident!("__{}_with_float_underflow_policy", builder_ident);
    let bindings_ident = format_ident!("{}_bindings", function_ident);
    let crate_path = args.crate_path;
    let underflow_flag = args.underflow_flag;
    // Invocation operators already enforce the strict scalar contract.
    let _numerical_mode = args.numerical_mode;
    let clamp_range = args.clamp_range;
    let const_generics = validate_const_generics(function)?;
    let generic_scalar = generic_scalar_type(function)?;
    let invocation_expr = lower_invocation_expr(&args.invocations, &const_generics)?;
    let builder_lifetime = unique_builder_lifetime(function);
    let generated_generics = if function.sig.generics.params.is_empty() {
        quote! { <#builder_lifetime> }
    } else {
        let params = &function.sig.generics.params;
        quote! { <#builder_lifetime, #params> }
    };
    let binding_specs = parse_bindings(
        &function.sig.inputs,
        generic_scalar.as_ref(),
        &const_generics,
    )?;
    validate_binding_shapes(&binding_specs)?;
    if clamp_range
        && !binding_specs
            .iter()
            .all(|binding| matches!(binding.scalar, ScalarKind::F32 | ScalarKind::F64))
    {
        return Err(Error::new_spanned(
            function,
            "`flag(clamp_range)` is supported only by concrete f32/f64 invocation kernels; integer and generic profiles do not support range recovery",
        ));
    }
    let mut runtime_helper_body = None;
    let (loop_extent, data_ops) = if let Some((extent, data_ops)) =
        checked_div_rem::lower(function, &binding_specs, &crate_path)?
    {
        (extent, data_ops)
    } else {
        let body = validate_body(function)?;
        let (invocation, assignment, extent, matrix_locals) = match &body {
            ValidatedBody::Indexed {
                invocation,
                assignment,
                matrix_locals,
            } => (invocation, *assignment, None, matrix_locals.as_ref()),
            ValidatedBody::GridStride {
                invocation,
                assignment,
                extent,
                matrix_locals,
            } => (
                invocation,
                *assignment,
                Some(*extent),
                matrix_locals.as_ref(),
            ),
        };
        if let Some(locals) = matrix_locals {
            validate_matrix_locals(locals, &binding_specs, &const_generics)?;
        }
        if let Some(extent) = extent
            && binding_specs.iter().any(|binding| binding.matrix.is_some())
        {
            validate_matrix_grid_extent(extent, &binding_specs)?;
        }
        let normalized_assignment = if let Some(locals) = matrix_locals {
            normalize_matrix_local_indices(assignment, invocation, locals, &binding_specs)?
        } else {
            assignment.clone()
        };
        let assignment = &normalized_assignment;
        let output_binding = validate_assignment_target(assignment, &binding_specs, invocation)?;
        if let Some(scalar_type) = &generic_scalar {
            if generic_scalar_wrapping(function) {
                validate_generic_wrapping_map(assignment, &binding_specs, invocation, scalar_type)?;
                let mut emitter = ExprEmitter::new(
                    &binding_specs,
                    invocation,
                    &crate_path,
                    extent.is_some(),
                    output_binding.scalar,
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
            if expr_contains_call(&assignment.right)
                || matches!(output_binding.scalar, ScalarKind::F32 | ScalarKind::F64)
            {
                if !matches!(output_binding.scalar, ScalarKind::F32 | ScalarKind::F64) {
                    return Err(Error::new(
                        assignment.right.span(),
                        "PCU scalar helper calls require f32 or f64 output bindings",
                    ));
                }
                let mut emitter = RuntimeExprEmitter::new(
                    &binding_specs,
                    invocation,
                    &crate_path,
                    extent.is_some(),
                    output_binding.scalar,
                    true,
                );
                let (result_value, result_type) = emitter.emit_expr(&assignment.right)?;
                if result_type != output_binding.scalar {
                    return Err(Error::new(
                        assignment.right.span(),
                        "PCU store value type must match the output binding element type",
                    ));
                }
                let output_slot = output_binding.binding;
                let pcu = &crate_path;
                let statements = emitter.statements;
                let store_index = if extent.is_some() {
                    quote! { GridStrideId }
                } else {
                    quote! { InvocationId }
                };
                runtime_helper_body = Some(quote! {
                    let mut __pcu_context = #pcu::PcuScalarLowering::with_float_underflow_policy(builder, 1, __pcu_float_underflow_policy)
                        .with_range_policy(__pcu_float_range_policy);
                    #(#statements)*
                    __pcu_context.store_value(
                        #pcu::PcuBindingRef::new(0, #output_slot),
                        #pcu::PcuDispatchIndex::#store_index,
                        #result_value,
                    )?;
                    let builder = __pcu_context.finish()?;
                });
                (extent, Vec::new())
            } else {
                let mut emitter = ExprEmitter::new(
                    &binding_specs,
                    invocation,
                    &crate_path,
                    extent.is_some(),
                    output_binding.scalar,
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
        }
    };
    let pcu = &crate_path;

    let runtime_grid_stride = runtime_helper_body.is_some() && loop_extent.is_some();
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
    } else if runtime_helper_body.is_some() {
        let runtime_body = runtime_helper_body
            .take()
            .expect("runtime helper body was checked");
        let operations = quote! { #runtime_body };
        let capacity = 256_usize;
        (operations, capacity, quote! {})
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
    let normal_builder = if runtime_grid_stride {
        let extent = lower_invocation_expr(
            loop_extent.expect("runtime grid-stride lowering has an extent"),
            &const_generics,
        )?;
        quote! {
            let builder = #pcu::model::PcuDispatchKernelBuilder::<#op_count>::new(
                #kernel_id,
                "main",
                [invocations, 1, 1],
            )
            .with_bindings(bindings);
            #operations
            #pcu::model::PcuGridStrideKernelBuilder::new(
                builder,
                const {
                    let extent: usize = #extent;
                    assert!(extent != 0, "PCU grid-stride extent must be nonzero");
                    assert!(extent <= u32::MAX as usize, "PCU grid-stride extent exceeds u32");
                    extent as u32
                },
            )
        }
    } else {
        quote! {
            let builder = #pcu::model::PcuDispatchKernelBuilder::<#op_count>::new(
                #kernel_id,
                "main",
                [invocations, 1, 1],
            )
            .with_bindings(bindings);
            #operations
            builder.with_control_op(#pcu::PcuDispatchControlOp::Return)
        }
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
    let builder_type = if runtime_grid_stride {
        quote! { #pcu::model::PcuGridStrideKernelBuilder<#builder_lifetime, #op_count> }
    } else {
        quote! { #pcu::model::PcuDispatchKernelBuilder<#builder_lifetime, #op_count> }
    };
    let builder_result = if generic_identity {
        quote! { #pcu::model::PcuScalarIdentityBuildError }
    } else {
        quote! { #pcu::PcuError }
    };
    let matrix_count_check = if let Some(matrix) = binding_specs
        .iter()
        .find_map(|binding| binding.matrix.as_ref())
    {
        let rows = lower_invocation_expr(&matrix.rows, &const_generics)?;
        let columns = lower_invocation_expr(&matrix.columns, &const_generics)?;
        if let Some(extent) = loop_extent {
            let extent = lower_invocation_expr(extent, &const_generics)?;
            quote! {
                let __pcu_matrix_rows: usize = #rows;
                let __pcu_matrix_columns: usize = #columns;
                assert!(__pcu_matrix_rows != 0, "PCU matrix row count must be nonzero");
                assert!(__pcu_matrix_columns != 0, "PCU matrix column count must be nonzero");
                let __pcu_matrix_count = __pcu_matrix_rows
                    .checked_mul(__pcu_matrix_columns)
                    .expect("PCU matrix element count overflow");
                let __pcu_grid_extent: usize = #extent;
                assert!(__pcu_grid_extent == __pcu_matrix_count, "PCU grid-stride extent must equal matrix rows times columns");
            }
        } else {
            quote! {
                let __pcu_matrix_rows: usize = #rows;
                let __pcu_matrix_columns: usize = #columns;
                assert!(__pcu_matrix_rows != 0, "PCU matrix row count must be nonzero");
                assert!(__pcu_matrix_columns != 0, "PCU matrix column count must be nonzero");
                let __pcu_matrix_count = __pcu_matrix_rows
                    .checked_mul(__pcu_matrix_columns)
                    .expect("PCU matrix element count overflow");
                assert!(count == __pcu_matrix_count, "PCU invocation count must equal matrix rows times columns");
            }
        }
    } else {
        quote! {}
    };
    let generic_arguments = function
        .sig
        .generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(param) => Some(param.ident.to_token_stream()),
            GenericParam::Const(param) => Some(param.ident.to_token_stream()),
            GenericParam::Lifetime(_) => None,
        })
        .collect::<Vec<_>>();
    let supports_float_range = generic_scalar.is_none()
        && !binding_specs.is_empty()
        && binding_specs
            .iter()
            .all(|binding| matches!(binding.scalar, ScalarKind::F32 | ScalarKind::F64));
    let bindings_call = generic_scalar.map_or_else(
        || quote! { #bindings_ident() },
        |scalar_ident| quote! { #bindings_ident::<#scalar_ident>() },
    );
    let builder_call = if generic_arguments.is_empty() {
        quote! { #builder_ident(&bindings) }
    } else {
        quote! { #builder_ident::<#(#generic_arguments),*>(&bindings) }
    };
    let prepared_arguments = binding_specs
        .iter()
        .zip(&function.sig.inputs)
        .map(|(binding, input)| {
            let FnArg::Typed(input) = input else {
                unreachable!("receivers are rejected by parse_bindings")
            };
            let scalar = if binding.scalar == ScalarKind::Generic {
                let generic = binding
                    .generic_scalar
                    .as_ref()
                    .expect("generic binding scalar");
                quote! { #generic }
            } else {
                let scalar = binding.scalar.rust_type();
                quote! { #scalar }
            };
            prepared::Argument {
                ident: binding.ident.clone(),
                binding: binding.binding,
                read_write: binding.access == BindingAccess::ReadWrite,
                scalar_reference: binding.scalar_reference,
                flatten_matrix: binding.matrix.is_some(),
                scalar,
                ty: input.ty.as_ref().clone(),
            }
        })
        .collect::<Vec<_>>();
    let explicit_policy = underflow_flag.map(|flag| match flag {
        PcuOwnedFlag::IeeeUnderflow => quote! { #pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding },
        PcuOwnedFlag::AllowGradualUnderflow => {
            quote! { #pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow }
        }
        PcuOwnedFlag::RejectSubnormalResult => {
            quote! { #pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult }
        }
    });
    let explicit_range_policy = clamp_range.then(|| quote! { #pcu::PcuRangePolicy::Clamp });
    let public_policy = explicit_policy
        .clone()
        .unwrap_or_else(|| quote! { #pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding });
    let public_range_policy = explicit_range_policy
        .clone()
        .unwrap_or_else(|| quote! { #pcu::PcuRangePolicy::Reject });
    let policy_builder_direct_call = if generic_arguments.is_empty() {
        quote! { #policy_builder_ident(bindings, #public_policy, #public_range_policy) }
    } else {
        quote! { #policy_builder_ident::<#(#generic_arguments),*>(bindings, #public_policy, #public_range_policy) }
    };
    let prepared_input = prepared::Input {
        pcu,
        function,
        visibility: &vis,
        function_ident: &function_ident,
        arguments: &prepared_arguments,
        generic_arguments: &generic_arguments,
        prepare_error: &builder_result,
        bindings_call,
        builder_call,
        policy_builder_ident: policy_builder_ident.clone(),
        explicit_policy,
        explicit_range_policy,
        supports_float_range,
    };
    let direct = direct_entry.then(|| hosted::generate(&prepared_input));
    let generated = prepared::generate(prepared_input);
    Ok(quote! {
        #wrapping_body_item
        #vis const fn #bindings_ident #binding_generic() -> [#pcu::PcuBinding<#binding_lifetime>; #binding_count] {
            [#(#binding_items),*]
        }

        #vis fn #builder_ident #generated_generics(
            bindings: &#builder_lifetime [#pcu::PcuBinding<#builder_lifetime>],
        ) -> ::core::result::Result<#builder_type, #builder_result> {
            #policy_builder_direct_call
        }

        #[doc(hidden)]
        #vis fn #policy_builder_ident #generated_generics(
            bindings: &#builder_lifetime [#pcu::PcuBinding<#builder_lifetime>],
            __pcu_float_underflow_policy: #pcu::PcuFloatUnderflowPolicy,
            __pcu_float_range_policy: #pcu::PcuRangePolicy,
        ) -> ::core::result::Result<#builder_type, #builder_result> {
            let invocations: u32 = const {
                let count: usize = #invocation_expr;
                #matrix_count_check
                assert!(count != 0, "PCU invocation count must be nonzero");
                assert!(count <= u32::MAX as usize, "PCU invocation count exceeds u32");
                count as u32
            };
            #builder_body
        }

        #generated
        #direct
    })
}

fn expr_contains_call(expression: &Expr) -> bool {
    match expression {
        Expr::Call(_) => true,
        Expr::Binary(binary) => {
            expr_contains_call(&binary.left) || expr_contains_call(&binary.right)
        }
        Expr::Paren(paren) => expr_contains_call(&paren.expr),
        _ => false,
    }
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

fn attribute_ends_with(attribute: &syn::Attribute, name: &str) -> bool {
    attribute
        .path()
        .segments
        .last()
        .is_some_and(|segment| segment.ident == name)
}

fn project_companion_cfg(attribute: &syn::Attribute) -> Vec<syn::Attribute> {
    if attribute.path().is_ident("cfg") {
        return vec![attribute.clone()];
    }
    if !attribute.path().is_ident("cfg_attr") {
        return Vec::new();
    }
    let syn::Meta::List(list) = &attribute.meta else {
        return Vec::new();
    };
    let parser = syn::punctuated::Punctuated::<syn::Meta, Token![,]>::parse_terminated;
    let Ok(arguments) = parser.parse2(list.tokens.clone()) else {
        return Vec::new();
    };
    if arguments.is_empty() {
        return Vec::new();
    }
    let condition = arguments.first().expect("nonempty cfg_attr").clone();
    let nested = arguments
        .iter()
        .skip(1)
        .flat_map(|meta| match meta {
            syn::Meta::List(nested_list) if nested_list.path.is_ident("cfg_attr") => {
                let nested_attribute: syn::Attribute = syn::parse_quote!(#[cfg_attr #nested_list]);
                project_companion_cfg(&nested_attribute)
            }
            syn::Meta::List(_) if meta.path().is_ident("cfg") => {
                vec![syn::parse_quote!(#[#meta])]
            }
            _ => Vec::new(),
        })
        .collect::<Vec<_>>();
    if nested.is_empty() {
        return Vec::new();
    }
    let nested_meta = nested
        .iter()
        .map(|attribute| &attribute.meta)
        .collect::<Vec<_>>();
    vec![syn::parse_quote!(#[cfg_attr(#condition, #(#nested_meta),*)])]
}

fn parse_pcu_helper(function: &ItemFn) -> Result<PcuHelper, Error> {
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
    let Some(scalar) = scalar_kind(return_type) else {
        return Err(Error::new_spanned(
            return_type,
            "this PCU helper profile supports only `f32` or `f64` scalar types",
        ));
    };
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
        if scalar_kind(&argument.ty) != Some(scalar) {
            return Err(Error::new_spanned(
                &argument.ty,
                "PCU helper parameters must match the declared f32 or f64 result type",
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
        scalar,
        body: body.clone(),
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
            "generic `T: PcuScalar` kernels currently support only an indexed identity copy",
        ));
    };
    let source_matrix = shape::matrix_base(source_index).is_some();
    let source_base = shape::matrix_base(source_index).unwrap_or(&source_index.expr);
    let Some(source) = expr_ident(source_base) else {
        return Err(Error::new(
            source_base.span(),
            "identity source must be a binding",
        ));
    };
    let Expr::Index(target_index) = assignment.left.as_ref() else {
        unreachable!("validated generic output binding is indexed")
    };
    let target_matrix = shape::matrix_base(target_index).is_some();
    let target_base = shape::matrix_base(target_index).unwrap_or(&target_index.expr);
    let Some(target) = expr_ident(target_base) else {
        unreachable!("validated generic output binding is named")
    };
    let source_binding = bindings
        .iter()
        .find(|binding| binding.ident == *source)
        .ok_or_else(|| Error::new(source.span(), "identity source must be a binding"))?;
    let target_binding = bindings
        .iter()
        .find(|binding| binding.ident == *target)
        .ok_or_else(|| Error::new(target.span(), "identity target must be a binding"))?;
    if source_matrix != target_matrix || source_binding.matrix.is_some() != source_matrix {
        return Err(Error::new(
            assignment.span(),
            "generic identity copy must use matching flat or rank-two bindings",
        ));
    }
    if let Some(dimensions) = &source_binding.matrix {
        if !shape::is_canonical_matrix_index(source_index, invocation, &dimensions.columns) {
            return Err(Error::new(
                source_index.span(),
                "generic rank-two identity source must use canonical row and column indexing",
            ));
        }
    } else {
        validate_invocation_index(&source_index.index, invocation)?;
    }
    if let Some(dimensions) = &target_binding.matrix {
        if !shape::is_canonical_matrix_index(target_index, invocation, &dimensions.columns) {
            return Err(Error::new(
                target_index.span(),
                "generic rank-two identity target must use canonical row and column indexing",
            ));
        }
    } else {
        validate_invocation_index(&target_index.index, invocation)?;
    }
    if source == target
        || source_binding.access != BindingAccess::ReadOnly
        || target_binding.access != BindingAccess::ReadWrite
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
    const_generics: &[Ident],
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
        let (access, scalar, binding_generic, scalar_reference, matrix) =
            parse_binding_type(&input.ty, generic_scalar, const_generics)?;
        let binding = u32::try_from(bindings.len())
            .map_err(|_| Error::new(input.span(), "too many PCU bindings for this macro"))?;
        bindings.push(BindingSpec {
            ident: pat.ident.clone(),
            access,
            binding,
            scalar,
            generic_scalar: binding_generic,
            scalar_reference,
            matrix,
        });
    }
    Ok(bindings)
}

fn validate_binding_shapes(bindings: &[BindingSpec]) -> Result<(), Error> {
    let mut matrix_bindings = bindings
        .iter()
        .filter_map(|binding| binding.matrix.as_ref().map(|matrix| (binding, matrix)));
    let Some((reference_binding, reference)) = matrix_bindings.next() else {
        return Ok(());
    };
    let generic_scalar = reference_binding.generic_scalar.as_ref();
    if generic_scalar.is_none()
        && !matches!(reference_binding.scalar, ScalarKind::F32 | ScalarKind::F64)
    {
        return Err(Error::new(
            reference_binding.ident.span(),
            "rank-two PCU maps currently support concrete f32/f64 or one generic `T: PcuScalar` element type",
        ));
    }
    for binding in bindings {
        let same_scalar = generic_scalar.map_or_else(
            || binding.scalar == reference_binding.scalar && binding.generic_scalar.is_none(),
            |generic| {
                binding.scalar == ScalarKind::Generic
                    && binding.generic_scalar.as_ref() == Some(generic)
            },
        );
        if !same_scalar {
            return Err(Error::new(
                binding.ident.span(),
                "rank-two PCU maps require every binding to use the same scalar type",
            ));
        }
        if binding.matrix.is_none() && (generic_scalar.is_some() || !binding.scalar_reference) {
            return Err(Error::new(
                binding.ident.span(),
                "rank-two PCU maps do not mix flat vectors or scalar references with matrices",
            ));
        }
    }
    for (binding, matrix) in matrix_bindings {
        if !shape::same_dimension(&reference.rows, &matrix.rows)
            || !shape::same_dimension(&reference.columns, &matrix.columns)
        {
            return Err(Error::new(
                binding.ident.span(),
                "all PCU rank-two matrix bindings must have identical row and column extents",
            ));
        }
    }
    Ok(())
}

type ParsedBindingType = (
    BindingAccess,
    ScalarKind,
    Option<Ident>,
    bool,
    Option<shape::FixedMatrixShape>,
);

fn parse_binding_type(
    ty: &Type,
    generic_scalar: Option<&Ident>,
    const_generics: &[Ident],
) -> Result<ParsedBindingType, Error> {
    let Type::Reference(reference) = ty else {
        return Err(Error::new(
            ty.span(),
            "PCU binding types must be scalar references to f32 or f64, or slices/fixed arrays of supported PCU scalars",
        ));
    };
    let (element_type, scalar_reference, matrix) = match reference.elem.as_ref() {
        Type::Slice(slice) => (slice.elem.as_ref(), false, None),
        Type::Array(array) => {
            if let Some((element, matrix)) = shape::nested_array_type(&reference.elem) {
                (element, false, Some(matrix))
            } else {
                (array.elem.as_ref(), false, None)
            }
        }
        Type::Path(_) if reference.mutability.is_none() => (reference.elem.as_ref(), true, None),
        _ => {
            return Err(Error::new(
                reference.elem.span(),
                "PCU dispatch resources must be scalar slices or fixed arrays",
            ));
        }
    };
    if let Some(matrix) = &matrix {
        lower_invocation_expr(&matrix.rows, const_generics)?;
        lower_invocation_expr(&matrix.columns, const_generics)?;
    }
    let element_type = transparent_type(element_type);
    let Type::Path(element) = element_type else {
        return Err(Error::new(
            element_type.span(),
            "this PCU dispatch macro currently supports only f32, f64, u8, u16, u32, u64, i8, i16, i32, and i64 elements",
        ));
    };
    if let Some(generic) = generic_scalar
        && element.path.is_ident(generic)
    {
        if scalar_reference {
            return Err(Error::new(
                element.span(),
                "generic scalar references are unsupported; use a concrete f32 or f64 reference",
            ));
        }
        return Ok((
            if reference.mutability.is_some() {
                BindingAccess::ReadWrite
            } else {
                BindingAccess::ReadOnly
            },
            ScalarKind::Generic,
            Some(generic.clone()),
            false,
            matrix,
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
    if scalar_reference && !matches!(scalar, ScalarKind::F32 | ScalarKind::F64) {
        return Err(Error::new(
            element.span(),
            "read-only scalar parameters currently support only f32 and f64",
        ));
    }
    Ok((
        if reference.mutability.is_some() {
            BindingAccess::ReadWrite
        } else {
            BindingAccess::ReadOnly
        },
        scalar,
        None,
        scalar_reference,
        matrix,
    ))
}

enum ValidatedBody<'a> {
    Indexed {
        invocation: Ident,
        assignment: &'a ExprAssign,
        matrix_locals: Option<MatrixLocals>,
    },
    GridStride {
        invocation: Ident,
        assignment: &'a ExprAssign,
        extent: &'a Expr,
        matrix_locals: Option<MatrixLocals>,
    },
}

fn validate_body(function: &ItemFn) -> Result<ValidatedBody<'_>, Error> {
    let statements = &function.block.stmts;
    if statements.len() == 3
        && matches!(statements.first(), Some(Stmt::Local(local)) if matches!(local.pat, Pat::Ident(ref pat) if pat.mutability.is_some()))
    {
        return validate_grid_stride_body(statements);
    }
    if statements.len() != 2 && statements.len() != 4 {
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

    let (assignment_index, matrix_locals) = if statements.len() == 4 {
        let locals = parse_matrix_locals(&statements[1], &statements[2], &pat.ident, None)?;
        (3, Some(locals))
    } else {
        (1, None)
    };
    let Stmt::Expr(Expr::Assign(assignment), Some(_)) = &statements[assignment_index] else {
        return Err(Error::new(
            diagnostic_statement_span(&statements[assignment_index]),
            "second PCU dispatch statement must be `output[invocation] = <expr>;`",
        ));
    };
    Ok(ValidatedBody::Indexed {
        invocation: pat.ident.clone(),
        assignment,
        matrix_locals,
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
    let (assignment, increment, matrix_locals) = match loop_expr.body.stmts.as_slice() {
        [
            Stmt::Expr(Expr::Assign(assignment), Some(_)),
            Stmt::Expr(Expr::Binary(increment), Some(_)),
        ] => (assignment, increment, None),
        [
            row,
            column,
            Stmt::Expr(Expr::Assign(assignment), Some(_)),
            Stmt::Expr(Expr::Binary(increment), Some(_)),
        ] => {
            let locals =
                parse_matrix_locals(row, column, &id_pat.ident, Some(stride_pat.ident.clone()))
                    .map_err(|_| unsupported(row.span()))?;
            (assignment, increment, Some(locals))
        }
        _ => {
            let span = loop_expr
                .body
                .stmts
                .first()
                .map_or_else(|| loop_expr.while_token.span(), grid_stride_statement_span);
            return Err(unsupported(span));
        }
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
        matrix_locals,
    })
}

fn parse_matrix_locals(
    row_statement: &Stmt,
    column_statement: &Stmt,
    invocation: &Ident,
    stride: Option<Ident>,
) -> Result<MatrixLocals, Error> {
    let (row_name, row_expression) = matrix_local_declaration(row_statement)?;
    let (column_name, column_expression) = matrix_local_declaration(column_statement)?;
    let Some(columns) = shape::matrix_local_extent(row_expression, column_expression, invocation)
    else {
        return Err(Error::new(
            row_expression.span(),
            "matrix locals must be `let row = id / C; let column = id % C` using one extent",
        ));
    };
    if *row_name == *column_name {
        return Err(Error::new(
            row_statement.span(),
            "row and column coordinates must use distinct local names",
        ));
    }
    Ok(MatrixLocals {
        row: row_name.clone(),
        column: column_name.clone(),
        columns,
        invocation: invocation.clone(),
        stride,
    })
}

fn matrix_local_declaration(statement: &Stmt) -> Result<(&Ident, &Expr), Error> {
    let Stmt::Local(local) = statement else {
        return Err(Error::new(
            statement.span(),
            "matrix coordinates must be immutable local bindings",
        ));
    };
    let Pat::Ident(pattern) = &local.pat else {
        return Err(Error::new(
            local.pat.span(),
            "matrix coordinates must use plain local names",
        ));
    };
    if !local.attrs.is_empty()
        || pattern.mutability.is_some()
        || pattern.by_ref.is_some()
        || pattern.subpat.is_some()
    {
        return Err(Error::new(
            pattern.span(),
            "matrix coordinates must be immutable plain local bindings",
        ));
    }
    let Some(initializer) = &local.init else {
        return Err(Error::new(
            pattern.span(),
            "matrix coordinates must be initialized from the invocation id",
        ));
    };
    if initializer.diverge.is_some() {
        return Err(Error::new(
            initializer.expr.span(),
            "matrix coordinate initializer cannot diverge",
        ));
    }
    Ok((&pattern.ident, &initializer.expr))
}

fn validate_matrix_locals(
    locals: &MatrixLocals,
    bindings: &[BindingSpec],
    const_generics: &[Ident],
) -> Result<(), Error> {
    let Some(matrix) = bindings.iter().find_map(|binding| binding.matrix.as_ref()) else {
        return Err(Error::new(
            locals.row.span(),
            "row and column locals are supported only with fixed rank-two matrix bindings",
        ));
    };
    if !shape::same_dimension(&locals.columns, &matrix.columns) {
        return Err(Error::new(
            locals.columns.span(),
            "matrix coordinate divisor must equal the declared column extent",
        ));
    }
    let reserved = bindings
        .iter()
        .map(|binding| binding.ident.clone())
        .chain(core::iter::once(locals.invocation.clone()))
        .chain(locals.stride.iter().cloned())
        .chain(const_generics.iter().cloned());
    if let Some(shadowed) = [locals.row.clone(), locals.column.clone()]
        .into_iter()
        .find(|coordinate| reserved.clone().any(|name| name == *coordinate))
    {
        return Err(Error::new(
            shadowed.span(),
            "matrix coordinate locals cannot shadow a binding, invocation, stride, or const parameter",
        ));
    }
    Ok(())
}

fn validate_matrix_grid_extent(extent: &Expr, bindings: &[BindingSpec]) -> Result<(), Error> {
    let Some(matrix) = bindings.iter().find_map(|binding| binding.matrix.as_ref()) else {
        return Err(Error::new(
            extent.span(),
            "rank-two grid-stride extent requires fixed rank-two matrix bindings",
        ));
    };
    if !shape::is_matrix_element_count(extent, &matrix.rows, &matrix.columns) {
        return Err(Error::new(
            extent.span(),
            "rank-two grid-stride loop extent must equal the checked matrix row count times column count",
        ));
    }
    Ok(())
}

struct MatrixIndexNormalizer<'a> {
    invocation: &'a Ident,
    locals: &'a MatrixLocals,
    bindings: &'a [BindingSpec],
    error: Option<Error>,
}

impl VisitMut for MatrixIndexNormalizer<'_> {
    fn visit_expr_index_mut(&mut self, index: &mut ExprIndex) {
        let Some(base) = shape::matrix_base(index).and_then(expr_ident) else {
            visit_mut::visit_expr_index_mut(self, index);
            return;
        };
        let Some(binding) = self
            .bindings
            .iter()
            .find(|binding| binding.ident == *base && binding.matrix.is_some())
        else {
            visit_mut::visit_expr_index_mut(self, index);
            return;
        };
        let columns = &binding.matrix.as_ref().expect("matrix checked").columns;
        let row_column = shape::is_matrix_local_index(index, &self.locals.row, &self.locals.column);
        if row_column {
            let invocation = self.invocation;
            let Expr::Index(inner) = index.expr.as_mut() else {
                unreachable!("matrix base check guarantees nested index");
            };
            let row_index: Expr = syn::parse_quote!(#invocation / #columns);
            let column_index: Expr = syn::parse_quote!(#invocation % #columns);
            *inner.index = row_index;
            *index.index = column_index;
            visit_mut::visit_expr_index_mut(self, index);
            return;
        }
        if !shape::is_canonical_matrix_index(index, self.invocation, columns) {
            self.error = Some(Error::new(
                index.span(),
                "rank-two binding access must use canonical `matrix[row][column]` locals or `matrix[id / C][id % C]` indexing",
            ));
        }
        visit_mut::visit_expr_index_mut(self, index);
    }
}

fn normalize_matrix_local_indices(
    assignment: &ExprAssign,
    invocation: &Ident,
    locals: &MatrixLocals,
    bindings: &[BindingSpec],
) -> Result<ExprAssign, Error> {
    let mut assignment = assignment.clone();
    let mut normalizer = MatrixIndexNormalizer {
        invocation,
        locals,
        bindings,
        error: None,
    };
    normalizer.visit_expr_assign_mut(&mut assignment);
    if let Some(error) = normalizer.error {
        return Err(error);
    }
    Ok(assignment)
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
            "PCU dispatch assignment target must be `output[invocation]` or canonical `output[invocation / C][invocation % C]`",
        ));
    };
    let output_ident = if let Expr::Index(inner) = expr.as_ref() {
        let Some(output_ident) = expr_ident(&inner.expr) else {
            return Err(Error::new(
                inner.expr.span(),
                "PCU matrix assignment target must be a named fixed rank-two binding",
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
        let Some(dimensions) = binding.matrix.as_ref() else {
            return Err(Error::new(
                inner.span(),
                "nested PCU assignment is supported only for fixed rank-two array bindings",
            ));
        };
        if !shape::is_canonical_matrix_components(
            &inner.index,
            index,
            invocation_ident,
            &dimensions.columns,
        ) {
            return Err(Error::new(
                assignment.left.span(),
                "rank-two PCU assignment target must use `output[invocation / C][invocation % C]` with the declared column extent",
            ));
        }
        output_ident.clone()
    } else {
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
        if binding.matrix.is_some() {
            return Err(Error::new(
                expr.span(),
                "rank-two PCU assignment targets must use canonical row and column indexing",
            ));
        }
        output_ident.clone()
    };
    let Some(binding) = bindings
        .iter()
        .find(|binding| binding.ident == output_ident)
    else {
        return Err(Error::new(
            output_ident.span(),
            "unknown PCU output binding",
        ));
    };
    if binding.access != BindingAccess::ReadWrite {
        return Err(Error::new(
            output_ident.span(),
            "PCU assignment target must be mutable",
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

fn parse_float_bits(float: &LitFloat, expected: ScalarKind) -> Result<(ScalarKind, u64), Error> {
    let suffix = float.suffix();
    let scalar = match suffix {
        "" => match expected {
            ScalarKind::F32 | ScalarKind::F64 => expected,
            _ => ScalarKind::F32,
        },
        "f32" => ScalarKind::F32,
        "f64" => ScalarKind::F64,
        _ => {
            return Err(Error::new(
                float.span(),
                "PCU floating literals may use only the f32 or f64 suffix",
            ));
        }
    };
    if matches!(expected, ScalarKind::F32 | ScalarKind::F64) && scalar != expected {
        return Err(Error::new(
            float.span(),
            "PCU floating literal suffix does not match the scalar profile",
        ));
    }
    match scalar {
        ScalarKind::F32 => {
            let value = float.base10_parse::<f32>()?;
            if !value.is_finite() {
                return Err(Error::new(
                    float.span(),
                    "PCU floating literal is outside the finite f32 range",
                ));
            }
            Ok((scalar, u64::from(value.to_bits())))
        }
        ScalarKind::F64 => {
            let value = float.base10_parse::<f64>()?;
            if !value.is_finite() {
                return Err(Error::new(
                    float.span(),
                    "PCU floating literal is outside the finite f64 range",
                ));
            }
            Ok((scalar, value.to_bits()))
        }
        _ => unreachable!("the scalar profile was restricted to f32/f64"),
    }
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
    #[rustfmt::skip]
    use super::{
        PcuOwnedFlag,
        PcuScalarHelperArgs,
        PcuDispatchArgs,
        expand_pcu_dispatch,
        expand_pcu_scalar_helper,
    };
    use syn::ItemFn;

    #[test]
    fn numerical_flags_are_independent_and_unambiguous() {
        for (flag, strict) in [("strict", true), ("non_strict", false)] {
            let args = syn::parse_str::<PcuScalarHelperArgs>(&format!(
                "flag({flag}), flag(allow_gradual_underflow)"
            ))
            .unwrap();
            assert_eq!(args.numerical_mode, Some(strict));
            assert_eq!(
                args.underflow_flag,
                Some(PcuOwnedFlag::AllowGradualUnderflow)
            );
            let args = syn::parse_str::<PcuDispatchArgs>(&format!(
                "invocations = 4, flag({flag}), flag(clamp_range)"
            ))
            .unwrap();
            assert_eq!(args.numerical_mode, Some(strict));
            assert!(args.clamp_range);
        }
        for flags in [
            "flag(strict), flag(strict)",
            "flag(non_strict), flag(non_strict)",
            "flag(strict), flag(non_strict)",
            "flag(non_strict), flag(strict)",
        ] {
            assert!(syn::parse_str::<PcuScalarHelperArgs>(flags).is_err());
            assert!(syn::parse_str::<PcuDispatchArgs>(&format!("invocations=4, {flags}")).is_err());
        }
    }

    #[test]
    fn invocation_mode_flags_preserve_checked_scalar_lowering() {
        let source: ItemFn = syn::parse_quote! {
            fn kernel(input: &[f32], rhs: &[f32], output: &mut [f32]) {
                let invocation = context.global_invocation_id;
                output[invocation] = input[invocation] * rhs[invocation];
            }
        };
        let default = expand_pcu_dispatch(syn::parse_str("invocations=4").unwrap(), &source)
            .unwrap()
            .to_string();
        for flag in ["strict", "non_strict"] {
            let args =
                syn::parse_str::<PcuDispatchArgs>(&format!("invocations=4, flag({flag})")).unwrap();
            assert_eq!(
                expand_pcu_dispatch(args, &source).unwrap().to_string(),
                default
            );
        }
        assert!(default.contains("checked_binary_value"));
    }

    #[test]
    fn owned_numerical_flags_emit_scoped_overrides() {
        let source: ItemFn = syn::parse_quote! {
            fn multiply(lhs: &[[f32; 2]; 2], rhs: &[[f32; 2]; 2]) -> Result<PcuTensor<f32>, PcuExecutionError> {
                Ok(pcu::matmul(lhs, rhs)?)
            }
        };
        for (mode, token) in [
            (None, "None"),
            (Some(true), "Strict"),
            (Some(false), "Boundary"),
        ] {
            let generated = super::owned::expand_owned_return_with_policies(
                &source,
                &syn::parse_quote!(::fusion_pcu),
                None,
                mode,
            )
            .unwrap()
            .to_string();
            assert!(generated.contains("with_numerical_mode"));
            assert!(generated.contains(token));
        }
    }

    #[test]
    fn parses_owned_float_policy_flags_and_rejects_ambiguous_values() {
        let allowed = syn::parse_str::<PcuScalarHelperArgs>(
            "flag(allow_gradual_underflow), crate_path = ::pcu_alias",
        )
        .expect("the gradual-underflow flag parses");
        assert_eq!(
            allowed.underflow_flag,
            Some(PcuOwnedFlag::AllowGradualUnderflow)
        );
        assert!(allowed.crate_path.leading_colon.is_some());
        assert_eq!(
            allowed.crate_path.segments.last().unwrap().ident,
            "pcu_alias"
        );

        let rejected = syn::parse_str::<PcuScalarHelperArgs>("flag(reject_subnormal_result)")
            .expect("the subnormal-result flag parses");
        assert_eq!(
            rejected.underflow_flag,
            Some(PcuOwnedFlag::RejectSubnormalResult)
        );
        let ieee = syn::parse_str::<PcuScalarHelperArgs>("flag(ieee_underflow)")
            .expect("the explicit IEEE flag parses");
        assert_eq!(ieee.underflow_flag, Some(PcuOwnedFlag::IeeeUnderflow));

        for (arguments, diagnostic) in [
            ("flag(unknown)", "unknown `pcu` float flag"),
            (
                "flag(allow_gradual_underflow), flag(allow_gradual_underflow)",
                "duplicate `pcu` float flag",
            ),
            (
                "flag(allow_gradual_underflow), flag(reject_subnormal_result)",
                "conflicting `pcu` float flags",
            ),
        ] {
            let Err(error) = syn::parse_str::<PcuScalarHelperArgs>(arguments) else {
                panic!("invalid policy flag is rejected: {arguments}");
            };
            assert!(
                error.to_string().contains(diagnostic),
                "{arguments}: {error}"
            );
        }
    }

    #[test]
    fn invocation_kernel_flags_parse_and_reject_ambiguous_values() {
        let allowed =
            syn::parse_str::<PcuDispatchArgs>("flag(allow_gradual_underflow), invocations = 64")
                .expect("invocation kernels accept an explicit checked-float policy");
        assert_eq!(
            allowed.underflow_flag,
            Some(PcuOwnedFlag::AllowGradualUnderflow)
        );
        let clamp = syn::parse_str::<PcuDispatchArgs>(
            "flag(clamp_range), flag(allow_gradual_underflow), invocations = 64",
        )
        .expect("clamp and underflow policies are independent");
        assert!(clamp.clamp_range);
        assert_eq!(
            clamp.underflow_flag,
            Some(PcuOwnedFlag::AllowGradualUnderflow)
        );
        let helper_clamp = syn::parse_str::<PcuScalarHelperArgs>("flag(clamp_range)").expect(
            "the parser recognizes clamp so expansion can issue its bounded-profile diagnostic",
        );
        assert!(helper_clamp.clamp_range);
        for (arguments, diagnostic) in [
            (
                "flag(unknown), invocations = 64",
                "unknown `pcu` float flag",
            ),
            (
                "flag(allow_gradual_underflow), flag(allow_gradual_underflow), invocations = 64",
                "duplicate `pcu` float flag",
            ),
            (
                "flag(allow_gradual_underflow), flag(reject_subnormal_result), invocations = 64",
                "conflicting `pcu` float flags",
            ),
            (
                "flag(clamp_range), flag(clamp_range), invocations = 64",
                "duplicate `pcu` clamp flag",
            ),
        ] {
            let Err(error) = syn::parse_str::<PcuDispatchArgs>(arguments) else {
                panic!("invalid invocation policy is rejected: {arguments}");
            };
            assert!(
                error.to_string().contains(diagnostic),
                "{arguments}: {error}"
            );
        }
    }

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
        assert!(generated.contains("load_value"));
        assert!(generated.contains("store_value"));
        assert!(generated.contains("checked_binary_value"), "{generated}");
        assert!(generated.contains("PcuDispatchFloatBinaryOp :: Mul"));
        assert!(generated.contains("with_float_underflow_policy"));
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
        assert!(
            generated.contains("PcuBinding :: scalar :: < f64 >"),
            "{generated}"
        );
        assert!(generated.contains("checked_binary_value"), "{generated}");
        assert!(
            generated.contains("PcuDispatchFloatBinaryOp :: Add"),
            "{generated}"
        );
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
                underflow_flag: None,
                numerical_mode: None,
                clamp_range: false,
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
                underflow_flag: None,
                numerical_mode: None,
                clamp_range: false,
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
        assert!(generated.contains("load_value"));
        assert!(generated.contains("store_value"));
        assert!(generated.contains("PcuGridStrideKernelBuilder"));
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
        assert!(tokens.to_string().contains("PcuGridStrideKernelBuilder"));
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
    fn invocation_f64_to_f32_cast_uses_checked_conversion_and_captured_policy() {
        for (flag, variant) in [
            ("ieee_underflow", "IeeeAfterRounding"),
            ("allow_gradual_underflow", "AllowGradualUnderflow"),
            ("reject_subnormal_result", "RejectSubnormalResult"),
        ] {
            let source = "fn kernel(input: &[f64], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as f32; }";
            let function = syn::parse_str::<ItemFn>(source).expect("function parses");
            let args = syn::parse_str::<PcuDispatchArgs>(&format!("invocations = 8, flag({flag})"))
                .expect("policy parses");
            let generated = expand_pcu_dispatch(args, &function)
                .expect("concrete f64 to f32 cast lowers")
                .to_string();
            assert!(
                generated.contains("checked_f64_to_f32_value"),
                "{generated}"
            );
            assert!(generated.contains(variant), "{generated}");
            assert!(
                !generated.contains("PcuDispatchDataOp :: Convert"),
                "{generated}"
            );
        }

        for literal in ["1.0_f64", "1.0"] {
            let source = format!(
                "fn kernel(output: &mut [f32]) {{ let invocation = context.global_invocation_id; output[invocation] = {literal} as f32; }}"
            );
            let function = syn::parse_str::<ItemFn>(&source).expect("literal cast function parses");
            let args = syn::parse_str::<PcuDispatchArgs>("invocations = 8")
                .expect("default policy parses");
            let generated = expand_pcu_dispatch(args, &function)
                .expect("f64 literal cast lowers")
                .to_string();
            assert!(generated.contains("constant_f64_value"), "{generated}");
            assert!(
                generated.contains("checked_f64_to_f32_value"),
                "{generated}"
            );
        }
    }

    #[test]
    fn invocation_f32_to_f64_cast_uses_checked_conversion_and_scoped_source_kind() {
        for (source, expect_f32_literal) in [
            (
                "fn kernel(input: &[f32], output: &mut [f64]) { let invocation = context.global_invocation_id; output[invocation] = (input[invocation] + 1.0) as f64 + 1.0_f64; }",
                false,
            ),
            (
                "fn kernel(output: &mut [f64]) { let invocation = context.global_invocation_id; output[invocation] = 1.0_f32 as f64; }",
                true,
            ),
        ] {
            let function = syn::parse_str::<ItemFn>(source).expect("function parses");
            let args =
                syn::parse_str::<PcuDispatchArgs>("invocations = 8, flag(reject_subnormal_result)")
                    .expect("policy parses");
            let generated = expand_pcu_dispatch(args, &function)
                .expect("concrete f32 to f64 cast lowers")
                .to_string();
            assert!(
                generated.contains("checked_f32_to_f64_value"),
                "{generated}"
            );
            assert!(
                generated.contains("PcuFloatUnderflowPolicy :: RejectSubnormalResult"),
                "{generated}"
            );
            assert!(
                !generated.contains("PcuDispatchDataOp :: Convert"),
                "{generated}"
            );
            if expect_f32_literal {
                assert!(generated.contains("constant_f32_value"), "{generated}");
                assert!(!generated.contains("constant_f64_value"), "{generated}");
            } else {
                assert!(
                    generated.contains("PcuDispatchFloatBinaryOp :: Add"),
                    "{generated}"
                );
            }
        }
    }

    #[test]
    fn f32_to_f64_cast_in_helper_inherits_scalar_lowering_policy() {
        let helper =
            syn::parse_str::<ItemFn>("fn helper(value: f64) -> f64 { 1.0_f32 as f64 + value }")
                .expect("helper parses");
        let path = syn::parse_quote!(::pcu);
        let generated = expand_pcu_scalar_helper(helper, &path)
            .expect("helper widening cast lowers into companion context")
            .to_string();
        assert!(
            generated.contains("checked_f32_to_f64_value"),
            "{generated}"
        );
        assert!(
            !generated.contains("__pcu_float_underflow_policy"),
            "{generated}"
        );
        assert!(generated.contains("checked_binary_value"), "{generated}");
    }

    #[test]
    fn f64_to_f32_cast_in_helper_inherits_scalar_lowering_policy() {
        let helper =
            syn::parse_str::<ItemFn>("fn helper(value: f32) -> f32 { 1.0_f64 as f32 + value }")
                .expect("helper parses");
        let path = syn::parse_quote!(::pcu);
        let generated = expand_pcu_scalar_helper(helper, &path)
            .expect("helper cast lowers into companion context")
            .to_string();
        assert!(
            generated.contains("checked_f64_to_f32_value"),
            "{generated}"
        );
        assert!(
            !generated.contains("__pcu_float_underflow_policy"),
            "{generated}"
        );
        assert!(generated.contains("checked_binary_value"), "{generated}");

        let mixed = syn::parse_str::<ItemFn>("fn helper(value: f64) -> f32 { value as f32 }")
            .expect("mixed helper signature parses as Rust syntax");
        let error = expand_pcu_scalar_helper(mixed, &path)
            .expect_err("the existing homogeneous helper signature contract remains");
        assert!(
            error
                .to_string()
                .contains("helper parameters must match the declared f32 or f64 result type"),
            "{error}"
        );
    }

    #[test]
    fn rejects_non_f64_to_f32_invocation_casts() {
        for (source, expected) in [
            (
                "fn kernel(input: &[f32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as f32; }",
                "PCU checked cast to f32 requires a concrete f64 source",
            ),
            (
                "fn kernel(input: &[i32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as f32; }",
                "f32 or f64 bindings",
            ),
            (
                "fn kernel(input: &[f64], output: &mut [f64]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as f64; }",
                "PCU checked cast to f64 requires an explicitly typed f32 source",
            ),
            (
                "fn kernel(input: &[i32], output: &mut [f64]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as f64; }",
                "explicitly typed f32 source",
            ),
            (
                "fn kernel(output: &mut [f64]) { let invocation = context.global_invocation_id; output[invocation] = 1.0 as f64; }",
                "unsuffixed float literals default to f64",
            ),
            (
                "fn kernel(input: &[f64], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as u32; }",
                "explicit `f64 as f32` and `f32 as f64`",
            ),
            (
                "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] as f32; }",
                "indexed identity copy",
            ),
        ] {
            let function = syn::parse_str::<ItemFn>(source).expect("function parses");
            let args =
                syn::parse_str::<PcuDispatchArgs>("invocations = 8").expect("arguments parse");
            let error = expand_pcu_dispatch(args, &function).expect_err("unsupported cast rejects");
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn invocation_policy_reaches_runtime_float_lowering_and_nested_helpers() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel(input: &[f32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = helper(input[invocation]) + input[invocation]; }",
        )
        .expect("test function parses");
        let args =
            syn::parse_str::<PcuDispatchArgs>("invocations = 64, flag(reject_subnormal_result)")
                .expect("policy flag parses");
        let generated = expand_pcu_dispatch(args, &function)
            .expect("checked float helper expansion lowers")
            .to_string();
        assert!(
            generated.contains("__kernel_with_float_underflow_policy"),
            "{generated}"
        );
        assert!(
            generated.contains("PcuFloatUnderflowPolicy :: RejectSubnormalResult"),
            "{generated}"
        );
        assert!(generated.contains("checked_binary_value"), "{generated}");
        assert!(generated.contains("__pcu_lower"), "{generated}");

        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 64")
            .expect("default invocation arguments parse");
        let hosted = super::expand_pcu_direct(args, &function)
            .expect("hosted entry expansion lowers")
            .to_string();
        assert!(hosted.contains("float_underflow_policy"), "{hosted}");
        assert!(
            hosted.contains("__kernel_ir_with_float_underflow_policy"),
            "{hosted}"
        );

        let clamp_args = syn::parse_str::<PcuDispatchArgs>(
            "invocations = 64, flag(allow_gradual_underflow), flag(clamp_range)",
        )
        .expect("the independent policies parse");
        let generated = expand_pcu_dispatch(clamp_args, &function)
            .expect("range policy reaches generated lowering")
            .to_string();
        assert!(generated.contains("with_range_policy"), "{generated}");
        assert!(generated.contains("PcuRangePolicy :: Clamp"), "{generated}");
        assert!(
            generated.contains("PcuFloatUnderflowPolicy :: AllowGradualUnderflow"),
            "{generated}"
        );

        let clamp_args = syn::parse_str::<PcuDispatchArgs>("invocations = 64, flag(clamp_range)")
            .expect("clamp flag parses");
        let hosted = super::expand_pcu_direct(clamp_args, &function)
            .expect("hosted range flag expands")
            .to_string();
        assert!(hosted.contains("range_policy"), "{hosted}");
        assert!(hosted.contains("PcuRangePolicy :: Clamp"), "{hosted}");
    }

    #[test]
    fn clamp_range_rejects_integer_invocation_profiles() {
        let function = syn::parse_str::<ItemFn>(
            "fn kernel(input: &[u32], output: &mut [u32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation]; }",
        )
        .expect("test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 8, flag(clamp_range)")
            .expect("clamp flag parses");
        let error = expand_pcu_dispatch(args, &function)
            .expect_err("integer invocation profiles cannot request float range recovery");
        assert!(
            error.to_string().contains("integer and generic profiles"),
            "{error}"
        );

        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 8")
            .expect("unflagged invocation arguments parse");
        let hosted = super::expand_pcu_direct(args, &function)
            .expect("integer invocation wrapper expands")
            .to_string();
        assert!(
            hosted.contains("PcuExecutionError :: UnsupportedRangePolicy"),
            "global clamp is rejected by the cold hosted entry: {hosted}"
        );

        // A float operation does not make a mixed binding profile safe: clamp
        // applies to the invocation as a whole, and integer arithmetic in the
        // same body must not silently retain reject-only semantics.
        let mixed = syn::parse_str::<ItemFn>(
            "fn kernel(input: &[f32], _offset: &[u32], output: &mut [f32]) { let invocation = context.global_invocation_id; output[invocation] = input[invocation] * 2.0_f32; }",
        )
        .expect("mixed test function parses");
        let args = syn::parse_str::<PcuDispatchArgs>("invocations = 8")
            .expect("unflagged invocation arguments parse");
        let hosted = super::expand_pcu_direct(args, &mixed)
            .expect("mixed invocation wrapper still expands under default reject policy")
            .to_string();
        assert!(
            hosted.contains("PcuExecutionError :: UnsupportedRangePolicy"),
            "mixed float/integer bindings reject global clamp at cold admission: {hosted}"
        );
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
        assert!(message.contains("currently support only an indexed identity copy"));
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
        assert!(generated.contains(":: pcu_alias :: PcuDispatchFloatBinaryOp"));
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
