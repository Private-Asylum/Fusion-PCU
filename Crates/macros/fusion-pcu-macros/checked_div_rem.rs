#[rustfmt::skip]
use super::{
    BindingAccess,
    BindingSpec,
    ExprEmitter,
    ScalarKind,
    expr_ident,
    is_context_invocation_count_expr,
    is_context_invocation_expr,
    is_ident_expr,
    validate_invocation_index,
};
use proc_macro2::TokenStream as TokenStream2;
#[rustfmt::skip]
use quote::{
    format_ident,
    quote,
};
use syn::spanned::Spanned;
#[rustfmt::skip]
use syn::{
    BinOp,
    Error,
    Expr,
    ExprAssign,
    Ident,
    ItemFn,
    Pat,
    Path,
    Stmt,
};

type Lowered<'a> = (Option<&'a Expr>, Vec<TokenStream2>);

#[cfg(test)]
#[path = "checked_div_rem/tests/tests.rs"]
mod tests;

struct CheckedDivRemBody<'a> {
    invocation: Ident,
    lhs: &'a Expr,
    rhs: &'a Expr,
    quotient: Ident,
    remainder: Ident,
    quotient_store: &'a ExprAssign,
    remainder_store: &'a ExprAssign,
    extent: Option<&'a Expr>,
}

/// An associated const specializes the typed body without allocation or an
/// invalid function-local const capturing the outer generic parameter.
pub fn generic_grid_operations(
    data_ops: &[TokenStream2],
    extent: &TokenStream2,
    function: &Ident,
    scalar: &Ident,
    pcu: &Path,
) -> (TokenStream2, usize, TokenStream2) {
    let body_ident = format_ident!("__{}_PcuDivRemBody", function);
    let body_len = data_ops.len();
    let body_ops = data_ops
        .iter()
        .map(|op| quote! { #pcu::PcuDispatchOp::Data(#op) });
    let body_item = quote! {
        #[allow(non_camel_case_types)]
        struct #body_ident<#scalar: #pcu::PcuCheckedIntegerDivision>(::core::marker::PhantomData<#scalar>);
        impl<#scalar: #pcu::PcuCheckedIntegerDivision> #body_ident<#scalar> {
            const BODY: [#pcu::PcuDispatchOp<'static>; #body_len] = [#(#body_ops),*];
        }
    };
    let operation = quote! {
        let builder = builder.with_op(#pcu::PcuDispatchOp::GridStrideLoop {
            extent: const {
                let extent: usize = #extent;
                assert!(extent != 0, "PCU grid-stride extent must be nonzero");
                assert!(extent <= u32::MAX as usize, "PCU grid-stride extent exceeds u32");
                extent as u32
            },
            body: &#body_ident::<#scalar>::BODY,
        })?;
    };
    (operation, 2, body_item)
}

pub fn lower<'a>(
    function: &'a ItemFn,
    bindings: &[BindingSpec],
    pcu: &Path,
) -> Result<Option<Lowered<'a>>, Error> {
    let Some(body) = parse_checked_div_rem_body(function)? else {
        return Ok(None);
    };
    let lowered = lower_checked_div_rem_body(function, &body, bindings, pcu)?;
    Ok(Some(lowered))
}

#[allow(clippy::too_many_lines)]
fn parse_checked_div_rem_body(function: &ItemFn) -> Result<Option<CheckedDivRemBody<'_>>, Error> {
    let statements = &function.block.stmts;
    if statements.len() == 4
        && matches!(statements.get(1), Some(Stmt::Local(local)) if matches!(local.pat, Pat::Tuple(_)))
    {
        let invocation = checked_invocation_binding(&statements[0])?;
        let (quotient, remainder, lhs, rhs) = checked_div_rem_local(&statements[1])?;
        let Stmt::Expr(Expr::Assign(quotient_store), Some(_)) = &statements[2] else {
            return Err(Error::new(
                statements[2].span(),
                "checked DivRem requires a quotient store",
            ));
        };
        let Stmt::Expr(Expr::Assign(remainder_store), Some(_)) = &statements[3] else {
            return Err(Error::new(
                statements[3].span(),
                "checked DivRem requires a remainder store",
            ));
        };
        return Ok(Some(CheckedDivRemBody {
            invocation,
            lhs,
            rhs,
            quotient,
            remainder,
            quotient_store,
            remainder_store,
            extent: None,
        }));
    }
    if statements.len() == 3
        && matches!(statements.get(2), Some(Stmt::Expr(Expr::While(loop_expr), None)) if matches!(loop_expr.body.stmts.first(), Some(Stmt::Local(local)) if matches!(local.pat, Pat::Tuple(_))))
    {
        let id_local = checked_grid_local(&statements[0], true)?;
        let stride_local = checked_grid_local(&statements[1], false)?;
        let Stmt::Expr(Expr::While(loop_expr), None) = &statements[2] else {
            unreachable!()
        };
        let Expr::Binary(condition) = loop_expr.cond.as_ref() else {
            return Err(Error::new(
                loop_expr.cond.span(),
                "checked DivRem grid loop requires `id < extent`",
            ));
        };
        if !matches!(condition.op, BinOp::Lt(_)) || !is_ident_expr(&condition.left, &id_local) {
            return Err(Error::new(
                condition.span(),
                "checked DivRem grid loop requires canonical `id < extent`",
            ));
        }
        let [
            tuple,
            quotient_statement,
            remainder_statement,
            increment_statement,
        ] = loop_expr.body.stmts.as_slice()
        else {
            return Err(Error::new(
                loop_expr.body.span(),
                "checked DivRem grid loop requires two output stores and the canonical stride increment",
            ));
        };
        let (quotient, remainder, lhs, rhs) = checked_div_rem_local(tuple)?;
        let Stmt::Expr(Expr::Assign(quotient_store), Some(_)) = quotient_statement else {
            return Err(Error::new(
                quotient_statement.span(),
                "checked DivRem requires a quotient store",
            ));
        };
        let Stmt::Expr(Expr::Assign(remainder_store), Some(_)) = remainder_statement else {
            return Err(Error::new(
                remainder_statement.span(),
                "checked DivRem requires a remainder store",
            ));
        };
        let Stmt::Expr(Expr::Binary(increment), Some(_)) = increment_statement else {
            return Err(Error::new(
                increment_statement.span(),
                "checked DivRem grid loop requires canonical stride increment",
            ));
        };
        if !matches!(increment.op, BinOp::AddAssign(_))
            || !is_ident_expr(&increment.left, &id_local)
            || !is_ident_expr(&increment.right, &stride_local)
        {
            return Err(Error::new(
                increment.span(),
                "checked DivRem grid loop requires canonical `id += stride`",
            ));
        }
        return Ok(Some(CheckedDivRemBody {
            invocation: id_local,
            lhs,
            rhs,
            quotient,
            remainder,
            quotient_store,
            remainder_store,
            extent: Some(&condition.right),
        }));
    }
    Ok(None)
}

fn checked_invocation_binding(statement: &Stmt) -> Result<Ident, Error> {
    let Stmt::Local(local) = statement else {
        return Err(Error::new(
            statement.span(),
            "checked DivRem must bind the invocation first",
        ));
    };
    let Pat::Ident(pattern) = &local.pat else {
        return Err(Error::new(
            local.pat.span(),
            "invocation binding must be a plain identifier",
        ));
    };
    let Some(init) = &local.init else {
        return Err(Error::new(
            local.pat.span(),
            "invocation binding must initialize from the global invocation id",
        ));
    };
    if pattern.mutability.is_some()
        || pattern.by_ref.is_some()
        || pattern.subpat.is_some()
        || init.diverge.is_some()
        || !is_context_invocation_expr(&init.expr)
    {
        return Err(Error::new(
            local.span(),
            "first checked DivRem statement must bind the global invocation id",
        ));
    }
    Ok(pattern.ident.clone())
}

fn checked_grid_local(statement: &Stmt, mutable: bool) -> Result<Ident, Error> {
    let Stmt::Local(local) = statement else {
        return Err(Error::new(
            statement.span(),
            "checked DivRem grid loop requires invocation and stride locals",
        ));
    };
    let Pat::Ident(pattern) = &local.pat else {
        return Err(Error::new(
            local.pat.span(),
            "grid local must be a plain identifier",
        ));
    };
    let Some(init) = &local.init else {
        return Err(Error::new(
            local.pat.span(),
            "grid local requires an initializer",
        ));
    };
    if pattern.mutability.is_some() != mutable
        || pattern.by_ref.is_some()
        || pattern.subpat.is_some()
        || init.diverge.is_some()
    {
        return Err(Error::new(
            local.span(),
            "checked DivRem grid locals must use canonical mutability",
        ));
    }
    let valid_init = if mutable {
        is_context_invocation_expr(&init.expr)
    } else {
        is_context_invocation_count_expr(&init.expr)
    };
    if !valid_init {
        return Err(Error::new(
            init.expr.span(),
            "checked DivRem grid locals must bind global invocation id and invocation count",
        ));
    }
    Ok(pattern.ident.clone())
}

fn checked_div_rem_local(statement: &Stmt) -> Result<(Ident, Ident, &Expr, &Expr), Error> {
    let Stmt::Local(local) = statement else {
        return Err(Error::new(
            statement.span(),
            "checked DivRem must bind `(quotient, remainder)`",
        ));
    };
    let Pat::Tuple(tuple) = &local.pat else {
        return Err(Error::new(
            local.pat.span(),
            "checked DivRem result must use a two-name tuple pattern",
        ));
    };
    if tuple.elems.len() != 2 || !local.attrs.is_empty() {
        return Err(Error::new(
            tuple.span(),
            "checked DivRem result requires exactly two names",
        ));
    }
    let (Pat::Ident(q), Pat::Ident(r)) = (&tuple.elems[0], &tuple.elems[1]) else {
        return Err(Error::new(
            tuple.span(),
            "checked DivRem result requires two plain names",
        ));
    };
    if q.mutability.is_some()
        || q.by_ref.is_some()
        || q.subpat.is_some()
        || r.mutability.is_some()
        || r.by_ref.is_some()
        || r.subpat.is_some()
        || q.ident == r.ident
    {
        return Err(Error::new(
            tuple.span(),
            "checked DivRem result names must be distinct immutable identifiers",
        ));
    }
    let Some(init) = &local.init else {
        return Err(Error::new(
            tuple.span(),
            "checked DivRem result tuple requires an initializer",
        ));
    };
    if init.diverge.is_some() {
        return Err(Error::new(
            init.expr.span(),
            "checked DivRem tuple initializer cannot diverge",
        ));
    }
    let Expr::Call(call) = init.expr.as_ref() else {
        return Err(Error::new(
            init.expr.span(),
            "use `pcu::checked_div_rem(lhs, rhs)` for checked integer division",
        ));
    };
    let Expr::Path(path) = call.func.as_ref() else {
        return Err(Error::new(
            call.func.span(),
            "checked DivRem must call `pcu::checked_div_rem`",
        ));
    };
    let segments = &path.path.segments;
    let valid_path = segments
        .last()
        .is_some_and(|segment| segment.ident == "checked_div_rem")
        && segments.iter().any(|segment| segment.ident == "pcu");
    if !valid_path || path.qself.is_some() || call.args.len() != 2 {
        return Err(Error::new(
            call.span(),
            "checked DivRem must call `pcu::checked_div_rem(lhs, rhs)` with two arguments",
        ));
    }
    Ok((
        q.ident.clone(),
        r.ident.clone(),
        &call.args[0],
        &call.args[1],
    ))
}

fn lower_checked_div_rem_body<'a>(
    function: &ItemFn,
    body: &CheckedDivRemBody<'a>,
    bindings: &[BindingSpec],
    pcu: &Path,
) -> Result<(Option<&'a Expr>, Vec<TokenStream2>), Error> {
    if !(3..=4).contains(&bindings.len()) {
        return Err(Error::new(
            body.lhs.span(),
            "checked DivRem requires one or two read-only inputs and two matching integer outputs",
        ));
    }
    let scalar = bindings[0].scalar;
    let scalar_ty = division_scalar(function, bindings, body.lhs)?;
    if bindings
        .iter()
        .filter(|binding| binding.access == BindingAccess::ReadWrite)
        .count()
        != 2
    {
        return Err(Error::new(
            body.lhs.span(),
            "checked DivRem requires exactly two writable outputs, independent of declaration order",
        ));
    }
    let mut emitter = ExprEmitter::new(
        bindings,
        &body.invocation,
        pcu,
        body.extent.is_some(),
        ScalarKind::Generic,
    );
    let (lhs, lhs_type) = emit_operand(&mut emitter, body.lhs)?;
    let (rhs, rhs_type) = emit_operand(&mut emitter, body.rhs)?;
    if lhs_type != scalar || rhs_type != scalar {
        return Err(Error::new(
            body.lhs.span(),
            "checked DivRem requires matching fixed-width integer operands",
        ));
    }
    if emitter.ops.len() != 2 {
        return Err(Error::new(
            body.lhs.span(),
            "checked DivRem operands must each be one read-only resource load",
        ));
    }
    let q_value = emitter.alloc_value(body.quotient.span())?;
    let r_value = emitter.alloc_value(body.remainder.span())?;
    let pcu_ref = pcu;
    emitter.ops.push(quote! {
        #pcu_ref::PcuDispatchDataOp::CheckedDivRem {
            value_type: #pcu_ref::PcuValueType::Scalar(<#scalar_ty as #pcu_ref::PcuScalar>::TYPE),
            flags: #pcu_ref::model::PcuIntegerDivFlags::CHECKED,
            quotient: #pcu_ref::PcuDispatchValueId(#q_value),
            remainder: #pcu_ref::PcuDispatchValueId(#r_value),
            lhs: #pcu_ref::PcuDispatchValueId(#lhs),
            rhs: #pcu_ref::PcuDispatchValueId(#rhs),
        }
    });
    let (quotient_store, remainder_store) =
        if is_ident_expr(&body.quotient_store.right, &body.remainder)
            && is_ident_expr(&body.remainder_store.right, &body.quotient)
        {
            (body.remainder_store, body.quotient_store)
        } else {
            (body.quotient_store, body.remainder_store)
        };
    let q_binding =
        validate_div_rem_store(quotient_store, &body.quotient, &body.invocation, bindings)?;
    let r_binding =
        validate_div_rem_store(remainder_store, &body.remainder, &body.invocation, bindings)?;
    if q_binding.binding == r_binding.binding {
        return Err(Error::new(
            body.remainder_store.left.span(),
            "checked DivRem outputs must be distinct",
        ));
    }
    let q_output_slot = q_binding.binding;
    let r_output_slot = r_binding.binding;
    let store_index = if body.extent.is_some() {
        quote! { GridStrideId }
    } else {
        quote! { InvocationId }
    };
    emitter.ops.push(quote! { #pcu_ref::PcuDispatchDataOp::BindingStore { binding: #pcu_ref::PcuBindingRef::new(0, #q_output_slot), index: #pcu_ref::PcuDispatchIndex::#store_index, value: #pcu_ref::PcuDispatchValueId(#q_value) } });
    emitter.ops.push(quote! { #pcu_ref::PcuDispatchDataOp::BindingStore { binding: #pcu_ref::PcuBindingRef::new(0, #r_output_slot), index: #pcu_ref::PcuDispatchIndex::#store_index, value: #pcu_ref::PcuDispatchValueId(#r_value) } });
    Ok((body.extent, emitter.ops))
}

/// Division's stronger type bound permits scalar reads without broadening ordinary expressions.
fn emit_operand(
    emitter: &mut ExprEmitter<'_>,
    expression: &Expr,
) -> Result<(u16, ScalarKind), Error> {
    match expression {
        Expr::Paren(paren) => return emit_operand(emitter, &paren.expr),
        Expr::Group(group) => return emit_operand(emitter, &group.expr),
        Expr::Index(index) => {
            let ident = expr_ident(&index.expr).ok_or_else(|| {
                Error::new(
                    index.expr.span(),
                    "checked DivRem indexed operand must name a binding",
                )
            })?;
            if emitter.binding(ident, BindingAccess::ReadOnly)?.access != BindingAccess::ReadOnly {
                return Err(Error::new(
                    expression.span(),
                    "checked DivRem operands require read-only bindings",
                ));
            }
            return emitter.emit_expr(expression);
        }
        _ => {}
    }
    let scalar = match expression {
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => unary.expr.as_ref(),
        Expr::Path(_) => expression,
        _ => {
            return Err(Error::new(
                expression.span(),
                "checked DivRem operands must be read-only resource loads",
            ));
        }
    };
    let ident = expr_ident(scalar).ok_or_else(|| {
        Error::new(
            scalar.span(),
            "checked DivRem scalar operand must name a binding",
        )
    })?;
    let (slot, scalar) = {
        let binding = emitter.binding(ident, BindingAccess::ReadOnly)?;
        if !binding.scalar_reference || binding.access != BindingAccess::ReadOnly {
            return Err(Error::new(
                expression.span(),
                "checked DivRem bare operands require read-only scalar bindings",
            ));
        }
        (binding.binding, binding.scalar)
    };
    let result = emitter.alloc_value(expression.span())?;
    let pcu = emitter.crate_path;
    // Each authored operand retains its own SSA load even when both name the same scalar.
    emitter.ops.push(quote! {
        #pcu::PcuDispatchDataOp::BindingLoad {
            result: #pcu::PcuDispatchValueId(#result),
            binding: #pcu::PcuBindingRef::new(0, #slot),
            index: #pcu::PcuDispatchIndex::BindingElementZero,
        }
    });
    Ok((result, scalar))
}

/// The source vocabulary is broader than any provider's admitted execution ABI.
/// Generic division requires the stronger sealed bound; `PcuScalar` alone does not
/// establish integer semantics. No wrapping, total, or Clamp division is introduced.
fn division_scalar(
    function: &ItemFn,
    bindings: &[BindingSpec],
    operand: &Expr,
) -> Result<TokenStream2, Error> {
    let first = &bindings[0];
    let scalar = first.scalar;
    let concrete_integer = matches!(
        scalar,
        ScalarKind::U8
            | ScalarKind::U16
            | ScalarKind::U32
            | ScalarKind::U64
            | ScalarKind::U128
            | ScalarKind::I8
            | ScalarKind::I16
            | ScalarKind::I32
            | ScalarKind::I64
            | ScalarKind::I128
    );
    let generic_integer =
        scalar == ScalarKind::Generic && super::generic_integer::has_division_bound(function);
    if !(concrete_integer || generic_integer)
        || bindings.iter().any(|binding| {
            binding.scalar != scalar
                || binding.generic_scalar != first.generic_scalar
                || binding.scalar_reference && binding.access != BindingAccess::ReadOnly
                || binding.matrix.is_some()
        })
    {
        return Err(Error::new_spanned(
            operand,
            "checked DivRem requires matching integer slice/array outputs and read-only slice/array/scalar inputs; generic resources require `T: PcuCheckedIntegerDivision`",
        ));
    }
    Ok(first
        .generic_scalar
        .as_ref()
        .map_or_else(|| scalar.rust_type(), |generic| quote! { #generic }))
}

fn validate_div_rem_store<'a>(
    assignment: &ExprAssign,
    expected_value: &Ident,
    invocation: &Ident,
    bindings: &'a [BindingSpec],
) -> Result<&'a BindingSpec, Error> {
    let Expr::Index(index) = assignment.left.as_ref() else {
        return Err(Error::new(
            assignment.left.span(),
            "checked DivRem output must be an indexed binding",
        ));
    };
    validate_invocation_index(&index.index, invocation)?;
    let Some(ident) = expr_ident(&index.expr) else {
        return Err(Error::new(
            index.expr.span(),
            "checked DivRem output must name a binding",
        ));
    };
    let output = bindings
        .iter()
        .find(|binding| binding.ident == *ident)
        .ok_or_else(|| Error::new(ident.span(), "unknown PCU binding"))?;
    if output.access != BindingAccess::ReadWrite || output.scalar != bindings[0].scalar {
        return Err(Error::new(
            index.span(),
            "checked DivRem output must be a writable binding with the input integer type",
        ));
    }
    if expr_ident(&assignment.right).is_none_or(|ident| ident != expected_value) {
        return Err(Error::new(
            assignment.right.span(),
            "each checked DivRem output must be stored exactly once to its matching binding",
        ));
    }
    output.written.set(true);
    Ok(output)
}
