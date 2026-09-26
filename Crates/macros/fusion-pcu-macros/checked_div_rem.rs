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
use quote::quote;
use syn::spanned::Spanned;
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

pub fn lower<'a>(
    function: &'a ItemFn,
    bindings: &[BindingSpec],
    pcu: &Path,
) -> Result<Option<Lowered<'a>>, Error> {
    let Some(body) = parse_checked_div_rem_body(function)? else {
        return Ok(None);
    };
    let lowered = lower_checked_div_rem_body(&body, bindings, pcu)?;
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
    body: &CheckedDivRemBody<'a>,
    bindings: &[BindingSpec],
    pcu: &Path,
) -> Result<(Option<&'a Expr>, Vec<TokenStream2>), Error> {
    if bindings.len() != 4
        || bindings
            .iter()
            .any(|binding| binding.scalar != ScalarKind::U32)
    {
        return Err(Error::new(
            body.lhs.span(),
            "checked u32 DivRem requires exactly four u32 bindings",
        ));
    }
    if bindings[0].access != BindingAccess::ReadOnly
        || bindings[1].access != BindingAccess::ReadOnly
        || bindings[2].access != BindingAccess::ReadWrite
        || bindings[3].access != BindingAccess::ReadWrite
    {
        return Err(Error::new(
            body.lhs.span(),
            "checked u32 DivRem requires two read-only inputs followed by two writable outputs",
        ));
    }
    let mut emitter = ExprEmitter::new(bindings, &body.invocation, pcu, body.extent.is_some());
    let (lhs, lhs_type) = emitter.emit_expr(body.lhs)?;
    let (rhs, rhs_type) = emitter.emit_expr(body.rhs)?;
    if lhs_type != ScalarKind::U32 || rhs_type != ScalarKind::U32 {
        return Err(Error::new(
            body.lhs.span(),
            "checked DivRem requires matching u32 operands",
        ));
    }
    let q_value = emitter.alloc_value(body.quotient.span())?;
    let r_value = emitter.alloc_value(body.remainder.span())?;
    let pcu_ref = pcu;
    emitter.ops.push(quote! {
        #pcu_ref::PcuDispatchDataOp::CheckedDivRem {
            value_type: #pcu_ref::PcuValueType::u32(),
            flags: #pcu_ref::model::PcuIntegerDivFlags::CHECKED,
            quotient: #pcu_ref::PcuDispatchValueId(#q_value),
            remainder: #pcu_ref::PcuDispatchValueId(#r_value),
            lhs: #pcu_ref::PcuDispatchValueId(#lhs),
            rhs: #pcu_ref::PcuDispatchValueId(#rhs),
        }
    });
    let q_binding = validate_div_rem_store(
        body.quotient_store,
        &body.quotient,
        &body.invocation,
        bindings,
    )?;
    let r_binding = validate_div_rem_store(
        body.remainder_store,
        &body.remainder,
        &body.invocation,
        bindings,
    )?;
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
    if output.access != BindingAccess::ReadWrite || output.scalar != ScalarKind::U32 {
        return Err(Error::new(
            index.span(),
            "checked DivRem output must be a writable u32 binding",
        ));
    }
    if expr_ident(&assignment.right).is_none_or(|ident| ident != expected_value) {
        return Err(Error::new(
            assignment.right.span(),
            "each checked DivRem output must be stored exactly once to its matching binding",
        ));
    }
    Ok(output)
}
