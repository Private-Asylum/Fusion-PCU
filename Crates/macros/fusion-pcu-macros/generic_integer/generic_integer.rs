//! Bounded homogeneous checked-integer maps, independently of constant/backend coverage.
#[rustfmt::skip]
use super::{
    BindingSpec,
    ScalarKind,
};
#[rustfmt::skip]
use syn::{
    BinOp,
    Error,
    Expr,
    ExprAssign,
    GenericParam,
    Ident,
    ItemFn,
    TypeParamBound,
};

pub fn has_bound(function: &ItemFn) -> bool {
    function.sig.generics.params.iter().any(|parameter| {
        matches!(parameter, GenericParam::Type(parameter)
        if parameter.bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound)
                if bound.path.segments.last().is_some_and(|segment|
                    segment.ident == "PcuCheckedInteger"
                        || segment.ident == "PcuCheckedIntegerDivision"))
        }))
    })
}

pub fn has_division_bound(function: &ItemFn) -> bool {
    function.sig.generics.params.iter().any(|parameter| {
        matches!(parameter, GenericParam::Type(parameter)
        if parameter.bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound)
                if bound.path.segments.last().is_some_and(|segment|
                    segment.ident == "PcuCheckedIntegerDivision"))
        }))
    })
}

pub fn validate(
    assignment: &ExprAssign,
    bindings: &[BindingSpec],
    scalar: &Ident,
) -> Result<(), Error> {
    if bindings.iter().any(|binding| {
        binding.scalar != ScalarKind::Generic || binding.generic_scalar.as_ref() != Some(scalar)
    }) {
        return Err(Error::new_spanned(
            assignment,
            "generic checked integer maps require homogeneous `T: PcuCheckedInteger` resources",
        ));
    }
    validate_expression(&assignment.right)
}

pub fn validate_expression(expression: &Expr) -> Result<(), Error> {
    match expression {
        Expr::Index(_) | Expr::Path(_) => Ok(()),
        Expr::Paren(paren) => validate_expression(&paren.expr),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            validate_expression(&unary.expr)
        }
        Expr::Binary(binary)
            if matches!(binary.op, BinOp::Add(_) | BinOp::Sub(_) | BinOp::Mul(_)) =>
        {
            validate_expression(&binary.left)?;
            validate_expression(&binary.right)
        }
        _ => Err(Error::new_spanned(
            expression,
            "generic checked integer maps support indexed resources, read-only scalar borrows, and + - *; literals, casts, negation, division and helper calls require a separately supported profile",
        )),
    }
}
