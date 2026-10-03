//! Bounded generic checked-float source contract, independent of backend admission.

#[rustfmt::skip]
use super::{
    BindingSpec,
    ScalarKind,
};
#[rustfmt::skip]
use syn::{
    Error,
    Expr,
    ExprAssign,
    GenericParam,
    Ident,
    ItemFn,
    Path,
    TypeParamBound,
};

pub fn has_bound(function: &ItemFn) -> bool {
    function.sig.generics.params.iter().any(|parameter| {
        matches!(parameter, GenericParam::Type(parameter)
        if parameter.bounds.iter().any(|bound| {
            matches!(bound, TypeParamBound::Trait(bound)
                if bound.path.segments.last().is_some_and(|segment|
                    segment.ident == "PcuCheckedFloat"))
        }))
    })
}

pub fn validate(
    assignment: &ExprAssign,
    bindings: &[BindingSpec],
    scalar: &Ident,
    crate_path: &Path,
) -> Result<(), Error> {
    if bindings.iter().any(|binding| {
        binding.scalar != ScalarKind::Generic || binding.generic_scalar.as_ref() != Some(scalar)
    }) {
        return Err(Error::new_spanned(
            assignment,
            "generic checked floating maps require homogeneous `T: PcuCheckedFloat` resources",
        ));
    }
    validate_expression(&assignment.right, crate_path)
}

pub fn validate_expression(expression: &Expr, crate_path: &Path) -> Result<(), Error> {
    match expression {
        Expr::Index(_) | Expr::Path(_) => Ok(()),
        Expr::Paren(paren) => validate_expression(&paren.expr, crate_path),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            validate_expression(&unary.expr, crate_path)
        }
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            validate_expression(&unary.expr, crate_path)
        }
        Expr::Binary(binary) => {
            validate_expression(&binary.left, crate_path)?;
            validate_expression(&binary.right, crate_path)
        }
        Expr::Call(call) => super::scalar_intrinsic::relu_operand(call, crate_path)?.map_or_else(
            || Err(unsupported(expression)),
            |operand| validate_expression(operand, crate_path),
        ),
        _ => Err(unsupported(expression)),
    }
}

fn unsupported(expression: &Expr) -> Error {
    Error::new_spanned(
        expression,
        "generic checked floating expressions support indexed resources, unary negation, pcu::relu, and + - * /; typed literals, casts, and helper calls require a concrete float profile",
    )
}
