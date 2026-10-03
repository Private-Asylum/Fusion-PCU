//! Generic transport source has no numerical operation or unchecked escape.

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
    Ident,
};

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
            "generic transport requires homogeneous `T: PcuScalar` resources",
        ));
    }
    validate_expression(&assignment.right)
}

/// The typed emitter separately resolves local/resource names and canonical
/// indices. This allowlist cannot add arithmetic, conversion or literal values.
pub fn validate_expression(expression: &Expr) -> Result<(), Error> {
    match expression {
        Expr::Index(_) | Expr::Path(_) => Ok(()),
        Expr::Paren(paren) => validate_expression(&paren.expr),
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Deref(_)) => {
            validate_expression(&unary.expr)
        }
        _ => Err(Error::new_spanned(
            expression,
            "generic scalar transport permits resource loads and scalar local reuse; arithmetic requires a checked or explicitly permitted numeric bound",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_and_opaque_expressions_never_gain_transport_permission() {
        for source in [
            "input[id] + input[id]",
            "input[id] / denominator[id]",
            "-input[id]",
            "input[id] as f32",
            "helper(input[id])",
            "pcu::relu(input[id])",
            "1.0",
            "if ready { input[id] } else { seed }",
        ] {
            let expression = syn::parse_str::<Expr>(source).unwrap();
            assert!(validate_expression(&expression).is_err(), "{source}");
        }
    }

    #[test]
    fn transport_locals_use_typed_preparation_instead_of_the_exact_identity_builder() {
        for source in [
            "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let mut value = input[id]; value = input[0]; output[id] = value; }",
            "fn kernel<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < N { let value = input[id]; output[id] = value; id += stride; } }",
        ] {
            let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
            let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 3").unwrap();
            let generated = crate::expand_pcu_direct(arguments, &function)
                .unwrap()
                .to_string();
            assert!(generated.contains("PcuScalarLowering"));
            assert!(!generated.contains("PcuScalarIdentityBuilder"));
        }
    }

    #[test]
    fn transport_preserves_inert_clamp_without_granting_numeric_recovery() {
        for source in [
            "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id]; }",
            "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let saved = input[id]; output[id] = saved; }",
        ] {
            let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
            for arguments in ["invocations = 3", "invocations = 3, flag(clamp_range)"] {
                let arguments = syn::parse_str::<crate::PcuDispatchArgs>(arguments).unwrap();
                let generated = crate::expand_pcu_direct(arguments, &function)
                    .unwrap()
                    .to_string();
                assert!(!generated.contains("PcuExecutionError :: UnsupportedRangePolicy"));
                assert!(!generated.contains("checked_float"));
                assert!(!generated.contains("checked_integer"));
            }
        }
        let wrapping = syn::parse_str::<syn::ItemFn>(
            "fn kernel<T: PcuWrappingInteger>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] + input[id]; }",
        )
        .unwrap();
        let arguments =
            syn::parse_str::<crate::PcuDispatchArgs>("invocations = 3, flag(clamp_range)").unwrap();
        assert!(crate::expand_pcu_direct(arguments, &wrapping).is_err());
    }
}

#[cfg(test)]
mod declaration_tests {
    #[test]
    fn simple_transport_preserves_unused_reordered_and_mutable_declarations() {
        for source in [
            "fn kernel<T: PcuScalar>(input: &[T], ghost: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id]; }",
            "fn kernel<T: PcuScalar>(ghost: &mut [T], output: &mut [T], input: &[T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id]; }",
            "fn kernel<T: PcuScalar>(output: &mut [T], input: &[T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id]; }",
            "fn kernel<T: PcuScalar>(output: &mut [T], input: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id]; }",
            "fn kernel<T: PcuScalar>(output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = output[id]; }",
            "fn kernel<T: PcuScalar>(input: &T, ghost: &mut [T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = *input; }",
            "fn kernel<T: PcuScalar>(input: &[T], ghost: &mut [T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { output[id] = input[id]; id += stride; } }",
        ] {
            let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
            let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 3").unwrap();
            let generated = crate::expand_pcu_direct(arguments, &function)
                .unwrap()
                .to_string();
            assert!(generated.contains("PcuScalarLowering"), "{generated}");
            assert!(
                !generated.contains("PcuScalarIdentityBuilder"),
                "{generated}"
            );
        }
    }

    #[test]
    fn expanded_declarations_do_not_admit_arithmetic_casts_or_arbitrary_indices() {
        for expression in [
            "input[id] + input[id]",
            "input[id] as f32",
            "input[1]",
            "helper(input[id])",
        ] {
            let source = format!(
                "fn kernel<T: PcuScalar>(input: &[T], ghost: &mut [T], output: &mut [T]) {{ let id = pcu::context::global_invocation_id(); output[id] = {expression}; }}"
            );
            let function = syn::parse_str::<syn::ItemFn>(&source).unwrap();
            let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 3").unwrap();
            assert!(
                crate::expand_pcu_direct(arguments, &function).is_err(),
                "{source}"
            );
        }
    }
}
