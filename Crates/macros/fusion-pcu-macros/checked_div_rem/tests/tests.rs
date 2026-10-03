//! Generic scalar availability must not authorize arbitrary division semantics.
use crate::expand_pcu_dispatch;

fn function(bound: &str) -> syn::ItemFn {
    syn::parse_str(&format!(
        "fn kernel<T: {bound}>(lhs: &[T], rhs: &[T], quotient: &mut [T], remainder: &mut [T]) {{
            let id = pcu::context::global_invocation_id();
            let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
            quotient[id] = q;
            remainder[id] = r;
        }}"
    ))
    .unwrap()
}

#[test]
fn generic_division_requires_its_operation_bound() {
    for bound in [
        "PcuScalar",
        "PcuCheckedInteger",
        "PcuWrappingInteger",
        "PcuCheckedFloat",
    ] {
        let error =
            expand_pcu_dispatch(syn::parse_quote!(invocations = 7), &function(bound)).unwrap_err();
        assert!(error.to_string().contains("PcuCheckedIntegerDivision"));
    }
}

#[test]
fn division_does_not_acquire_clamp_or_total_semantics() {
    let error = expand_pcu_dispatch(
        syn::parse_quote!(invocations = 7, flag(clamp_range)),
        &function("PcuCheckedIntegerDivision"),
    )
    .unwrap_err();
    assert!(error.to_string().contains("flag(clamp_range)"));
    let error = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 7, flag(div_0_is_0)")
        .err()
        .expect("reserved total division flag remains rejected");
    assert_eq!(error.to_string(), "unknown `pcu` float flag");
}

#[test]
fn role_extensions_preserve_resource_and_dual_result_laws() {
    for body in [
        "let (q, r) = pcu::checked_div_rem(lhs[id] + rhs[id], rhs[id]); quotient[id] = q; remainder[id] = r;",
        "let (q, r) = pcu::checked_div_rem(quotient[id], rhs[id]); quotient[id] = q; remainder[id] = r;",
        "let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]); quotient[id] = q; quotient[id] = r;",
        "let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]); quotient[id] = q; remainder[id] = q;",
        "let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]); quotient[0] = q; remainder[id] = r;",
    ] {
        let source: syn::ItemFn = syn::parse_str(&format!(
            "fn kernel<T: PcuCheckedIntegerDivision>(lhs: &[T], rhs: &[T], quotient: &mut [T], remainder: &mut [T]) {{
                let id = pcu::context::global_invocation_id(); {body}
            }}"
        )).unwrap();
        assert!(
            expand_pcu_dispatch(syn::parse_quote!(invocations = 7), &source).is_err(),
            "{body}"
        );
    }
    let wrong_roles: syn::ItemFn = syn::parse_quote! {
        fn kernel<T: PcuCheckedIntegerDivision>(lhs: &[T], rhs: &[T], quotient: &mut [T], remainder: &[T]) {
            let id = pcu::context::global_invocation_id();
            let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
            quotient[id] = q;
            remainder[id] = r;
        }
    };
    assert!(expand_pcu_dispatch(syn::parse_quote!(invocations = 7), &wrong_roles).is_err());
}
