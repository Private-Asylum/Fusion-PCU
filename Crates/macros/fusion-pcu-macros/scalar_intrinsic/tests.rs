use super::relu_operand;
#[rustfmt::skip]
use syn::{ExprCall, Path};

#[test]
fn only_reserved_or_exact_crate_qualified_paths_are_intrinsics() {
    let crate_path: Path = syn::parse_quote!(::renamed::facade);
    for expression in [
        "pcu::relu(input[id])",
        "::renamed::facade::pcu::relu(input[id])",
    ] {
        let call: ExprCall = syn::parse_str(expression).unwrap();
        assert!(relu_operand(&call, &crate_path).unwrap().is_some());
    }
    for expression in [
        "relu(input[id])",
        "unrelated::pcu::relu(input[id])",
        "renamed::other::pcu::relu(input[id])",
        "pcu::not_relu(input[id])",
    ] {
        let call: ExprCall = syn::parse_str(expression).unwrap();
        assert!(relu_operand(&call, &crate_path).unwrap().is_none());
    }
}

#[test]
fn malformed_reserved_intrinsics_fail_in_the_frontend() {
    let crate_path: Path = syn::parse_quote!(::fusion_pcu);
    for expression in ["pcu::relu()", "pcu::relu(a, b)", "pcu::relu::<f32>(a)"] {
        let call: ExprCall = syn::parse_str(expression).unwrap();
        assert!(relu_operand(&call, &crate_path).is_err());
    }
}
