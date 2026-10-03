//! Unsupported wide operations must stop at source lowering, not native Rust execution.
use crate::expand_pcu_dispatch;
#[rustfmt::skip]
use syn::{
    ItemFn,
    parse_quote,
};

#[test]
fn wide_constants_require_borrowed_resources_until_inline_encoding_exists() {
    for expression in ["input[id] + 1", "input[id] + 1_i128", "input[id] + -1_i128"] {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn wide(input: &[i128], output: &mut [i128]) {{ let id = context.global_invocation_id; output[id] = {expression}; }}"
        ))
        .unwrap();
        let error = expand_pcu_dispatch(parse_quote!(invocations = 7), &function).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("wide integer literals require a borrowed scalar")
        );
    }
}

#[test]
fn concrete_wide_type_does_not_authorize_division_or_legacy_wrapping() {
    for expression in ["input[id] / input[id]", "input[id].wrapping_add(input[id])"] {
        let function = syn::parse_str::<ItemFn>(&format!(
            "fn wide(input: &[u128], output: &mut [u128]) {{ let id = context.global_invocation_id; output[id] = {expression}; }}"
        ))
        .unwrap();
        let error = expand_pcu_dispatch(parse_quote!(invocations = 7), &function).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("integer division is unsupported")
                || error
                    .to_string()
                    .contains("concrete 128-bit wrapping is not yet supported")
        );
    }
}
