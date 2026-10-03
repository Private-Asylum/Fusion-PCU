use super::super::expand_pcu_scalar_helper;

#[test]
fn bounded_scalar_helper_modes_keep_checked_operators_and_refuse_other_policies() {
    let function = syn::parse_quote! { fn helper(value: f32) -> f32 { value + value } };
    for flag in [
        "clamp_range",
        "ieee_underflow",
        "backend_precision",
        "native_compound",
        "deterministic",
    ] {
        let arguments = syn::parse_str(&format!("flag({flag})")).unwrap();
        assert!(
            super::super::validate_scalar_helper_policy(&function, &arguments).is_err(),
            "{flag} was accepted without a scalar-helper policy implementation"
        );
    }
    let arguments = syn::parse_str("crate_path=::pcu_alias").unwrap();
    super::super::validate_scalar_helper_policy(&function, &arguments).unwrap();
    for mode in ["strict", "non_strict"] {
        let arguments = syn::parse_str(&format!("flag({mode})")).unwrap();
        super::super::validate_scalar_helper_policy(&function, &arguments).unwrap();
        let generated = expand_pcu_scalar_helper(function.clone(), &syn::parse_quote!(::pcu_alias))
            .unwrap()
            .to_string();
        assert!(generated.contains("checked_binary_value"), "{mode}");
        assert!(!generated.contains("PcuDispatchDataOp :: Alu"), "{mode}");
    }
}

#[test]
fn straight_line_helper_locals_keep_saved_values_and_legal_name_shadowing() {
    let source = syn::parse_quote! {
        fn helper(value: f32) -> f32 {
            let mut value: f32 = value;
            let saved: _ = value;
            value = value + 1.0;
            let helper = value * saved;
            helper + saved
        }
    };
    expand_pcu_scalar_helper(source, &syn::parse_quote!(::fusion_pcu)).unwrap();
}

#[test]
fn mutable_helper_parameters_keep_owned_value_and_saved_aliases() {
    let source = syn::parse_quote! {
        fn helper(mut value: f64, seed: f64) -> f64 {
            let original = value;
            value += seed;
            value *= original;
            value = value + original;
            value
        }
    };
    expand_pcu_scalar_helper(source, &syn::parse_quote!(::fusion_pcu)).unwrap();
}

#[test]
fn fixed_width_integer_helpers_use_existing_checked_ssa() {
    for ty in [
        "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128",
    ] {
        let source = format!(
            "fn helper(mut value: {ty}, seed: {ty}) -> {ty} {{\
             let original: {ty} = value; value += seed; value *= original; value - original }}"
        );
        let generated = expand_pcu_scalar_helper(
            syn::parse_str(&source).unwrap(),
            &syn::parse_quote!(::fusion_pcu),
        )
        .unwrap()
        .to_string();
        assert!(generated.contains("checked_integer_binary_value"), "{ty}");
        assert!(!generated.contains("checked_binary_value"), "{ty}");
    }
}

#[test]
fn helper_statement_faults_cannot_be_silently_erased_or_conditionally_removed() {
    for (source, diagnostic) in [
        (
            "fn helper(value: u32) -> u32 { value / value }",
            "explicit checked division API",
        ),
        (
            "fn helper(value: i32) -> i32 { -value }",
            "negation requires",
        ),
        (
            "fn helper(value: u32) -> u32 { nested(value, 1.0_f32) }",
            "matching fixed-width scalar arguments",
        ),
        (
            "fn helper(value: u128) -> u128 { value + 1 }",
            "inline constant representation",
        ),
        (
            "fn helper(value: u32) -> u32 { value as f32 }",
            "concrete f64 source",
        ),
        (
            "fn helper(value: usize) -> usize { value }",
            "fixed-width integer",
        ),
        (
            "fn helper(value: f32) -> f32 { value += 1.0; value }",
            "mutable local",
        ),
        (
            "fn helper(mut value: f32) -> f32 { let value = value; value += 1.0; value }",
            "mutable local",
        ),
        (
            "fn helper(value: f32) -> f32 { let value; value }",
            "initializer",
        ),
        (
            "fn helper(value: f32) -> f32 { let saved = value; saved = value; saved }",
            "mutable local",
        ),
        (
            "fn helper(value: f32) -> f32 { let saved = value; saved += value; saved }",
            "mutable local",
        ),
        (
            "fn helper(value: f32) -> f32 { let mut saved = value; saved %= value; saved }",
            "specified arithmetic contract",
        ),
        (
            "fn helper(value: f32) -> f32 { let value: f64 = value; value }",
            "declared scalar type",
        ),
        (
            "fn helper(value: f32) -> f32 { #[cfg(any())] let value = value; value }",
            "attributes",
        ),
        (
            "fn helper(value: f32) -> f32 { #[cfg(any())] value }",
            "attributes",
        ),
        (
            "fn helper(value: f32) -> f32 { if value > 0.0 { value } else { 0.0 } }",
            "unsupported PCU expression",
        ),
        (
            "fn helper(value: f32) -> f32 { loop {} }",
            "unsupported PCU expression",
        ),
        (
            "fn helper(value: f32) -> f32 { let (a, b) = (value, value); a + b }",
            "plain name",
        ),
        (
            "fn helper(#[cfg(any())] value: f32) -> f32 { value }",
            "plain identifiers",
        ),
    ] {
        let error = expand_pcu_scalar_helper(
            syn::parse_str(source).unwrap(),
            &syn::parse_quote!(::fusion_pcu),
        )
        .unwrap_err();
        assert!(error.to_string().contains(diagnostic), "{source}: {error}");
    }
}
