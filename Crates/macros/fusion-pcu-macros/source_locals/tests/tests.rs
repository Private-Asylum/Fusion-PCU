//! Expansion facts do not confer execution or arbitrary Rust source support.
use quote::ToTokens;

fn expand(source: &str) -> Result<String, syn::Error> {
    let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
    let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 3").unwrap();
    crate::expand_pcu_direct(arguments, &function)
        .map(|tokens| tokens.to_token_stream().to_string())
}

#[test]
fn local_ssa_is_emitted_once_and_reused_with_ordinary_shadowing() {
    for source in [
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let value = input[id]; let doubled = value + value; output[id] = doubled * value; }",
        "fn kernel<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id]; let value = value + value; output[id] = value * value; }",
    ] {
        let tokens = expand(source).unwrap();
        assert_eq!(tokens.matches("load_value").count(), 1, "{tokens}");
        assert_eq!(
            tokens.matches("checked_binary_value").count(),
            2,
            "{tokens}"
        );
        assert!(!tokens.contains("unused_read"), "{tokens}");
    }
    let tokens = expand("fn kernel<T: PcuCheckedInteger>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id]; let doubled = value + value; output[id] = doubled * value; }").unwrap();
    assert_eq!(tokens.matches("load_value").count(), 1);
    assert_eq!(tokens.matches("checked_integer_binary_value").count(), 2);
}

#[test]
fn grid_and_matrix_coordinates_remain_distinct_from_scalar_values() {
    for source in [
        "fn kernel<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { let value = input[id]; let doubled = value + value; output[id] = doubled * value; id += stride; } }",
        "fn kernel(input: &[[f32; 3]; 1], output: &mut [[f32; 3]; 1]) { let id = pcu::context::global_invocation_id(); let row = id / 3; let col = id % 3; let value = input[row][col]; let doubled = value + value; output[row][col] = doubled * value; }",
        "fn kernel(input: &[[f32; 3]; 1], output: &mut [[f32; 3]; 1]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 1 * 3 { let row = id / 3; let col = id % 3; let value = input[row][col]; let doubled = value + value; output[row][col] = doubled * value; id += stride; } }",
    ] {
        let tokens = expand(source).unwrap();
        assert_eq!(tokens.matches("load_value").count(), 1, "{tokens}");
        assert_eq!(
            tokens.matches("checked_binary_value").count(),
            2,
            "{tokens}"
        );
    }
}

#[test]
fn unused_checked_initializer_keeps_fault_effect_and_actual_input_borrow() {
    let tokens = expand("fn kernel(input: &[f32], denominator: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let unused = input[id] / denominator[id]; output[id] = input[id]; }").unwrap();
    assert_eq!(tokens.matches("checked_binary_value").count(), 1);
    assert!(tokens.contains("PcuDispatchFloatBinaryOp :: Div"));
    assert!(!tokens.contains("unused_read"));
}

#[test]
fn unsupported_effects_and_reserved_name_shadowing_are_rejected() {
    for source in [
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] += input[id]; }",
        "fn kernel<T: PcuCheckedInteger>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let mut value = input[id]; value /= input[id]; output[id] = value; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let mut value = input[id]; #[cfg(any())] value += value; output[id] = value; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let value = input[id]; value = value + value; output[id] = value; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let mut value = input[id]; let value = value; value = value + value; output[id] = value; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let mut value = input[id]; value = value as f64; output[id] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); missing = input[id]; output[id] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let mut value; value = input[id]; output[id] = value; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let value: f64 = input[id]; output[id] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let input = input[id]; output[id] = input; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let id = input[id]; output[id] = id; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { let stride = input[id]; output[id] = stride; id += stride; } }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); input[id] = output[id]; output[id] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[0] = input[id]; output[id] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); #[cfg(any())] output[id] = input[id]; output[id] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let value = if true { input[id] } else { input[0] }; output[id] = value; }",
        "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id] + input[id]; output[id] = value; }",
        "fn kernel<T: PcuCheckedInteger>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id] / input[id]; output[id] = value; }",
    ] {
        assert!(expand(source).is_err(), "{source}");
    }
}

#[test]
fn mutable_locals_rebind_ssa_without_reloading_or_losing_prior_aliases() {
    for source in [
        "fn kernel<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let mut value = input[id]; let original = value; value = value + value; output[id] = value * original; }",
        "fn kernel<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { let mut value = input[id]; let original = value; value = value + value; output[id] = value * original; id += stride; } }",
        "fn kernel(input: &[[f32; 3]; 1], output: &mut [[f32; 3]; 1]) { let id = pcu::context::global_invocation_id(); let row = id / 3; let col = id % 3; let mut value = input[row][col]; let original = value; value = value + value; output[row][col] = value * original; }",
    ] {
        let tokens = expand(source).unwrap();
        assert_eq!(tokens.matches("load_value").count(), 1, "{tokens}");
        assert_eq!(
            tokens.matches("checked_binary_value").count(),
            2,
            "{tokens}"
        );
    }
    let tokens = expand("fn kernel<T: PcuCheckedInteger>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id]; let mut value = value; value = value + value; output[id] = value * value; }").unwrap();
    assert_eq!(tokens.matches("load_value").count(), 1);
    assert_eq!(tokens.matches("checked_integer_binary_value").count(), 2);
}

#[test]
fn ordered_stores_retain_distinct_resource_loads_and_scalar_ssa() {
    for source in [
        "fn kernel<T: PcuCheckedFloat>(input: &[T], stage: &mut [T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id]; stage[id] = value + value; let updated = stage[id]; output[id] = updated * value; }",
        "fn kernel<T: PcuCheckedFloat>(input: &[T], stage: &mut [T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { let value = input[id]; stage[id] = value + value; let updated = stage[id]; output[id] = updated * value; id += stride; } }",
        "fn kernel<T: PcuCheckedInteger>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value = input[id]; output[id] = value + value; let updated = output[id]; output[id] = updated * value; }",
        "fn kernel(input: &[[f32; 3]; 1], output: &mut [[f32; 3]; 1]) { let id = pcu::context::global_invocation_id(); let row = id / 3; let col = id % 3; let value = input[row][col]; output[row][col] = value + value; let updated = output[row][col]; output[row][col] = updated * value; }",
    ] {
        let tokens = expand(source).unwrap();
        assert_eq!(tokens.matches("store_value").count(), 2, "{tokens}");
        assert_eq!(tokens.matches("load_value").count(), 2, "{tokens}");
        let first_store = tokens.find("store_value").unwrap();
        let last_load = tokens.rfind("load_value").unwrap();
        let last_store = tokens.rfind("store_value").unwrap();
        assert!(first_store < last_load && last_load < last_store);
    }
}

#[test]
fn explicit_scalar_local_types_preserve_ssa_and_check_initializer_types() {
    for source in [
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let value: f32 = input[id]; let mut next: f32 = value; next = next + value; output[id] = next; }",
        "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value: T = input[id]; let mut saved: T = value; saved = value; output[id] = saved; }",
        "fn kernel<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let value: T = input[id]; let result: _ = value * value; output[id] = result; }",
        "fn kernel(input: &[i32], output: &mut [i32]) { let id = pcu::context::global_invocation_id(); let value: i32 = input[id]; output[id] = value + value; }",
    ] {
        let tokens = expand(source).unwrap();
        assert_eq!(tokens.matches("load_value").count(), 1, "{tokens}");
    }
    for ty in ["f64", "u32", "Other", "&f32", "(f32, f32)", "T"] {
        let source = format!(
            "fn kernel(input: &[f32], output: &mut [f32]) {{ let id = pcu::context::global_invocation_id(); let value: {ty} = input[id]; output[id] = input[id]; }}"
        );
        assert!(expand(&source).is_err(), "{source}");
    }
}

#[test]
fn explicit_float_annotation_guides_literals_without_implicit_conversion() {
    let tokens = expand("fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let constant: f64 = 1.0; let converted: f32 = constant as f32; output[id] = input[id] + converted; }").unwrap();
    assert_eq!(tokens.matches("constant_f64_value").count(), 1, "{tokens}");
    assert_eq!(
        tokens.matches("checked_f64_to_f32_value").count(),
        1,
        "{tokens}"
    );
    assert!(expand("fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); let constant: f64 = 1.0_f32; output[id] = input[id]; }").is_err());
}
