//! Generated direct entries must not inspect unread owners or suppress actual loads.
use quote::ToTokens;

fn entry(source: &str) -> String {
    let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
    let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 8").unwrap();
    let output = crate::expand_pcu_direct(arguments, &function).unwrap();
    let generated = syn::parse2::<syn::File>(output).unwrap();
    generated
        .items
        .into_iter()
        .find_map(|item| match item {
            syn::Item::Fn(function) if function.sig.ident == "kernel" => {
                Some(function.block.to_token_stream().to_string())
            }
            _ => None,
        })
        .expect("same-name direct entry")
}

#[test]
fn repeated_and_cast_loads_keep_only_actual_source_reads() {
    for source in [
        "fn kernel(input: &[f32], unused: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] * input[id]; }",
        "fn kernel<T: PcuCheckedFloat>(input: &[T], unused: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] * input[id]; }",
        "fn kernel(input: &[f64], unused: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] as f32; }",
        "fn kernel(input: &[f32], unused: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = helper(input[id]); }",
    ] {
        let body = entry(source);
        assert_eq!(body.matches("unused_read").count(), 1, "{body}");
        assert!(body.contains("let _ = unused"), "{body}");
        assert!(body.contains("as_pcu_call_argument (input"), "{body}");
        assert!(body.contains("as_pcu_call_argument (output"), "{body}");
        assert!(!body.contains("as_pcu_call_argument (unused"), "{body}");
    }
}

#[test]
fn both_live_reads_identity_and_divrem_keep_actual_borrows() {
    for source in [
        "fn kernel(input: &[f32], rhs: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] * rhs[id]; }",
        "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[id]; }",
        "fn kernel(input: &[i8], rhs: &[i8], quotient: &mut [i8], remainder: &mut [i8]) { let id = pcu::context::global_invocation_id(); let (q, r) = pcu::checked_div_rem(input[id], rhs[id]); quotient[id] = q; remainder[id] = r; }",
    ] {
        let body = entry(source);
        assert!(!body.contains("unused_read"), "{body}");
        assert!(body.contains("as_pcu_call_argument (input"), "{body}");
    }
}

#[test]
fn scalar_broadcast_and_grid_reads_keep_the_same_compile_time_rule() {
    for source in [
        "fn kernel(input: &f32, unused: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = *input * *input; }",
        "fn kernel(input: &[f32], unused: &[f32], output: &mut [f32]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { output[id] = input[id] * input[id]; id += stride; } }",
    ] {
        let body = entry(source);
        assert_eq!(body.matches("unused_read").count(), 1, "{body}");
        assert!(body.contains("as_pcu_call_argument (input"), "{body}");
    }
}

#[test]
fn readonly_element_zero_is_independent_of_invocation_indexing() {
    for source in [
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] / input[0]; }",
        "fn kernel<T: PcuCheckedFloat>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[0usize] / input[id]; }",
        "fn kernel(input: &[f64], output: &mut [f64]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { output[id] = input[(0)] / input[id]; id += stride; } }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = helper(input[id], input[0]); }",
    ] {
        let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
        let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 8").unwrap();
        let generated = crate::expand_pcu_direct(arguments, &function)
            .unwrap()
            .to_string();
        assert!(generated.contains("BindingElementZero"), "{generated}");
        assert!(!entry(source).contains("unused_read"));
    }
}

#[test]
fn element_zero_does_not_widen_writes_or_arbitrary_indexing() {
    for source in [
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[0] = input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[1] + input[id]; }",
        "fn kernel(input: &[f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[0u32] + input[id]; }",
    ] {
        let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
        let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 8").unwrap();
        assert!(crate::expand_pcu_direct(arguments, &function).is_err());
    }
}

#[test]
fn generic_carrier_element_zero_selects_the_broadcast_builder() {
    for (source, expected) in [
        (
            "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[0]; }",
            "build_broadcast",
        ),
        (
            "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { output[id] = input[(0usize)]; id += stride; } }",
            "build_grid_stride_broadcast",
        ),
        (
            "fn kernel<T: PcuScalar>(input: &T, output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = *input; }",
            "build_broadcast",
        ),
    ] {
        let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
        let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 8").unwrap();
        let generated = crate::expand_pcu_direct(arguments, &function)
            .unwrap()
            .to_string();
        assert!(generated.contains(expected), "{generated}");
        assert!(entry(source).contains("as_pcu_call_argument (input"));
    }
    for source in [
        "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[1]; }",
        "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[0] = input[0]; }",
        "fn kernel<T: PcuScalar>(input: &[T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); output[id] = input[0u32]; }",
    ] {
        let function = syn::parse_str::<syn::ItemFn>(source).unwrap();
        let arguments = syn::parse_str::<crate::PcuDispatchArgs>("invocations = 8").unwrap();
        assert!(crate::expand_pcu_direct(arguments, &function).is_err());
    }
}

#[test]
fn unused_mutable_keeps_access_without_inspecting_the_owner() {
    for source in [
        "fn kernel(input: &[f32], ghost: &mut [f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] * input[id]; }",
        "fn kernel<T: PcuScalar>(input: &[T], ghost: &mut [T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); let saved = input[id]; output[id] = saved; }",
        "fn kernel<T: PcuScalar>(input: &[T], ghost: &mut [T], output: &mut [T]) { let mut id = pcu::context::global_invocation_id(); let stride = pcu::context::invocation_count(); while id < 17 { output[id] = input[id]; id += stride; } }",
    ] {
        let body = entry(source);
        assert_eq!(body.matches("unused_read_write").count(), 1, "{body}");
        assert!(body.contains("let _ = ghost"), "{body}");
        assert!(!body.contains("as_pcu_call_argument (ghost"), "{body}");
        assert!(body.contains("as_pcu_call_argument (output"), "{body}");
    }
}

#[test]
fn mutable_reads_and_overwritten_stores_are_still_actual_uses() {
    for source in [
        "fn kernel(input: &[f32], stage: &mut [f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); output[id] = input[id] + stage[id]; }",
        "fn kernel<T: PcuScalar>(input: &[T], stage: &mut [T], output: &mut [T]) { let id = pcu::context::global_invocation_id(); stage[id] = input[id]; stage[id] = input[id]; output[id] = input[id]; }",
        "fn kernel(input: &[f32], stage: &mut [f32], output: &mut [f32]) { let id = pcu::context::global_invocation_id(); stage[id] = input[id] / input[id]; stage[id] = input[id]; output[id] = input[id]; }",
    ] {
        let body = entry(source);
        assert!(!body.contains("unused_read_write"), "{body}");
        assert!(body.contains("as_pcu_call_argument (stage"), "{body}");
        assert!(body.contains("as_pcu_call_argument (output"), "{body}");
    }
}
