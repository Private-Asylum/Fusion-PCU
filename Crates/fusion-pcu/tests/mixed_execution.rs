//! Same-name source calls across RAM and owned device storage.
#![cfg(all(feature = "rocm", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuTensor,
};
#[pcu]
fn identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(invocations: N)]
fn accumulate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = output[id] + input[id];
}
fn assert_values(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}
#[test]
#[ignore = "requires a working ROCm device"]
fn same_function_accepts_host_resident_and_mixed_arguments() {
    global::use_defaults().unwrap();
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let resident_input = identity(&input).unwrap();
    let mut resident_output = identity(&[10.0_f32; 4]).unwrap();
    accumulate::<4>(&resident_input, &mut resident_output).unwrap();
    accumulate::<4>(&input, &mut resident_output).unwrap();
    // Reusing an existing mutable reference must reborrow it, as an ordinary Rust fn does.
    let borrowed_output = &mut resident_output;
    accumulate::<4>(&resident_input, borrowed_output).unwrap();
    accumulate::<4>(&input, borrowed_output).unwrap();
    let mut actual = [0.0_f32; 4];
    resident_output.read_into(&mut actual).unwrap();
    assert_values(&actual, &[14.0, 18.0, 22.0, 26.0]);
    let mut host_output = [20.0_f32; 4];
    accumulate::<4>(&resident_input, &mut host_output).unwrap();
    assert_values(&host_output, &[21.0, 22.0, 23.0, 24.0]);
    global::clear_thread_cache().unwrap();
    accumulate::<4>(&resident_input, &mut resident_output).unwrap();
    resident_output.read_into(&mut actual).unwrap();
    assert_values(&actual, &[15.0, 20.0, 25.0, 30.0]);
    global::clear_thread_cache().unwrap();
}
#[test]
#[ignore = "requires a working ROCm device"]
fn incompatible_resident_roots_are_rejected_before_mutation() {
    global::use_defaults().unwrap();
    let input = identity(&[1.0_f32, 2.0]).unwrap();
    global::clear_thread_cache().unwrap();
    let mut output = identity(&[10.0_f32, 20.0]).unwrap();
    let error = accumulate::<2>(&input, &mut output).unwrap_err();
    assert!(matches!(
        error,
        PcuExecutionError::Argument(global::PcuArgumentError::SessionMismatch)
    ));
    let mut actual = [0.0_f32; 2];
    output.read_into(&mut actual).unwrap();
    assert_values(&actual, &[10.0, 20.0]);
    global::clear_thread_cache().unwrap();
}
