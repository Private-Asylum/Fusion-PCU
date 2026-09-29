//! Per-function owned sources that consume an existing resident tensor.
#![cfg(feature = "tensor")]

use fusion_pcu::pcu;
#[cfg(feature = "rocm")]
use fusion_pcu::global;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn consume_relu<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::relu(&input)?)
}

#[pcu]
fn consume_identity<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn borrowed_identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn matrix_identity<const R: usize, const C: usize>(
    input: &[[f32; C]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[test]
#[cfg(not(feature = "rocm"))]
fn consumed_owner_profile_compiles_without_a_backend_fallback() {
    let _: fn(PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> = consume_relu::<f32>;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> =
        consume_identity::<f64>;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn consumed_owner_execution_preserves_f32_f64_values_and_retained_results() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let f32_source = [-2.0_f32, 3.0, -0.0, 8.5];
    let first_input = borrowed_identity(&f32_source).unwrap();
    let first_output = consume_relu(first_input).unwrap();

    let changed_source = [-4.0_f32, 1.5, 2.0, -9.0];
    let changed_input = borrowed_identity(&changed_source).unwrap();
    let changed_output = consume_relu(changed_input).unwrap();

    global::clear_thread_cache().unwrap();
    let after_clear_source = [7.0_f32, -6.0, 0.25, -0.0];
    let after_clear_input = borrowed_identity(&after_clear_source).unwrap();
    let after_clear_output = consume_relu(after_clear_input).unwrap();

    let mut observed_f32 = [0.0_f32; 4];
    first_output.read_into(&mut observed_f32).unwrap();
    assert_eq!(
        observed_f32.map(f32::to_bits),
        [0.0, 3.0, 0.0, 8.5].map(f32::to_bits)
    );
    changed_output.read_into(&mut observed_f32).unwrap();
    assert_eq!(
        observed_f32.map(f32::to_bits),
        [0.0, 1.5, 2.0, 0.0].map(f32::to_bits)
    );
    after_clear_output.read_into(&mut observed_f32).unwrap();
    assert_eq!(
        observed_f32.map(f32::to_bits),
        [7.0, 0.0, 0.25, 0.0].map(f32::to_bits)
    );

    let f64_source = [-16_777_217.0_f64, 16_777_219.0, -0.0, 0.125];
    let f64_input = borrowed_identity(&f64_source).unwrap();
    let f64_output = consume_identity(f64_input).unwrap();
    let mut observed_f64 = [0.0_f64; 4];
    f64_output.read_into(&mut observed_f64).unwrap();
    assert_eq!(observed_f64.map(f64::to_bits), f64_source.map(f64::to_bits));

    let f64_relu_input = borrowed_identity(&f64_source).unwrap();
    let f64_relu_output = consume_relu(f64_relu_input).unwrap();
    f64_relu_output.read_into(&mut observed_f64).unwrap();
    assert_eq!(
        observed_f64.map(f64::to_bits),
        [0.0, 16_777_219.0, 0.0, 0.125].map(f64::to_bits)
    );
    // The identity result is a distinct logical value and remains unchanged.
    f64_output.read_into(&mut observed_f64).unwrap();
    assert_eq!(observed_f64.map(f64::to_bits), f64_source.map(f64::to_bits));

    let matrix = [[1.0_f32, 2.0], [3.0, 4.0]];
    let rank_two = matrix_identity(&matrix).unwrap();
    assert!(matches!(
        consume_relu(rank_two),
        Err(PcuExecutionError::Argument(
            fusion_pcu::global::PcuArgumentError::ResidentShapeMismatch { .. }
        ))
    ));
}
