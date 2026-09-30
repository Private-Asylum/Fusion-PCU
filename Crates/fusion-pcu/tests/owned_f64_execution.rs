//! End-to-end double-precision ownership and per-function source conformance.
#![cfg(feature = "tensor")]

use fusion_pcu::pcu;
#[cfg(feature = "rocm")]
use fusion_pcu::global;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuTensor,
};

#[cfg(feature = "rocm")]
static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[pcu]
fn retain(input: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn difference(left: &[f64], right: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(left - right)
}

#[pcu]
fn product(left: &[f64], right: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(left * right)
}

#[pcu]
fn activate(input: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu]
fn sum_relu(left: &[f64], right: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(activate(&(left + right))?)
}

#[pcu(flag(strict))]
fn matrix<const R: usize, const K: usize, const C: usize>(
    left: &[[f64; K]; R],
    right: &[[f64; C]; K],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(pcu::matmul(left, right)?)
}

#[cfg(feature = "rocm")]
fn assert_bits(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires a working ROCm device with FP64 and rocBLAS"]
fn f64_source_owners_preserve_precision_current_inputs_and_root_lifetime() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let bits = [
        16_777_217.0_f64,
        1.0e-10,
        -0.0,
        f64::from_bits(0x7ff8_0000_0000_1234),
    ];
    let owner = retain(&bits).unwrap();
    let resident_copy = retain(&owner).unwrap();
    let changed = [8.0, -2.0, 0.25, 16_777_219.0];
    let warm = retain(&changed).unwrap();
    let left = [16_777_217.0, 1.0e-10, -2.0, 0.25];
    let right = [1.0, 2.0e-10, 3.0, -0.5];
    let left_owner = retain(&left).unwrap();
    let right_owner = retain(&right).unwrap();
    let sub = difference(&left_owner, &right).unwrap();
    let mul = product(&left_owner, &right_owner).unwrap();
    let activated = sum_relu(&left, &right_owner).unwrap();
    let mut actual = [0.0; 4];
    resident_copy.read_into(&mut actual).unwrap();
    assert_bits(&actual, &bits);
    warm.read_into(&mut actual).unwrap();
    assert_bits(&actual, &changed);
    sub.read_into(&mut actual).unwrap();
    assert_bits(
        &actual,
        &std::array::from_fn::<_, 4, _>(|i| left[i] - right[i]),
    );
    mul.read_into(&mut actual).unwrap();
    assert_bits(
        &actual,
        &std::array::from_fn::<_, 4, _>(|i| left[i] * right[i]),
    );
    activated.read_into(&mut actual).unwrap();
    assert_bits(
        &actual,
        &std::array::from_fn::<_, 4, _>(|i| (left[i] + right[i]).max(0.0)),
    );
    // Dropping the execution cache must not invalidate an escaped device owner.
    global::clear_thread_cache().unwrap();
    owner.read_into(&mut actual).unwrap();
    assert_bits(&actual, &bits);
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires a working ROCm device with FP64 and HIPRTC"]
fn f64_source_matmul_uses_double_precision_and_const_shapes() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let left = [[16_777_217.0_f64, 1.0e-10], [-3.0, 0.5]];
    let right = [[1.0_f64, 0.0], [0.0, 1.0]];
    let output = matrix::<2, 2, 2>(&left, &right).unwrap();
    assert_eq!(output.shape(), [2, 2]);
    let mut actual = [0.0; 4];
    output.read_into(&mut actual).unwrap();
    assert_bits(&actual, &[left[0][0], left[0][1], left[1][0], left[1][1]]);
    let changed = [[2.0_f64, -1.0], [4.0, 3.0]];
    let weights = [[3.0_f64, 2.0], [1.0, -2.0]];
    let warm = matrix::<2, 2, 2>(&changed, &weights).unwrap();
    warm.read_into(&mut actual).unwrap();
    assert_bits(&actual, &[5.0, 6.0, 15.0, 2.0]);
    global::clear_thread_cache().unwrap();
}

#[test]
#[cfg(not(all(any(feature = "rocm", feature = "cuda"), feature = "tensor")))]
fn f64_source_never_substitutes_host_execution_for_a_disabled_backend() {
    assert!(matches!(
        retain(&[16_777_217.0_f64]),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires a working ROCm device with FP64"]
fn f64_factory_is_cold_only_across_current_inputs_and_cache_clear() {
    #[rustfmt::skip]
    use core::{
        any::TypeId,
        sync::atomic::{
            AtomicUsize,
            Ordering,
        },
    };
    static SITE: global::PcuHostCallSite = global::PcuHostCallSite::new();
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    struct Specialization;
    fn capture(
        capture: &mut global::PcuTensorGraphCapture,
        [input]: [global::PcuTensorGraphValue<f64>; 1],
    ) -> Result<global::PcuTensorGraphValue<f64>, PcuExecutionError> {
        CALLS.fetch_add(1, Ordering::Relaxed);
        capture.identity(input)
    }
    fn execute(values: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
        let source =
            global::PcuTensorSource::as_tensor_source(values).map_err(global::argument_error)?;
        global::call_owned_tensor_capture(&SITE, TypeId::of::<Specialization>(), [source], capture)
    }
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    CALLS.store(0, Ordering::Relaxed);
    let first = execute(&[16_777_217.0, 1.0e-10]).unwrap();
    let second = execute(&[16_777_219.0, -1.0e-10]).unwrap();
    assert_eq!(CALLS.load(Ordering::Relaxed), 1);
    let shaped = execute(&[0.0, 1.0, 2.0]).unwrap();
    assert_eq!(CALLS.load(Ordering::Relaxed), 2);
    global::clear_thread_cache().unwrap();
    let cleared = execute(&[3.0, 4.0]).unwrap();
    assert_eq!(CALLS.load(Ordering::Relaxed), 3);
    let mut actual = [0.0; 2];
    first.read_into(&mut actual).unwrap();
    assert_bits(&actual, &[16_777_217.0, 1.0e-10]);
    second.read_into(&mut actual).unwrap();
    assert_bits(&actual, &[16_777_219.0, -1.0e-10]);
    cleared.read_into(&mut actual).unwrap();
    assert_bits(&actual, &[3.0, 4.0]);
    let mut shaped_values = [0.0; 3];
    shaped.read_into(&mut shaped_values).unwrap();
    assert_bits(&shaped_values, &[0.0, 1.0, 2.0]);
    global::clear_thread_cache().unwrap();
}
