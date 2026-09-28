//! Hardware acceptance for cold owned-source graph capture and warm cache reuse.
#![cfg(all(feature = "rocm", feature = "tensor"))]

#[rustfmt::skip]
use core::{
    any::TypeId,
    sync::atomic::{AtomicUsize, Ordering},
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuExecutionError,
    PcuTensor,
};

static SITE: global::PcuHostCallSite = global::PcuHostCallSite::new();
static FACTORY_CALLS: AtomicUsize = AtomicUsize::new(0);
static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct CaptureSpecialization;
struct CaptureMarker;

fn capture_relu(
    capture: &mut global::PcuTensorGraphCapture,
    [input]: [global::PcuTensorGraphValue; 1],
) -> Result<global::PcuTensorGraphValue, PcuExecutionError> {
    FACTORY_CALLS.fetch_add(1, Ordering::Relaxed);
    capture.enter(TypeId::of::<CaptureMarker>())?;
    let result = capture.relu(input);
    capture.leave();
    result
}

fn execute(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let source =
        global::PcuTensorSource::as_tensor_source(input).map_err(global::argument_error)?;
    global::call_owned_tensor_capture(
        &SITE,
        TypeId::of::<CaptureSpecialization>(),
        [source],
        capture_relu,
    )
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
#[ignore = "requires a working ROCm device"]
fn stable_factory_runs_only_on_cold_shape_and_cache_clear() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    FACTORY_CALLS.store(0, Ordering::Relaxed);

    let first_source = [-2.0_f32, 3.0, -4.0];
    let first = execute(&first_source).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 1);

    let changed_source = [5.0_f32, -6.0, 7.0];
    let changed = execute(&changed_source).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 1);

    let new_shape_source = [8.0_f32, -9.0, 10.0, -11.0];
    let new_shape = execute(&new_shape_source).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 2);

    global::clear_thread_cache().unwrap();
    let after_clear_source = [-12.0_f32, 13.0, -14.0];
    let after_clear = execute(&after_clear_source).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 3);

    let mut first_values = [f32::NAN; 3];
    let mut changed_values = [f32::NAN; 3];
    let mut new_shape_values = [f32::NAN; 4];
    let mut after_clear_values = [f32::NAN; 3];
    first.read_into(&mut first_values).unwrap();
    changed.read_into(&mut changed_values).unwrap();
    new_shape.read_into(&mut new_shape_values).unwrap();
    after_clear.read_into(&mut after_clear_values).unwrap();
    assert_bits(&first_values, &[0.0, 3.0, 0.0]);
    assert_bits(&changed_values, &[5.0, 0.0, 7.0]);
    assert_bits(&new_shape_values, &[8.0, 0.0, 10.0, 0.0]);
    assert_bits(&after_clear_values, &[0.0, 13.0, 0.0]);
    global::clear_thread_cache().unwrap();
}
