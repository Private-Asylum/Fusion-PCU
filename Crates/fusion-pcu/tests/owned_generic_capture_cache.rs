#![cfg(all(feature = "rocm", feature = "tensor"))]

//! Confirms generic scalar capture uses one scalar-neutral cache without mixing scalar types.

use core::any::TypeId;
#[rustfmt::skip]
use core::{
    marker::PhantomData,
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};
use fusion_pcu::global;
use fusion_pcu::PcuScalar;

static SITE: global::PcuHostCallSite = global::PcuHostCallSite::new();
static FACTORY_CALLS: AtomicUsize = AtomicUsize::new(0);
static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Specialization<T>(PhantomData<fn() -> T>);

fn capture_identity<T: PcuScalar>(
    capture: &mut global::PcuTensorGraphCapture,
    [input]: [global::PcuTensorGraphValue<T>; 1],
) -> Result<global::PcuTensorGraphValue<T>, global::PcuExecutionError> {
    FACTORY_CALLS.fetch_add(1, Ordering::Relaxed);
    capture.identity(input)
}

fn execute<T: PcuScalar>(values: &[T]) -> Result<global::PcuTensor<T>, global::PcuExecutionError> {
    let source =
        global::PcuTensorSource::as_tensor_source(values).map_err(global::argument_error)?;
    global::call_owned_tensor_capture::<T, 1, _>(
        &SITE,
        TypeId::of::<Specialization<T>>(),
        [source],
        capture_identity::<T>,
    )
}

#[test]
#[ignore = "requires a compatible ROCm device"]
fn generic_scalar_cache_keeps_f32_f64_entries_and_escaped_owners_independent() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    FACTORY_CALLS.store(0, Ordering::Relaxed);

    let f32_first = execute(&[1.25_f32, -2.5]).unwrap();
    let f64_first = execute(&[16_777_217.0_f64, 1.0e-10]).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 2);
    let f32_warm = execute(&[3.5_f32, 4.25]).unwrap();
    let f64_warm = execute(&[16_777_219.0_f64, -1.0e-10]).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 2);

    global::clear_thread_cache().unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 2);
    let f32_after_clear = execute(&[-7.0_f32, 8.5]).unwrap();
    let f64_after_clear = execute(&[-16_777_219.0_f64, 3.0e-12]).unwrap();
    assert_eq!(FACTORY_CALLS.load(Ordering::Relaxed), 4);

    let mut actual_f32 = [0.0_f32; 2];
    f32_first.read_into(&mut actual_f32).unwrap();
    assert_eq!(
        actual_f32.map(f32::to_bits),
        [1.25_f32.to_bits(), (-2.5_f32).to_bits()]
    );
    f32_warm.read_into(&mut actual_f32).unwrap();
    assert_eq!(
        actual_f32.map(f32::to_bits),
        [3.5_f32.to_bits(), 4.25_f32.to_bits()]
    );
    f32_after_clear.read_into(&mut actual_f32).unwrap();
    assert_eq!(
        actual_f32.map(f32::to_bits),
        [(-7.0_f32).to_bits(), 8.5_f32.to_bits()]
    );

    let mut actual_f64 = [0.0_f64; 2];
    f64_first.read_into(&mut actual_f64).unwrap();
    assert_eq!(
        actual_f64.map(f64::to_bits),
        [16_777_217.0_f64.to_bits(), 1.0e-10_f64.to_bits()]
    );
    f64_warm.read_into(&mut actual_f64).unwrap();
    assert_eq!(
        actual_f64.map(f64::to_bits),
        [16_777_219.0_f64.to_bits(), (-1.0e-10_f64).to_bits()]
    );
    f64_after_clear.read_into(&mut actual_f64).unwrap();
    assert_eq!(
        actual_f64.map(f64::to_bits),
        [(-16_777_219.0_f64).to_bits(), 3.0e-12_f64.to_bits()]
    );
}

static SELECTED_SITE: global::PcuHostCallSite = global::PcuHostCallSite::new();
static SELECTED_FACTORY_CALLS: AtomicUsize = AtomicUsize::new(0);
struct SelectedSpecialization;

fn capture_first(
    capture: &mut global::PcuTensorGraphCapture,
    [selected, _unused]: [global::PcuTensorGraphValue<f32>; 2],
) -> Result<global::PcuTensorGraphValue<f32>, global::PcuExecutionError> {
    SELECTED_FACTORY_CALLS.fetch_add(1, Ordering::Relaxed);
    capture.identity(selected)
}

fn execute_first(
    selected: &global::PcuTensor<f32>,
    unused: &global::PcuTensor<f32>,
) -> Result<global::PcuTensor<f32>, global::PcuExecutionError> {
    let sources = [selected, unused]
        .map(global::PcuTensorSource::<f32, global::SliceShape>::as_tensor_source);
    let [selected, unused] = sources;
    global::call_owned_tensor_capture::<f32, 2, _>(
        &SELECTED_SITE,
        TypeId::of::<SelectedSpecialization>(),
        [
            selected.map_err(global::argument_error)?,
            unused.map_err(global::argument_error)?,
        ],
        capture_first,
    )
}

#[test]
#[ignore = "requires a compatible ROCm device"]
fn selected_affinity_cache_ignores_unused_roots_without_warm_recapture() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let old = execute(&[101.0_f32, 202.0]).unwrap();
    global::clear_thread_cache().unwrap();
    let selected = execute(&[3.0_f32, 4.0]).unwrap();
    let changed = execute(&[5.0_f32, 6.0]).unwrap();
    let same_root_unused = execute(&[303.0_f32, 404.0]).unwrap();
    SELECTED_FACTORY_CALLS.store(0, Ordering::Relaxed);

    let first = execute_first(&selected, &old).unwrap();
    assert_eq!(SELECTED_FACTORY_CALLS.load(Ordering::Relaxed), 1);
    let warm_changed = execute_first(&changed, &same_root_unused).unwrap();
    assert_eq!(SELECTED_FACTORY_CALLS.load(Ordering::Relaxed), 1);
    let old_selected = execute_first(&old, &selected).unwrap();
    assert_eq!(SELECTED_FACTORY_CALLS.load(Ordering::Relaxed), 2);
    let revisited = execute_first(&changed, &old).unwrap();
    assert_eq!(SELECTED_FACTORY_CALLS.load(Ordering::Relaxed), 2);

    global::clear_thread_cache().unwrap();
    for (result, expected) in [
        (first, [3.0_f32, 4.0]),
        (warm_changed, [5.0_f32, 6.0]),
        (old_selected, [101.0_f32, 202.0]),
        (revisited, [5.0_f32, 6.0]),
    ] {
        let mut observed = [f32::NAN; 2];
        result.read_into(&mut observed).unwrap();
        assert_eq!(observed.map(f32::to_bits), expected.map(f32::to_bits));
    }
}
