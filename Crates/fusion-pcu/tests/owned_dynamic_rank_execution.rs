//! Dynamic-rank resident owners retain their runtime shape through borrowed and consumed calls.
#![cfg(feature = "tensor")]

#[cfg(feature = "rocm")]
#[rustfmt::skip]
use core::{
    any::TypeId,
    marker::PhantomData,
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuArgumentError,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn seed_matrix<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn seed_vector<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn borrowed_owner<T: PcuScalar>(input: &PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn consume_relu<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::relu(input)?)
}

#[pcu]
fn borrow_then_consume<T: PcuScalar>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let copied = borrowed_owner(&input)?;
    let activated = consume_relu(input)?;
    Ok(pcu::add(copied, activated)?)
}

#[pcu]
fn seed_then_borrow_and_consume<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let owner = seed_matrix::<T, R, C>(input)?;
    Ok(borrow_then_consume(owner)?)
}

#[cfg(feature = "rocm")]
struct CaptureSpecialization<T>(PhantomData<fn() -> T>);

#[cfg(feature = "rocm")]
static CACHE_FACTORY_CALLS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "rocm")]
static POLICY_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(feature = "rocm")]
static CACHE_SITE: global::PcuHostCallSite = global::PcuHostCallSite::new();

#[cfg(feature = "rocm")]
fn capture_identity<T: PcuScalar>(
    capture: &mut global::PcuTensorGraphCapture,
    [input]: [global::PcuTensorGraphValue<T>; 1],
) -> Result<global::PcuTensorGraphValue<T>, PcuExecutionError> {
    CACHE_FACTORY_CALLS.fetch_add(1, Ordering::Relaxed);
    capture.identity(input)
}

#[cfg(feature = "rocm")]
fn capture_dynamic_identity<T: PcuScalar>(
    input: &PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let source =
        global::PcuTensorSource::<T, global::DynamicResidentShape>::as_tensor_source(input)
            .map_err(global::argument_error)?;
    global::call_owned_tensor_capture::<T, 1, _>(
        &CACHE_SITE,
        TypeId::of::<CaptureSpecialization<T>>(),
        [source],
        capture_identity::<T>,
    )
}

#[test]
#[allow(clippy::type_complexity)] // Exact generated const-generic function signatures are tested.
fn dynamic_resident_profiles_compile_without_a_provider() {
    let _: for<'owner> fn(&'owner PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> =
        borrowed_owner;
    let _: fn(PcuTensor<f64>) -> Result<PcuTensor<f64>, PcuExecutionError> = borrow_then_consume;
    let _: for<'owner> fn(
        &'owner PcuTensor<f32>,
    ) -> Result<global::PcuTensorInput<'owner, f32>, PcuArgumentError> =
        global::PcuTensorSource::<f32, global::DynamicResidentShape>::as_tensor_source;
    let _: fn(&[[f64; 3]; 2]) -> Result<PcuTensor<f64>, PcuExecutionError> =
        seed_then_borrow_and_consume::<f64, 2, 3>;
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware"]
fn dynamic_resident_rank_and_extent_keys_preserve_values_across_cache_roots() {
    let _guard = POLICY_TEST_LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    CACHE_FACTORY_CALLS.store(0, Ordering::Relaxed);

    let matrix_23_values = [[1.0_f32, -2.0, 3.0], [-4.0, 5.0, -6.0]];
    let matrix_23 = seed_matrix(&matrix_23_values).unwrap();
    let matrix_23_result = capture_dynamic_identity(&matrix_23).unwrap();
    assert_eq!(CACHE_FACTORY_CALLS.load(Ordering::Relaxed), 1);

    let changed_23_values = [[7.0_f32, 8.0, 9.0], [10.0, 11.0, 12.0]];
    let changed_23 = seed_matrix(&changed_23_values).unwrap();
    let changed_23_result = capture_dynamic_identity(&changed_23).unwrap();
    assert_eq!(CACHE_FACTORY_CALLS.load(Ordering::Relaxed), 1);

    let vector_6_values = [13.0_f32, 14.0, 15.0, 16.0, 17.0, 18.0];
    let vector_6 = seed_vector(&vector_6_values).unwrap();
    let vector_6_result = capture_dynamic_identity(&vector_6).unwrap();
    assert_eq!(CACHE_FACTORY_CALLS.load(Ordering::Relaxed), 2);

    let matrix_32_values = [[19.0_f32, 20.0], [21.0, 22.0], [23.0, 24.0]];
    let matrix_32 = seed_matrix(&matrix_32_values).unwrap();
    let matrix_32_result = capture_dynamic_identity(&matrix_32).unwrap();
    assert_eq!(CACHE_FACTORY_CALLS.load(Ordering::Relaxed), 3);

    let warm_vector_values = [-1.0_f32, -2.0, -3.0, -4.0, -5.0, -6.0];
    let warm_vector = seed_vector(&warm_vector_values).unwrap();
    let warm_vector_result = capture_dynamic_identity(&warm_vector).unwrap();
    assert_eq!(CACHE_FACTORY_CALLS.load(Ordering::Relaxed), 3);

    let f32_pipeline_values = [[1.0_f32, -2.0, 3.0], [-4.0, 5.0, -6.0]];
    let f32_pipeline = seed_then_borrow_and_consume(&f32_pipeline_values).unwrap();
    assert_eq!(f32_pipeline.shape(), [2, 3]);
    let f32_resident = seed_matrix(&f32_pipeline_values).unwrap();
    let f32_borrowed_result = borrowed_owner(&f32_resident).unwrap();
    let f32_external_result = consume_relu(f32_borrowed_result).unwrap();
    assert_eq!(f32_external_result.shape(), [2, 3]);

    let f64_pipeline_values = [[16_777_217.0_f64, -16_777_219.0, 0.5], [0.125, -0.5, 2.0]];
    let f64_pipeline = seed_then_borrow_and_consume(&f64_pipeline_values).unwrap();
    assert_eq!(f64_pipeline.shape(), [2, 3]);
    let f64_resident = seed_matrix(&f64_pipeline_values).unwrap();
    let f64_borrowed_result = borrowed_owner(&f64_resident).unwrap();
    let f64_external_result = consume_relu(f64_borrowed_result).unwrap();
    assert_eq!(f64_external_result.shape(), [2, 3]);

    global::clear_thread_cache().unwrap();
    let after_clear_values = [[31.0_f32, 32.0, 33.0], [34.0, 35.0, 36.0]];
    let after_clear = seed_matrix(&after_clear_values).unwrap();
    let after_clear_result = capture_dynamic_identity(&after_clear).unwrap();
    assert_eq!(CACHE_FACTORY_CALLS.load(Ordering::Relaxed), 4);

    assert_f32_values(&matrix_23_result, [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    assert_f32_values(&changed_23_result, [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    assert_f32_values(&vector_6_result, vector_6_values);
    assert_f32_values(&matrix_32_result, [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]);
    assert_f32_values(&warm_vector_result, warm_vector_values);
    assert_f32_values(&after_clear_result, [31.0, 32.0, 33.0, 34.0, 35.0, 36.0]);

    assert_f32_values(&f32_pipeline, [2.0, -2.0, 6.0, -4.0, 10.0, -6.0]);
    assert_f32_values(&f32_external_result, [1.0, 0.0, 3.0, 0.0, 5.0, 0.0]);
    assert_f32_values(&f32_resident, [1.0, -2.0, 3.0, -4.0, 5.0, -6.0]);
    assert_f64_values(
        &f64_pipeline,
        [33_554_434.0, -16_777_219.0, 1.0, 0.25, -0.5, 4.0],
    );
    assert_f64_values(
        &f64_external_result,
        [16_777_217.0, 0.0, 0.5, 0.125, 0.0, 2.0],
    );

    assert_f64_values(
        &f64_resident,
        [16_777_217.0, -16_777_219.0, 0.5, 0.125, -0.5, 2.0],
    );
    global::clear_thread_cache().unwrap();
}

#[cfg(feature = "rocm")]
fn assert_f32_values(owner: &PcuTensor<f32>, expected: [f32; 6]) {
    let mut actual = [f32::NAN; 6];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f32::to_bits), expected.map(f32::to_bits));
}

#[cfg(feature = "rocm")]
fn assert_f64_values(owner: &PcuTensor<f64>, expected: [f64; 6]) {
    let mut actual = [f64::NAN; 6];
    owner.read_into(&mut actual).unwrap();
    assert_eq!(actual.map(f64::to_bits), expected.map(f64::to_bits));
}
