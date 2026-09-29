//! Generic scalar-owned source typing and execution.
#![cfg(feature = "tensor")]

use fusion_pcu::pcu;
#[cfg(feature = "rocm")]
use fusion_pcu::global;
#[cfg(feature = "rocm")]
use fusion_pcu::PcuScalarType;
#[cfg(feature = "rocm")]
use fusion_pcu_rocm::RocmTensorExecutionError;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn identity<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn through_helper<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(identity::<T, N>(input)?)
}

#[pcu]
fn where_identity<T, const N: usize>(input: &[T; N]) -> Result<PcuTensor<T>, PcuExecutionError>
where
    T: PcuScalar,
{
    Ok(pcu::identity(input)?)
}

#[pcu]
fn slice_identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::identity(input)?)
}

#[pcu]
fn through_slice<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(slice_identity(input)?)
}

#[pcu]
fn matrix_product<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    lhs: &[[T; K]; R],
    rhs: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::matmul(lhs, rhs)?)
}

#[pcu]
fn add_pair<T: PcuScalar, const N: usize>(
    lhs: &[T; N],
    rhs: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

#[test]
#[cfg(not(feature = "rocm"))]
fn generic_f32_f64_and_other_scalars_typecheck_without_host_fallback() {
    let f32_values = [1.0_f32, -2.0, 3.5];
    let f64_values = [16_777_217.0_f64, -2.0, 3.5];
    let u32_values = [1_u32, 2, 3];

    assert!(matches!(
        identity(&f32_values),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
    assert!(matches!(
        through_helper(&f64_values),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
    assert!(matches!(
        where_identity(&f64_values),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
    assert!(matches!(
        identity(&u32_values),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
    assert!(matches!(
        add_pair(&u32_values, &u32_values),
        Err(PcuExecutionError::TensorExecutionUnavailable)
    ));
}

#[test]
#[cfg(feature = "rocm")]
#[ignore = "requires ROCm hardware; also checks explicit unsupported-scalar admission"]
fn generic_owned_sources_support_f32_f64_and_helpers() {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = LOCK.lock().unwrap();
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let f32_input = [1.25_f32, -0.0, 3.5, -7.0];
    let f32_result = identity(&f32_input).unwrap();
    let f32_through_helper = through_helper::<f32, 4>(&f32_result).unwrap();
    let f32_changed = [-2.0_f32, 4.5, 0.25, 8.0];
    let f32_warm = through_helper(&f32_changed).unwrap();
    let mut f32_observed = [0.0_f32; 4];
    f32_result.read_into(&mut f32_observed).unwrap();
    assert_eq!(f32_observed.map(f32::to_bits), f32_input.map(f32::to_bits));
    f32_through_helper.read_into(&mut f32_observed).unwrap();
    assert_eq!(f32_observed.map(f32::to_bits), f32_input.map(f32::to_bits));
    f32_warm.read_into(&mut f32_observed).unwrap();
    assert_eq!(
        f32_observed.map(f32::to_bits),
        f32_changed.map(f32::to_bits)
    );

    let f64_input = [16_777_217.0_f64, -0.0, 1.0e-10, -3.0];
    let f64_result = through_helper(&f64_input).unwrap();
    let f64_where = where_identity(&f64_input).unwrap();
    let f64_slice = through_slice(&f64_input).unwrap();
    global::clear_thread_cache().unwrap();
    let f64_changed = [16_777_219.0_f64, 2.0, 1.0e-11, -4.0];
    let f64_warm = identity(&f64_changed).unwrap();
    let mut f64_observed = [0.0_f64; 4];
    f64_result.read_into(&mut f64_observed).unwrap();
    assert_eq!(f64_observed.map(f64::to_bits), f64_input.map(f64::to_bits));
    f64_where.read_into(&mut f64_observed).unwrap();
    assert_eq!(f64_observed.map(f64::to_bits), f64_input.map(f64::to_bits));
    f64_warm.read_into(&mut f64_observed).unwrap();
    assert_eq!(
        f64_observed.map(f64::to_bits),
        f64_changed.map(f64::to_bits)
    );
    f64_slice.read_into(&mut f64_observed).unwrap();
    assert_eq!(f64_observed.map(f64::to_bits), f64_input.map(f64::to_bits));

    let f32_left = [[1.0_f32, 2.0], [3.0, 4.0]];
    let f32_right = [[5.0_f32, 6.0], [7.0, 8.0]];
    let f32_matrix = matrix_product(&f32_left, &f32_right).unwrap();
    let mut f32_matrix_out = [0.0_f32; 4];
    f32_matrix.read_into(&mut f32_matrix_out).unwrap();
    assert_eq!(
        f32_matrix_out.map(f32::to_bits),
        [19.0_f32, 22.0, 43.0, 50.0].map(f32::to_bits)
    );

    let f64_left = [[16_777_217.0_f64, 1.0], [2.0, 3.0]];
    let f64_right = [[1.0_f64, 2.0], [4.0, 5.0]];
    let f64_matrix = matrix_product(&f64_left, &f64_right).unwrap();
    let mut f64_matrix_out = [0.0_f64; 4];
    f64_matrix.read_into(&mut f64_matrix_out).unwrap();
    assert_eq!(f64_matrix_out[0].to_bits(), 16_777_221.0_f64.to_bits());
    assert_eq!(f64_matrix_out[1].to_bits(), 33_554_439.0_f64.to_bits());
    assert_eq!(
        [f64_matrix_out[2].to_bits(), f64_matrix_out[3].to_bits()],
        [14.0_f64, 19.0].map(f64::to_bits)
    );

    let u32_values = [0_u32, u32::MAX, 0x8000_0001, 7];
    let u32_identity = identity(&u32_values).expect("u32 input-only transport is admitted");
    let mut u32_observed = [0_u32; 4];
    u32_identity.read_into(&mut u32_observed).unwrap();
    assert_eq!(u32_observed, u32_values);

    let error = add_pair(&u32_values, &u32_values)
        .expect_err("u32 arithmetic is not admitted by the current tensor backend");
    assert!(
        matches!(
            &error,
            PcuExecutionError::NoCompatibleDevice(rejections)
                if rejections.iter().any(|(_, error)| matches!(
                    error,
                    PcuExecutionError::TensorExecution(
                        RocmTensorExecutionError::UnsupportedScalarType(PcuScalarType::U32)
                    )
                ))
        ),
        "unsupported scalar rejection must retain its dtype: {error:?}"
    );
}
