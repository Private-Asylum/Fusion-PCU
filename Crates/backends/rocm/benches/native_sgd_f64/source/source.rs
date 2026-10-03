//! Ordinary per-function native F64 updates with a finite frozen F32 rate.
#[rustfmt::skip]
use fusion_pcu::{pcu, PcuScalar, PcuTensor, PcuExecutionError};
#[pcu(flag(native_compound), flag(preserve_precision))]
pub fn preserve<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.25_f32)
}
#[pcu(flag(native_compound), flag(backend_precision))]
pub fn optimized<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.25_f32)
}
pub fn update<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
    optimized_precision: bool,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    if optimized_precision {
        optimized(weights, gradient)
    } else {
        preserve(weights, gradient)
    }
}
#[pcu]
pub fn identity<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(flag(native_compound), flag(preserve_precision))]
pub fn witness_preserve(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}
#[pcu(flag(native_compound), flag(backend_precision))]
pub fn witness_optimized(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}
#[pcu(flag(native_compound))]
pub fn positive_zero(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.0_f32)
}
#[pcu(flag(native_compound))]
pub fn negative_zero(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.0_f32)
}
#[pcu(flag(native_compound))]
pub fn extreme_rate(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 3.402_823_5e38_f32)
}
#[pcu]
pub fn checked(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
#[pcu(flag(native_compound), flag(strict))]
pub fn strict_native(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
#[pcu(flag(native_compound), flag(deterministic))]
pub fn portable(
    weights: &[f64; 1],
    gradient: &[f64; 1],
) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
#[pcu(flag(native_compound), flag(reject_subnormal_result))]
pub fn tight(weights: &[f64; 1], gradient: &[f64; 1]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
