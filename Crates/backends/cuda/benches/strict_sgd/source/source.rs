//! Canonical ordinary annotated Strict optimizer entry.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(flag(strict))]
pub fn update<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu(flag(strict), flag(reject_subnormal_result))]
pub fn tight<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu(flag(strict), flag(allow_gradual_underflow))]
pub fn gradual<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu(flag(strict))]
pub fn zero<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, -0.0_f32)
}
#[pcu(flag(strict))]
pub fn overflow<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 2.0_f32)
}
#[pcu(flag(strict))]
pub fn witness<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}
#[pcu(flag(non_strict))]
pub fn boundary<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu(flag(strict), flag(native_compound))]
pub fn native_strict<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu(flag(strict), flag(deterministic))]
pub fn portable<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu]
pub fn identity<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(flag(strict))]
pub fn chain<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    gradient: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let updated = pcu::sgd_update(weights, gradient, 0.5_f32)?;
    pcu::sgd_update(&updated, gradient, 0.5_f32)
}
