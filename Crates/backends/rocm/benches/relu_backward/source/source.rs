//! Ordinary per-function derivative profiles; all calls perform real tensor work.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu]
pub fn checked<T: PcuScalar, const N: usize>(
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(strict))]
pub fn strict<T: PcuScalar, const N: usize>(
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(reject_subnormal_result))]
pub fn tight<T: PcuScalar, const N: usize>(
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(native_compound))]
pub fn native<T: PcuScalar, const N: usize>(
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(strict), flag(native_compound))]
pub fn strict_native<T: PcuScalar, const N: usize>(
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(deterministic))]
pub fn portable<T: PcuScalar, const N: usize>(
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(flag(strict))]
pub fn optimizer<T: PcuScalar, const N: usize>(
    weights: &[T; N],
    input: &[T; N],
    upstream: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let gradient = pcu::relu_backward(input, upstream)?;
    pcu::sgd_update(weights, &gradient, 0.5_f32)
}

#[pcu]
pub fn identity<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
