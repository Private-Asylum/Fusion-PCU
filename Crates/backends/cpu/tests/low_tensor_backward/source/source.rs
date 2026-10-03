//! Ordinary owned derivative functions; no graph construction in the authored workload.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuCheckedFloat,PcuTensor,PcuExecutionError};
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuCheckedFloat>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn checked<T: PcuCheckedFloat>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuCheckedFloat>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn discarded<T: PcuCheckedFloat>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _unused = pcu::relu_backward(input, upstream)?;
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade,flag(native_compound),flag(backend_precision))]
pub fn permitted<T: PcuCheckedFloat>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(crate_path=::pcu_facade,flag(reject_subnormal_result))]
pub fn tight<T: PcuCheckedFloat>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(crate_path=::pcu_facade,flag(allow_gradual_underflow))]
pub fn allow<T: PcuCheckedFloat>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
#[pcu(crate_path=::pcu_facade,flag(deterministic))]
pub fn portable<T: PcuCheckedFloat>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}
