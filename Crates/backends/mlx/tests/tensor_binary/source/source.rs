//! Genuine annotated owned binary closures; explicit capture is the bounded backend peer.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn sub<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sub(right, left)
}
#[pcu(crate_path=::pcu_facade)]
pub fn mul<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn div<T: PcuCheckedFloat>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(right, left)
}
#[pcu(crate_path=::pcu_facade)]
pub fn unused<T: PcuCheckedFloat>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::div(right, left)?;
    pcu::identity(right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn repeat<T: PcuCheckedFloat>(
    unused: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(right, right)
}
