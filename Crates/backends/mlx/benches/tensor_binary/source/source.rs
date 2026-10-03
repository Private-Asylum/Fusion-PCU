//! Genuine annotated owned binary closures; ordinary calls are the fourth ownership peer.
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
