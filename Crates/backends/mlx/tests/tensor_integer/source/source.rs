//! Actual generic owned source captures; ordinary provider lift is separate.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
pub fn add<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn sub<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sub(right, left)
}
#[pcu(crate_path=::pcu_facade)]
pub fn mul<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn repeated<T: PcuScalar>(
    unused: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(right, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn checked_unused<T: PcuScalar>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::add(left, right);
    pcu::identity(left)
}
