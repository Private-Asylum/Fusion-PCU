//! Actual owned source parameter roles, effects and unread declarations.
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
pub fn div<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(right, left)
}
#[pcu(crate_path=::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn unused<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _checked = pcu::add(left, right)?;
    pcu::identity(right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn right_square<T: PcuScalar>(
    unused: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(right, right)
}
#[pcu(crate_path=::pcu_facade,invocations=7)]
pub fn invalidate(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
