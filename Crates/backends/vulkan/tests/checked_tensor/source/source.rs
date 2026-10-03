//! Genuine generic owned source; warm companions execute actual retained native graph work.
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
    pcu::sub(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn mul<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn div<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn relu<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(crate_path=::pcu_facade)]
pub fn checked_unused<T: PcuScalar>(
    left: &[T],
    right: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::div(left, right)?;
    pcu::identity(left)
}

#[pcu(crate_path=::pcu_facade)]
pub fn backward<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(left, right)
}

pub fn call<
    T: PcuScalar,
    A: pcu_facade::PcuTensorSource<T> + ?Sized,
    B: pcu_facade::PcuTensorSource<T> + ?Sized,
>(
    op: u32,
    a: &A,
    b: &B,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    match op {
        0 => add(a, b),
        1 => sub(a, b),
        2 => mul(a, b),
        3 => div(a, b),
        4 => relu(a),
        5 => backward(a, b),
        _ => unreachable!(),
    }
}
