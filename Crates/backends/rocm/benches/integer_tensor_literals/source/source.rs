//! Closest genuine source: literal banks are inputs until owned literal syntax exists.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuScalar,PcuTensor,PcuExecutionError};
#[pcu(crate_path=::pcu_facade)]
pub fn pipeline<T: PcuScalar>(
    input: &[T],
    constant: &[T],
    uniform: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = pcu::add(input, constant)?;
    pcu::mul(&sum, uniform)
}
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(crate_path=::pcu_facade)]
pub fn mixed<T: PcuScalar>(
    input: &PcuTensor<T>,
    constant: &[T],
    uniform: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = pcu::add(input, constant)?;
    pcu::mul(&sum, uniform)
}
