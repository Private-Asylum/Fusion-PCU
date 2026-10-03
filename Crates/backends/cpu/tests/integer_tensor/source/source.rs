//! Genuine owned tensor maps preserve consumer scalar bounds and checked effects.
extern crate pcu_facade as fusion_pcu;
#[rustfmt::skip]
use fusion_pcu::{pcu,PcuScalar,PcuExecutionError,PcuTensor};
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
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
#[pcu(crate_path=::pcu_facade,flag(strict), flag(native_compound), flag(backend_precision))]
pub fn permitted<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu(crate_path=::pcu_facade,flag(deterministic))]
pub fn portable<T: PcuScalar>(left: &[T], right: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, right)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
