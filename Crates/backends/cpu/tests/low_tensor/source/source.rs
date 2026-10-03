//! Ordinary owned floating source functions preserve checked graph effects.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuCheckedFloat>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedFloat>(a: &[T], b: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(a + b)
}
#[pcu(crate_path=::pcu_facade)]
pub fn sub<T: PcuCheckedFloat>(a: &[T], b: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(a - b)
}
#[pcu(crate_path=::pcu_facade)]
pub fn mul<T: PcuCheckedFloat>(a: &[T], b: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(a * b)
}
#[pcu(crate_path=::pcu_facade)]
pub fn consumed_identity<T: PcuCheckedFloat>(
    input: PcuTensor<T>,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn discarded_add<T: PcuCheckedFloat>(
    a: &[T],
    b: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _unused = a + b;
    pcu::identity(a)
}

#[pcu(crate_path=::pcu_facade,flag(native_compound),flag(backend_precision))]
pub fn permitted_mul<T: PcuCheckedFloat>(
    a: &[T],
    b: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(a * b)
}

#[pcu(crate_path=::pcu_facade)]
pub fn div<T: PcuCheckedFloat>(a: &[T], b: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(a, b)
}
#[pcu(crate_path=::pcu_facade)]
pub fn relu<T: PcuCheckedFloat>(a: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(a)
}
