//! Ordinary generic source identity; wide float payloads never undergo arithmetic.
#[rustfmt::skip]
use pcu_facade::{pcu, PcuScalar, PcuTensor, PcuExecutionError};
#[pcu(invocations=N, crate_path=::pcu_facade)]
pub fn direct<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
#[pcu(invocations=17, crate_path=::pcu_facade)]
pub fn grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
