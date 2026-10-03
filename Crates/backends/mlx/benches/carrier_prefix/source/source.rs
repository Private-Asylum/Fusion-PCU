//! Genuine scalar source and owned retention; prefix reads never shrink the input owner.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn copy<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
#[pcu(invocations = 1, crate_path = ::pcu_facade)]
pub fn grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn broadcast<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
#[pcu(crate_path = ::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
