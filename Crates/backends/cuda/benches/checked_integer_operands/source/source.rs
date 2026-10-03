//! Genuine integer operand-role workloads; unread declarations remain part of the source ABI.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn repeated_add<T: PcuCheckedInteger, const N: usize>(output: &mut [T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + left[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn unused_mul<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    left: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * left[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn reordered_sub<T: PcuCheckedInteger, const N: usize>(
    right: &[T],
    output: &mut [T],
    left: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn indexed_zero_sub<T: PcuCheckedInteger, const N: usize>(output: &mut [T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - left[0usize];
}
#[pcu(crate_path=::pcu_facade,invocations=17)]
pub fn grid_zero_sub<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    left: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[0] - left[id];
        id += stride;
    }
}
