//! Actual checked generic source workloads specialized cold to admitted encodings.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(invocations=N, crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),crate_path=::pcu_facade)]
pub fn sub<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn mul<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn div<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range),flag(reject_subnormal_result),crate_path=::pcu_facade)]
pub fn add_tight_clamp<T: PcuCheckedFloat, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn swapped<T: PcuCheckedFloat, const N: usize>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn repeated<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * left[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn broadcast<T: PcuCheckedFloat, const N: usize>(left: &T, right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *left + right[id];
}
#[pcu(invocations=1,crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}

#[pcu(crate_path=::pcu_facade)]
pub fn retain<T: PcuCheckedFloat>(
    input: &[T],
) -> Result<pcu_facade::PcuTensor<T>, pcu_facade::PcuExecutionError> {
    pcu::identity(input)
}
