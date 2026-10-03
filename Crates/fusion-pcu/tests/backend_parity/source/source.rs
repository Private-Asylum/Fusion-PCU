//! Identical ordinary source workloads for every scalar execution provider.
use fusion_pcu::pcu;
use fusion_pcu::PcuCheckedFloat;

#[pcu(invocations = N)]
pub fn add<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}

#[pcu(invocations = N)]
pub fn sub<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}

#[pcu(invocations = N)]
pub fn mul<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

#[pcu(invocations = N)]
pub fn div<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
