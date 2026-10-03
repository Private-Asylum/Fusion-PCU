//! Actual checked generic source workloads specialized cold to admitted encodings.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(invocations=N, flag(deterministic), crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),flag(deterministic), crate_path=::pcu_facade)]
pub fn sub<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(deterministic), crate_path=::pcu_facade)]
pub fn mul<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(deterministic), crate_path=::pcu_facade)]
pub fn div<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
