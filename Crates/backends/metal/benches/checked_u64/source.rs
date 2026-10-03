//! Actual source workload for every measured unsigned operation.
use pcu_facade::pcu;
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add<const N: usize>(lhs: &[u64], rhs: &[u64], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn sub<const N: usize>(lhs: &[u64], rhs: &[u64], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] - rhs[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul<const N: usize>(lhs: &[u64], rhs: &[u64], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] * rhs[id];
}
