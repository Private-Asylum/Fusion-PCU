//! Actual source workload for every measured signed operation.
use pcu_facade::pcu;
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add<const N: usize>(lhs: &[i64], rhs: &[i64], output: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn sub<const N: usize>(lhs: &[i64], rhs: &[i64], output: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] - rhs[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul<const N: usize>(lhs: &[i64], rhs: &[i64], output: &mut [i64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] * rhs[id];
}
