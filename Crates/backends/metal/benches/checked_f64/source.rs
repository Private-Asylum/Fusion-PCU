//! Genuine annotated workload shared by matched benchmark peers and the example.
use pcu_facade::pcu;
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn add<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, flag(strict), crate_path = ::pcu_facade)]
pub fn sub<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn mul<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn div<const N: usize>(left: &[f64], right: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] / right[id];
}
