//! The same ordinary annotated source supplies global and explicit prepared calls.
use fusion_pcu::pcu;

#[pcu(invocations = N)]
pub fn negate<const N: usize>(input: &[f32; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations = N)]
pub fn add<const N: usize>(lhs: &[u64; N], rhs: &[u64; N], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] + rhs[id];
}
#[pcu(invocations = N)]
pub fn mul<const N: usize>(lhs: &[u64; N], rhs: &[u64; N], output: &mut [u64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = lhs[id] * rhs[id];
}
