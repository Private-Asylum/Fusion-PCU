//! Each timed ordinary call uses the public per-function source boundary.
use fusion_pcu::pcu;
#[pcu(invocations = N)]
pub fn narrow<const N: usize>(input: &[f64; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}
#[pcu(invocations = N, flag(strict))]
pub fn narrow_strict<const N: usize>(input: &[f64; N], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f32;
}
#[pcu(invocations = N)]
pub fn widen<const N: usize>(input: &[f32; N], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f64;
}
#[pcu(invocations = N, flag(strict))]
pub fn widen_strict<const N: usize>(input: &[f32; N], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f64;
}
