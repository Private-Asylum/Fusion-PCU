//! Actual checked source casts use exact conversion admission, not host Rust unchecked casts.
use pcu_facade::pcu;
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn widen<const N: usize>(input: &[f32], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = input[id] as f64;
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn narrow<const N: usize>(input: &[f64], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = input[id] as f32;
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn narrow_clamp<const N: usize>(input: &[f64], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = input[id] as f32;
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn widen_clamp<const N: usize>(input: &[f32], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = input[id] as f64;
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn broadcast_clamp<const N: usize>(input: &f64, output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = *input as f32;
}
#[pcu(invocations=3,crate_path=::pcu_facade,flag(clamp_range))]
pub fn grid_clamp(input: &[f64; 65], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < 65 {
        output[id] = input[id] as f32;
        id += stride;
    }
}
