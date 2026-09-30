//! Concrete annotated entries are the consumer authoring boundary.
use fusion_pcu::pcu;

#[pcu(invocations = N)]
pub fn negate_f32<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N)]
pub fn negate_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N, flag(reject_subnormal_result))]
pub fn negate_f32_reject_subnormal<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}

#[pcu(invocations = N, flag(reject_subnormal_result))]
pub fn negate_f64_reject_subnormal<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
