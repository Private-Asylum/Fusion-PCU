//! Genuine nested checked expressions; each operation retains its source policy.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn reject_ieee<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(allow_gradual_underflow))]
pub fn reject_gradual<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(strict),flag(reject_subnormal_result))]
pub fn reject_tight<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(clamp_range))]
pub fn clamp_ieee<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(clamp_range),flag(allow_gradual_underflow))]
pub fn clamp_gradual<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(clamp_range),flag(strict),flag(reject_subnormal_result))]
pub fn clamp_tight<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] + input[id]) * input[id];
}
#[pcu(crate_path=::pcu_facade,invocations=3)]
pub fn grid<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = (input[id] + input[id]) * input[id];
        id += stride;
    }
}
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn broadcast<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], input: &T) {
    let id = context.global_invocation_id;
    output[id] = (*input + *input) * *input;
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn ordered<T: PcuCheckedFloat, const N: usize>(input: &[T], stage: &mut [T], output: &mut [T]) {
    let id = context.global_invocation_id;
    let original = input[id];
    stage[id] = original + original;
    let loaded = stage[id];
    output[id] = loaded * original;
}

#[pcu(crate_path=::pcu_facade,invocations=N,flag(reject_subnormal_result),flag(clamp_range))]
pub fn ordered_clamp<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    stage[id] = input[id] + input[id];
    output[id] = stage[id] * input[id];
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn unused_fault<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    divisor: &[T],
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    let _must_check = input[id] / divisor[id];
    output[id] = input[id] + input[id];
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn mixed<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = (input[id] - input[id]) + pcu::relu(-input[id]);
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn constant_f32<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = ((input[id] + 2.0_f32) / 2.0_f32) - input[id];
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn constant_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = ((input[id] + 2.0_f64) / 2.0_f64) - input[id];
}

#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn element_zero_after_store<T: PcuCheckedFloat, const N: usize>(
    input: &[T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = context.global_invocation_id;
    stage[id] = input[id] + input[id];
    output[id] = stage[0] * input[id];
}
