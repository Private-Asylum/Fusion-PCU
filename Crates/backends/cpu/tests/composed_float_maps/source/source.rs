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
