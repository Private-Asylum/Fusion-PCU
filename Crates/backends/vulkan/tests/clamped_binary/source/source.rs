//! Genuine generic source retains checked instructions and explicit observable Clamp policy.
use pcu_facade::pcu;
use pcu_facade::PcuCheckedFloat;
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn add<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn sub<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn mul<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range))]
pub fn div<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] / right[id];
}
#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(clamp_range), flag(allow_gradual_underflow))]
pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(clamp_range), flag(strict), flag(reject_subnormal_result))]
pub fn scale<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * *right;
}
