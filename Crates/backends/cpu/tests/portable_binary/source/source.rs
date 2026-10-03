//! Actual generic annotated workloads preserve the selected scalar format.
use pcu_facade::pcu;
use pcu_facade::PcuCheckedFloat;
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(deterministic))]
pub fn add<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(deterministic))]
pub fn sub<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] - right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(deterministic))]
pub fn mul<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(deterministic))]
pub fn div<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] / right[id];
}
#[pcu(invocations = 3, crate_path = ::pcu_facade, flag(allow_gradual_underflow))]
pub fn grid<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id] / right[id];
        id += stride;
    }
}
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn scale<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * *right;
}
#[pcu(invocations = N, crate_path = ::pcu_facade, flag(strict), flag(reject_subnormal_result))]
pub fn strict<T: PcuCheckedFloat, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
