//! Actual generic integer Clamp source without consumer-side helper arithmetic.
use pcu_facade::{pcu, PcuCheckedInteger};
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn sub<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn mul<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] * right[id];
}
#[pcu(invocations=3,crate_path=::pcu_facade,flag(clamp_range),flag(strict))]
pub fn grid<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &T, output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = left[id] * *right;
        id += stride;
    }
}
