//! Genuine generic integer declarations with repeated and independently indexed reads.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn doubled<T: PcuCheckedInteger, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn squared<T: PcuCheckedInteger, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn reordered<T: PcuCheckedInteger, const N: usize>(right: &[T], output: &mut [T], left: &[T]) {
    let id = context.global_invocation_id;
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn independent<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] - input[0];
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedInteger, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[0] - input[id];
        id += stride;
    }
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn doubled_clamp<T: PcuCheckedInteger, const N: usize>(output: &mut [T], input: &[T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn squared_clamp<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    input: &[T],
) {
    let id = context.global_invocation_id;
    output[id] = input[id] * input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn reordered_clamp<T: PcuCheckedInteger, const N: usize>(
    right: &[T],
    output: &mut [T],
    left: &[T],
) {
    let id = context.global_invocation_id;
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade,flag(clamp_range))]
pub fn independent_clamp<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id] - input[0];
}
#[pcu(invocations=3,crate_path=::pcu_facade,flag(clamp_range))]
pub fn grid_clamp<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    input: &[T],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[0] - input[id];
        id += stride;
    }
}
