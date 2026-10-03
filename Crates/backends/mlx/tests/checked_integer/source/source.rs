//! Genuine generic fourteen-width source maps, with policies specialized only during preparation.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
};
#[pcu(invocations=N, crate_path=::pcu_facade)]
pub fn add<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range), crate_path=::pcu_facade)]
pub fn add_clamp<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}
#[pcu(invocations=N, crate_path=::pcu_facade)]
pub fn sub<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range), crate_path=::pcu_facade)]
pub fn sub_clamp<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N, crate_path=::pcu_facade)]
pub fn mul<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,flag(strict),flag(clamp_range), crate_path=::pcu_facade)]
pub fn mul_clamp<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn swapped<T: PcuCheckedInteger, const N: usize>(output: &mut [T], right: &[T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn broadcast<T: PcuCheckedInteger, const N: usize>(left: &T, right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *left + right[id];
}
#[pcu(invocations=1,crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedInteger, const N: usize>(left: &[T], right: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[id] * right[id];
        id += stride;
    }
}

#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn repeated<T: PcuCheckedInteger, const N: usize>(unused: &[T], right: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = right[id] * right[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn single<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn mixed_indices<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0] + input[id];
}
