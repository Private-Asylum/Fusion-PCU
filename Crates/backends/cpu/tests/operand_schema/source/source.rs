//! Actual read roles, independent of declaration count and parameter order.
use pcu_facade::pcu;
use pcu_facade::PcuCheckedFloat;

#[pcu(invocations = N, crate_path=::pcu_facade)]
pub fn one<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}

#[pcu(invocations = N, crate_path=::pcu_facade)]
pub fn repeated<T: PcuCheckedFloat, const N: usize>(input: &[T], unused: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}

#[pcu(invocations = N, crate_path=::pcu_facade)]
pub fn reordered<T: PcuCheckedFloat, const N: usize>(output: &mut [T], unused: &[T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[id];
}

#[pcu(invocations = N, crate_path=::pcu_facade)]
pub fn independent<T: PcuCheckedFloat, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] / input[0];
}

#[pcu(invocations = 1, crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id] / input[id];
        id += stride;
    }
}
