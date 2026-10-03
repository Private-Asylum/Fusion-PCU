//! Repeated and broadcast integer operands use their actual source binding roles.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuCheckedInteger,
};

#[pcu(invocations = N)]
pub fn doubled<T: PcuCheckedInteger, const N: usize>(output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] + input[id];
}

#[pcu(invocations = N)]
pub fn squared<T: PcuCheckedInteger, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * input[id];
}

#[pcu(invocations = N)]
pub fn reordered<T: PcuCheckedInteger, const N: usize>(right: &[T], output: &mut [T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}

#[pcu(invocations = N)]
pub fn independent<T: PcuCheckedInteger, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] - input[0];
}

#[pcu(invocations = 3)]
pub fn grid<T: PcuCheckedInteger, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[0] - input[id];
        id += stride;
    }
}
