//! Genuine sealed generic wide maps with ordered stores, reloads and saved aliases.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuCheckedInteger,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn direct<T: PcuCheckedInteger, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original: T = input[id];
    stage[id] = original + *seed;
    let mut value: T = stage[id];
    value *= original;
    output[id] = value - original;
}
#[pcu(invocations = 3, crate_path = ::pcu_facade)]
pub fn grid<T: PcuCheckedInteger, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original: T = input[id];
        stage[id] = original + *seed;
        let mut value: T = stage[id];
        value *= original;
        output[id] = value - original;
        id += stride;
    }
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn overwritten<T: PcuCheckedInteger, const N: usize>(input: &[T], seed: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    output[id] = original - *seed;
    output[id] = original;
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn unused<T: PcuCheckedInteger, const N: usize>(input: &[T], seed: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    let _must_check = original - *seed;
    output[id] = original;
}
