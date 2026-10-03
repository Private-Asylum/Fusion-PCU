//! Real source lowering for ordered effect/resource eligibility tests.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
    PcuCheckedInteger,
};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
#[allow(unused_variables)] // Checked discarded values remain observable effects.
pub fn integer<T: PcuCheckedInteger, const N: usize>(
    ghost: &mut [T],
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    stage[id] = original;
    let saved_stage = stage[id];
    let first = original + *seed;
    let discarded = first + *seed;
    let product = first * saved_stage;
    output[id] = product - saved_stage;
}
#[pcu(invocations = 3, crate_path = ::pcu_facade)]
#[allow(unused_variables)] // A dead checked division is retained, not optimized away.
pub fn floating<T: PcuCheckedFloat, const N: usize>(
    ghost: &mut [T],
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original = input[id];
        stage[id] = original;
        let saved_stage = stage[id];
        let first = original + *seed;
        let discarded = first / *seed;
        let product = first * saved_stage;
        output[id] = product + saved_stage;
        id += stride;
    }
}
