//! Genuine source programs preserve loaded SSA values across later same-resource stores.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
};

#[pcu(crate_path = ::pcu_facade, invocations = N)]
pub fn ordered<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let saved = stage[id];
    stage[id] = *seed;
    output[id] = saved;
}
#[pcu(crate_path = ::pcu_facade, invocations = 3)]
pub fn grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        stage[id] = input[id];
        let saved = stage[id];
        stage[id] = *seed;
        output[id] = saved;
        id += stride;
    }
}

#[pcu(crate_path = ::pcu_facade, invocations = N, flag(clamp_range))]
pub fn clamped_ordered<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let saved = stage[id];
    stage[id] = *seed;
    output[id] = saved;
}
#[pcu(crate_path = ::pcu_facade, invocations = 3, flag(clamp_range))]
pub fn clamped_grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        stage[id] = input[id];
        let saved = stage[id];
        stage[id] = *seed;
        output[id] = saved;
        id += stride;
    }
}
