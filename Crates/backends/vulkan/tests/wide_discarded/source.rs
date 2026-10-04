//! Genuine discarded checked Add; normal and requested Portable are independent source declarations.
use pcu_facade::{pcu, PcuCheckedInteger};
#[pcu(invocations = N, crate_path = ::pcu_facade)]
pub fn normal_direct<T: PcuCheckedInteger, const N: usize>(
    input: &[T],
    seed: &T,
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original: T = input[id];
    let _discarded: T = original + *seed;
    output[id] = original;
}
#[pcu(invocations = 3, crate_path = ::pcu_facade)]
pub fn normal_grid<T: PcuCheckedInteger, const N: usize>(input: &[T], seed: &T, output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original: T = input[id];
        let _discarded: T = original + *seed;
        output[id] = original;
        id += stride;
    }
}
#[pcu(invocations = N, flag(deterministic), crate_path = ::pcu_facade)]
pub fn portable_direct<T: PcuCheckedInteger, const N: usize>(
    input: &[T],
    seed: &T,
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let original: T = input[id];
    let _discarded: T = original + *seed;
    output[id] = original;
}
#[pcu(invocations = 3, flag(deterministic), crate_path = ::pcu_facade)]
pub fn portable_grid<T: PcuCheckedInteger, const N: usize>(
    input: &[T],
    seed: &T,
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original: T = input[id];
        let _discarded: T = original + *seed;
        output[id] = original;
        id += stride;
    }
}
