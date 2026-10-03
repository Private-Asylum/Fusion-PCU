//! Genuine carrier workloads; no scalar arithmetic is requested.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuScalar};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn direct<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}
#[pcu(invocations=17,crate_path=::pcu_facade)]
pub fn grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn broadcast<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = *input;
}
#[pcu(invocations=17,crate_path=::pcu_facade)]
pub fn grid_broadcast<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = *input;
        id += stride;
    }
}
