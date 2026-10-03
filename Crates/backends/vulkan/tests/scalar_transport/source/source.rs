//! Genuine generic identity and borrowed scalar broadcast, with no arithmetic consumer bound.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuScalar};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn dense<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}
#[pcu(invocations=17,crate_path=::pcu_facade)]
pub fn dense_grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn broadcast<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = *input;
}
#[pcu(invocations=17,crate_path=::pcu_facade)]
pub fn broadcast_grid<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = *input;
        id += stride;
    }
}
