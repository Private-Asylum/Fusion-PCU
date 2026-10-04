//! Genuine scalar slice-broadcast source, without arithmetic interpretation.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
};
#[pcu(crate_path=::pcu_facade, invocations=N)]
pub fn direct<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[0];
}
#[pcu(crate_path=::pcu_facade, invocations=17)]
pub fn grid<T: PcuScalar, const N: usize>(input: &[T], output: &mut [T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[0usize];
        id += stride;
    }
}
