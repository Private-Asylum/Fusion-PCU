//! Genuine generic borrowed-scalar transport has no arithmetic trait requirement.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn direct<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let id = context.global_invocation_id;
    output[id] = *input;
}
#[pcu(invocations=17,crate_path=::pcu_facade)]
pub fn grid<T: PcuScalar, const N: usize>(input: &T, output: &mut [T]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = *input;
        id += stride;
    }
}
