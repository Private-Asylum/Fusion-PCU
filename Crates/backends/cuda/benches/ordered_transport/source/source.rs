//! Genuine typed ordered byte transport, without arithmetic or numeric conversion.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn direct<T: PcuScalar, const N: usize>(
    input: &[T],
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let mut value = input[id];
    stage[id] = value;
    value = stage[id];
    output[id] = value;
}
#[pcu(crate_path=::pcu_facade,invocations=17)]
pub fn grid<T: PcuScalar, const N: usize>(
    input: &[T],
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let mut value = input[id];
        stage[id] = value;
        value = stage[id];
        output[id] = value;
        id += stride;
    }
}
