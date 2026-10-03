//! Genuine mutable-local spelling of the unchanged ordered two-output workload.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn direct<T: PcuCheckedFloat, const N: usize>(stage: &mut [T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    let original = input[id];
    let mut value = original;
    value = value + value;
    stage[id] = value;
    value = stage[id];
    value = value * original;
    output[id] = value;
}
#[pcu(crate_path=::pcu_facade,invocations=17)]
pub fn grid<T: PcuCheckedFloat, const N: usize>(stage: &mut [T], output: &mut [T], input: &[T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let original = input[id];
        let mut value = original;
        value = value + value;
        stage[id] = value;
        value = stage[id];
        value = value * original;
        output[id] = value;
        id += stride;
    }
}
