//! Real scalar locals and ordered stores; both outputs are observed.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(crate_path=::pcu_facade,invocations=N)]
pub fn direct<T: PcuCheckedFloat, const N: usize>(stage: &mut [T], output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    let value = input[id];
    let doubled = value + value;
    stage[id] = doubled;
    let reloaded = stage[id];
    output[id] = reloaded * value;
}
#[pcu(crate_path=::pcu_facade,invocations=17)]
pub fn grid<T: PcuCheckedFloat, const N: usize>(stage: &mut [T], output: &mut [T], input: &[T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let value = input[id];
        let doubled = value + value;
        stage[id] = doubled;
        let reloaded = stage[id];
        output[id] = reloaded * value;
        id += stride;
    }
}
