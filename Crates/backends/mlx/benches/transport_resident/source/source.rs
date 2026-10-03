//! Actual saved-load transport source; no numerical operations.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(invocations=N,crate_path=::pcu_facade)]
pub fn saved<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &[T],
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    stage[id] = input[id];
    let retained = stage[id];
    stage[id] = seed[0];
    output[id] = retained;
}
#[pcu(invocations=3,crate_path=::pcu_facade)]
pub fn saved_grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &[T],
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        stage[id] = input[id];
        let retained = stage[id];
        stage[id] = seed[0];
        output[id] = retained;
        id += stride;
    }
}

#[pcu(crate_path=::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
