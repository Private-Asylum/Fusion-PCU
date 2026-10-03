use super::*;

#[pcu]
pub(super) fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(invocations = N)]
pub(super) fn ordered<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let saved = stage[id];
    stage[id] = input[id];
    output[id] = saved;
    stage[id] = *seed;
}

#[pcu(invocations = 3)]
pub(super) fn grid<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    ghost: &mut [T],
    stage: &mut [T],
    output: &mut [T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        let saved = stage[id];
        stage[id] = input[id];
        output[id] = saved;
        stage[id] = *seed;
        id += stride;
    }
}

// Four actual initial snapshots: two readonly inputs and two exclusive old banks.
// All stores remain source semantics; physical input order follows first access.
#[pcu(invocations = N)]
pub(super) fn four_inputs<T: PcuScalar, const N: usize>(
    input: &[T],
    seed: &T,
    stage: &mut [T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let left = stage[id];
    let right = output[id];
    stage[id] = input[id];
    output[id] = *seed;
    stage[id] = right;
    output[id] = left;
}
