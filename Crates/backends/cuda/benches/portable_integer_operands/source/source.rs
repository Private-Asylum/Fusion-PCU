//! Requested portable integer workloads; identity remains ordinary transport.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedInteger,
    PcuCheckedIntegerDivision,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn repeated_add<T: PcuCheckedInteger, const N: usize>(output: &mut [T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + left[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn unused_mul<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    left: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * left[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn reordered_sub<T: PcuCheckedInteger, const N: usize>(
    right: &[T],
    output: &mut [T],
    left: &[T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn indexed_zero_sub<T: PcuCheckedInteger, const N: usize>(output: &mut [T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - left[0usize];
}
#[pcu(crate_path=::pcu_facade,invocations=17,flag(deterministic))]
pub fn grid_zero_sub<T: PcuCheckedInteger, const N: usize>(
    unused: &[T],
    output: &mut [T],
    left: &[T],
) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = left[0] - left[id];
        id += stride;
    }
}

#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn zero_broadcast_add<T: PcuCheckedInteger, const N: usize>(output: &mut [T], left: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[0] + left[0usize];
}

#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn unsupported_transport<T: PcuScalar, const N: usize>(left: &[T], output: &mut [T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id];
}
#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn requested_div_rem<T: PcuCheckedIntegerDivision, const N: usize>(
    left: &[T],
    right: &[T],
    quotient: &mut [T],
    remainder: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    let (q, r) = pcu::checked_div_rem(left[id], right[id]);
    quotient[id] = q;
    remainder[id] = r;
}
#[pcu(crate_path=::pcu_facade,flag(deterministic))]
pub fn unsupported_owned<T: PcuScalar>(left: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::add(left, left)
}

#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn distinct_add<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[id];
}

#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn distinct_sub<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] - right[id];
}

#[pcu(crate_path=::pcu_facade,invocations=N,flag(deterministic))]
pub fn distinct_mul<T: PcuCheckedInteger, const N: usize>(
    left: &[T],
    right: &[T],
    output: &mut [T],
) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] * right[id];
}

#[pcu(crate_path=::pcu_facade)]
pub fn owned_repeated<T: PcuScalar>(
    unused: &[T],
    input: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mul(input, input)
}
#[pcu(crate_path=::pcu_facade)]
pub fn owned_discarded<T: PcuScalar>(
    effect_input: &[T],
    input: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::mul(effect_input, effect_input);
    pcu::identity(input)
}

#[pcu(crate_path=::pcu_facade)]
pub fn owned_sub<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sub(input, input)
}

// Identity is no arithmetic and does not rewrite the original producer's policy.
#[pcu(crate_path=::pcu_facade,flag(deterministic))]
pub fn scoped_identity<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
