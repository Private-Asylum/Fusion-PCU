//! Genuine exact checked unary maps under explicitly requested Portable selection.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuCheckedFloat,
};
#[pcu(flag(deterministic), invocations=N, crate_path=::pcu_facade)]
pub fn neg<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(flag(deterministic), invocations=N, crate_path=::pcu_facade)]
pub fn relu<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[id]);
}

#[pcu(flag(deterministic), invocations=17, crate_path=::pcu_facade)]
pub fn grid<T: PcuCheckedFloat, const N: usize>(unused: &[T], output: &mut [T], input: &[T]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = -input[id];
        id += stride;
    }
}
#[pcu(flag(deterministic), invocations=N, crate_path=::pcu_facade)]
pub fn broadcast<T: PcuCheckedFloat, const N: usize>(output: &mut [T], input: &[T]) {
    let id = pcu::context::global_invocation_id();
    output[id] = pcu::relu(input[0]);
}
#[pcu(crate_path=::pcu_facade)]
pub fn identity<T: pcu_facade::PcuScalar>(
    input: &[T],
) -> Result<pcu_facade::PcuTensor<T>, pcu_facade::PcuExecutionError> {
    pcu::identity(input)
}
