//! One source specialization supplies all compared routes.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(invocations = N)]
pub fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0 + 1.0;
}

#[pcu]
pub fn identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
