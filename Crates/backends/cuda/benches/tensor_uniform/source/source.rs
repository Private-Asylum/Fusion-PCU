//! The measured retained dispatch is authored through the ordinary function frontend.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(invocations = N)]
pub fn add<const N: usize>(left: &[f32], right: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[0];
}

#[pcu]
pub fn retain(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
