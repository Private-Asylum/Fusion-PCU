//! The measured retained dispatch is authored through the ordinary function frontend.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(invocations = N)]
pub fn add<const N: usize>(left: &[u32], right: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = left[id] + right[0];
}

#[pcu]
pub fn retain(input: &[u32]) -> Result<PcuTensor<u32>, PcuExecutionError> {
    pcu::identity(input)
}
