//! Source measurements execute these annotated operations; no backend APIs enter the workload.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuTensor,
    pcu,
};

#[pcu(flag(native_compound), flag(preserve_precision))]
pub fn preserved<const N: usize>(
    weights: &[f32; N],
    gradient: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}

#[pcu(flag(native_compound), flag(backend_precision))]
pub fn contracted<const N: usize>(
    weights: &[f32; N],
    gradient: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}

#[pcu]
pub fn identity<const N: usize>(input: &[f32; N]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
