//! Every source measurement executes the annotated native loss operation.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuTensor,
    pcu,
};
#[pcu(flag(non_strict), flag(native_compound))]
pub fn loss<const N: usize>(
    prediction: &[f32; N],
    target: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu]
pub fn identity<const N: usize>(input: &[f32; N]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
