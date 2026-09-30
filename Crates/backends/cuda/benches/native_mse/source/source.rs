//! Actual authored owned scalar loss under explicit, independent native permissions.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(preserve_precision),
    flag(non_deterministic)
)]
pub fn preserve<const N: usize>(
    prediction: &[f32; N],
    target: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
pub fn optimized<const N: usize>(
    prediction: &[f32; N],
    target: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu]
pub fn checked(
    prediction: &[f32; 1],
    target: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(strict), flag(native_compound), flag(non_deterministic))]
pub fn strict(
    prediction: &[f32; 1],
    target: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(non_strict), flag(native_compound), flag(deterministic))]
pub fn portable(
    prediction: &[f32; 1],
    target: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(non_deterministic),
    flag(reject_subnormal_result)
)]
pub fn tight(
    prediction: &[f32; 1],
    target: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

pub fn loss<const N: usize>(
    prediction: &[f32; N],
    target: &[f32; N],
    optimized_precision: bool,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    if optimized_precision {
        optimized(prediction, target)
    } else {
        preserve(prediction, target)
    }
}

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(preserve_precision),
    flag(non_deterministic)
)]
pub fn project_loss(
    left: &[[f32; 2]; 2],
    right: &[[f32; 2]; 2],
    target: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let projected = pcu::matmul(left, right)?;
    pcu::mean_squared_error(&projected, target)
}
