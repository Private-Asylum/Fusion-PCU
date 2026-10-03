//! Actual ordinary owned source loss and two-sample training computation.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(flag(strict))]
pub fn loss<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu]
pub fn boundary_loss<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(native_compound))]
pub fn native_loss<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(deterministic))]
pub fn portable_loss<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(backend_precision))]
pub fn precision_loss<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu]
pub fn identity<T: PcuScalar, const N: usize>(
    input: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

/// Two samples: MSE gradient factor 2/N=1, preserving explicit checked loss side effects.
#[pcu(flag(strict))]
pub fn training<T: PcuScalar>(
    input: &[[T; 2]; 2],
    transpose: &[[T; 2]; 2],
    weights: &[[T; 1]; 2],
    target: &[[T; 1]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let activation = pcu::matmul(input, weights)?;
    let prediction = pcu::relu(&activation)?;
    let _loss = pcu::mean_squared_error(&prediction, target)?;
    let difference = pcu::sub(&prediction, target)?;
    let derivative = pcu::relu_backward(&activation, &difference)?;
    let gradient = pcu::matmul(transpose, &derivative)?;
    pcu::sgd_update(weights, &gradient, 0.5_f32)
}
