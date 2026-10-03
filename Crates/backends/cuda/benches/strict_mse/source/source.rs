//! Ordinary source loss profiles and a checked producer/consumer fault boundary.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};
#[pcu(flag(strict))]
pub fn strict<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu]
pub fn boundary<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(reject_subnormal_result))]
pub fn tight<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(allow_gradual_underflow))]
pub fn gradual<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(deterministic))]
pub fn portable<T: PcuScalar, const N: usize>(
    prediction: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict), flag(backend_precision))]
pub fn optimized<T: PcuScalar, const N: usize>(
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
#[pcu(flag(strict))]
pub fn forward_loss<T: PcuScalar, const N: usize>(
    input: &[T; N],
    target: &[T; N],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let predicted = pcu::relu(input)?;
    pcu::mean_squared_error(&predicted, target)
}

/// Full bounded two-feature, two-sample forward/loss/backward/update; gradient scale=2/N=1.
#[pcu(flag(strict))]
pub fn training_step<T: PcuScalar>(
    input: &[[T; 2]; 2],
    input_transpose: &[[T; 2]; 2],
    weights: &[[T; 1]; 2],
    target: &[[T; 1]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let preactivation = pcu::matmul(input, weights)?;
    let predicted = pcu::relu(&preactivation)?;
    let _loss = pcu::mean_squared_error(&predicted, target)?;
    let difference = pcu::sub(&predicted, target)?;
    let derivative = pcu::relu_backward(&preactivation, &difference)?;
    let gradient = pcu::matmul(input_transpose, &derivative)?;
    pcu::sgd_update(weights, &gradient, 0.5_f32)
}
