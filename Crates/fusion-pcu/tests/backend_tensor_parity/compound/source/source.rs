//! Strict operations retain their checks when other policies permit faster implementations.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
};

#[pcu(flag(strict))]
pub fn product<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(strict))]
pub fn dot<T: PcuScalar>(
    left: &[[T; 2]; 1],
    right: &[[T; 1]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(strict))]
pub fn loss<T: PcuScalar>(
    prediction: &[[T; 2]; 2],
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(strict))]
pub fn update<T: PcuScalar>(
    weights: &[[T; 2]; 2],
    gradient: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5)
}
