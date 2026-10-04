//! Ordinary Rust borrows retain the actual selected numerical graph and local mode.
#[rustfmt::skip]
use fusion_pcu::{pcu,PcuScalar,PcuTensor,PcuExecutionError};
#[pcu]
pub fn retain<T: PcuScalar>(input: &[[T; 2]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(flag(strict))]
pub fn product<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(strict))]
pub fn derivative<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(left, right)
}
#[pcu(flag(strict))]
pub fn update<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(left, right, 0.5)
}
#[pcu(flag(strict))]
pub fn discarded_product<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::matmul(left, right)?;
    pcu::identity(right)
}
#[pcu(flag(strict))]
pub fn discarded_derivative<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::relu_backward(left, right)?;
    pcu::identity(right)
}
#[pcu(flag(strict))]
pub fn discarded_update<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::sgd_update(left, right, 0.5)?;
    pcu::identity(right)
}

#[pcu(flag(strict))]
pub fn loss<T: PcuScalar>(
    prediction: &[[T; 2]; 2],
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}
#[pcu(flag(strict))]
pub fn discarded_loss<T: PcuScalar>(
    prediction: &[[T; 2]; 2],
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::mean_squared_error(prediction, target)?;
    pcu::identity(target)
}
