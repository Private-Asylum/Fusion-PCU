//! Genuine ordered compound source; all returned values escape as actual native owners.
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuScalar,
    PcuTensor,
    PcuExecutionError,
};
#[pcu(flag(strict))]
pub fn product<T: PcuScalar>(
    left: &[[T; 3]; 2],
    right: &[[T; 2]; 3],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}
#[pcu(flag(strict))]
pub fn loss<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(left, right)
}
#[pcu(flag(strict))]
pub fn update<T: PcuScalar>(
    left: &[[T; 2]; 2],
    right: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(left, right, 0.5)
}
#[pcu]
pub fn identity<T: PcuScalar>(left: &[[T; 2]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(left)
}
#[pcu]
pub fn retain_left<T: PcuScalar>(left: &[[T; 3]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(left)
}
#[pcu]
pub fn retain_right<T: PcuScalar>(right: &[[T; 2]; 3]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(right)
}
#[pcu(flag(strict))]
pub fn chain<T: PcuScalar>(
    left: &[[T; 3]; 2],
    right: &[[T; 2]; 3],
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let product = pcu::matmul(left, right)?;
    let _checked_loss = pcu::mean_squared_error(&product, target)?;
    pcu::sgd_update(&product, target, 0.5)
}
