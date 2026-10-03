//! Genuine target-selective differentiation; no caller-provided gradient tensor.
#[rustfmt::skip]
use pcu_facade::{pcu, PcuScalar, PcuTensor, PcuExecutionError};
#[pcu(flag(strict))]
pub fn train<T: PcuScalar>(
    input: &[[T; 2]; 2],
    weights: &[[T; 1]; 2],
    target: &[[T; 1]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let prediction = pcu::relu(pcu::matmul(input, weights));
    let loss = pcu::mean_squared_error(&prediction, target);
    let gradient = pcu::gradient(&loss, weights);
    pcu::sgd_update(weights, &gradient, 0.5)
}
#[pcu]
pub fn retain_input<T: PcuScalar>(input: &[[T; 2]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu]
pub fn retain_vector<T: PcuScalar>(input: &[[T; 1]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
