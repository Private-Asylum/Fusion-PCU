//! Actual forward/loss/reverse/update authoring; no supplied or host-computed gradient.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(flag(non_strict), flag(native_compound))]
pub fn multiply_subtract<const R: usize, const K: usize>(
    samples: &[[f32; K]; R],
    weights: &[[f32; 1]; K],
    target: &[[f32; 1]; R],
    rate: &[[f32; 1]; K],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let prediction = pcu::matmul(samples, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    let gradient = pcu::gradient(&loss, weights);
    let scaled = pcu::mul(&gradient, rate);
    pcu::sub(weights, &scaled)
}

#[pcu(flag(non_strict), flag(native_compound))]
pub fn separate_update<const R: usize, const K: usize>(
    samples: &[[f32; K]; R],
    weights: &[[f32; 1]; K],
    target: &[[f32; 1]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let prediction = pcu::matmul(samples, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    let gradient = pcu::gradient(&loss, weights);
    pcu::sgd_update(weights, &gradient, 0.001_f32)
}

#[pcu(flag(strict))]
pub fn checked<const R: usize, const K: usize>(
    samples: &[[f32; K]; R],
    weights: &[[f32; 1]; K],
    target: &[[f32; 1]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let prediction = pcu::matmul(samples, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    let gradient = pcu::gradient(&loss, weights);
    pcu::sgd_update(weights, &gradient, 0.001_f32)
}

#[pcu]
pub fn retain<const R: usize, const K: usize>(
    input: &[[f32; K]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}
