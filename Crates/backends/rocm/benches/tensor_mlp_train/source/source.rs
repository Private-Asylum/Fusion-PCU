//! One forward, one requested reverse union, and all three SGD updates in one source capture.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu(flag(non_strict), flag(native_compound))]
#[allow(clippy::type_complexity)] // Four heterogeneous shapes share the same F32 owner scalar.
pub fn train<const B: usize, const I: usize, const H: usize, const O: usize>(
    samples: &[[f32; I]; B],
    w1: &[[f32; H]; I],
    w2: &[[f32; H]; H],
    w3: &[[f32; O]; H],
    targets: &[[f32; O]; B],
) -> Result<
    (
        PcuTensor<f32>,
        PcuTensor<f32>,
        PcuTensor<f32>,
        PcuTensor<f32>,
    ),
    PcuExecutionError,
> {
    let z1 = pcu::matmul(samples, w1);
    let h1 = pcu::relu(&z1);
    let z2 = pcu::matmul(&h1, w2);
    let h2 = pcu::relu(&z2);
    let prediction = pcu::matmul(&h2, w3);
    let loss = pcu::mean_squared_error(&prediction, targets);
    let (g1, g2, g3) = pcu::gradients(&loss, (w1, w2, w3));
    let updated_w1 = pcu::sgd_update(w1, &g1, 0.000_001_f32);
    let updated_w2 = pcu::sgd_update(w2, &g2, 0.000_001_f32);
    let updated_w3 = pcu::sgd_update(w3, &g3, 0.000_001_f32);
    (updated_w1, updated_w2, updated_w3, loss)
}

#[pcu]
pub fn retain<const R: usize, const K: usize>(
    input: &[[f32; K]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

/// Strict small diagnostic counterpart; backend-native GPU numerical policy is separately qualified.
#[pcu(flag(strict))]
#[allow(clippy::type_complexity)] // The complete training result has four owner roles.
pub fn checked<const B: usize, const I: usize, const H: usize, const O: usize>(
    samples: &[[f32; I]; B],
    w1: &[[f32; H]; I],
    w2: &[[f32; H]; H],
    w3: &[[f32; O]; H],
    targets: &[[f32; O]; B],
) -> Result<
    (
        PcuTensor<f32>,
        PcuTensor<f32>,
        PcuTensor<f32>,
        PcuTensor<f32>,
    ),
    PcuExecutionError,
> {
    let z1 = pcu::matmul(samples, w1);
    let h1 = pcu::relu(&z1);
    let z2 = pcu::matmul(&h1, w2);
    let h2 = pcu::relu(&z2);
    let prediction = pcu::matmul(&h2, w3);
    let loss = pcu::mean_squared_error(&prediction, targets);
    let (g1, g2, g3) = pcu::gradients(&loss, (w1, w2, w3));
    let updated_w1 = pcu::sgd_update(w1, &g1, 0.000_001_f32);
    let updated_w2 = pcu::sgd_update(w2, &g2, 0.000_001_f32);
    let updated_w3 = pcu::sgd_update(w3, &g3, 0.000_001_f32);
    (updated_w1, updated_w2, updated_w3, loss)
}
