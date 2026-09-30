//! Actual source preserves a compile-time finite literal, independently of warm input slots.
#![allow(
    unused_parens,
    reason = "Actual source regression intentionally exercises grouped negative literals."
)]
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
    weights: &[f32; N],
    gradient: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, (-0.25_f32))
}
#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
pub fn optimized<const N: usize>(
    weights: &[f32; N],
    gradient: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, (-0.25_f32))
}
#[pcu(flag(non_strict), flag(native_compound), flag(non_deterministic))]
pub fn other_rate<const N: usize>(
    weights: &[f32; N],
    gradient: &[f32; N],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}
#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(preserve_precision),
    flag(non_deterministic)
)]
pub fn witness_preserve(
    weights: &[f32; 1],
    gradient: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}
#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
pub fn witness_optimized(
    weights: &[f32; 1],
    gradient: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 1.000_000_1_f32)
}
#[pcu]
pub fn checked(
    weights: &[f32; 1],
    gradient: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
#[pcu(flag(strict), flag(native_compound), flag(non_deterministic))]
pub fn strict(
    weights: &[f32; 1],
    gradient: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
#[pcu(flag(non_strict), flag(native_compound), flag(deterministic))]
pub fn portable(
    weights: &[f32; 1],
    gradient: &[f32; 1],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}
#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(non_deterministic),
    flag(reject_subnormal_result)
)]
pub fn tight(weights: &[f32; 1], gradient: &[f32; 1]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.25_f32)
}

pub fn update<const N: usize>(
    weights: &[f32; N],
    gradient: &[f32; N],
    optimized_precision: bool,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    if optimized_precision {
        optimized(weights, gradient)
    } else {
        preserve(weights, gradient)
    }
}
