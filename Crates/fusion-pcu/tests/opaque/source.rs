//! Real ordinary source workloads shared by tests, examples and matched benchmarks.
#[rustfmt::skip]
use fusion_pcu::{
    pcu,
    PcuExecutionError,
    PcuTensor,
};

#[pcu]
pub fn identity_matrix<const R: usize, const C: usize>(
    input: &[[f32; C]; R],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
pub fn consume_identity(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu(flag(native_compound), flag(backend_precision))]
pub fn matrix<const R: usize, const K: usize, const C: usize>(
    left: &[[f32; K]; R],
    right: &[[f32; C]; K],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(native_compound), flag(backend_precision))]
pub fn consume_matrix(
    left: PcuTensor<f32>,
    right: &[[f32; 2]; 2],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(strict))]
pub fn checked_matrix<const R: usize, const K: usize, const C: usize>(
    left: &[[f32; K]; R],
    right: &[[f32; C]; K],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(native_compound), flag(backend_precision), flag(deterministic))]
pub fn portable_matrix<const R: usize, const K: usize, const C: usize>(
    left: &[[f32; K]; R],
    right: &[[f32; C]; K],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::matmul(left, right)
}
