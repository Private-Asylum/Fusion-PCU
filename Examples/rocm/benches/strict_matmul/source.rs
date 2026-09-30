//! Executed annotated source supplies the compound operation in every source measurement.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
    pcu,
};

#[pcu(flag(strict))]
pub fn product<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(non_strict))]
pub fn boundary<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu]
pub fn identity<T: PcuScalar, const R: usize, const C: usize>(
    input: &[[T; C]; R],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[pcu(flag(strict), flag(reject_subnormal_result))]
pub fn reject_subnormal<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(strict), flag(allow_gradual_underflow))]
pub fn gradual<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}
