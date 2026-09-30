//! Public source permission combinations are exercised independently of the diagnostic graph.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
    pcu,
};

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(preserve_precision),
    flag(non_deterministic)
)]
pub fn preserve<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(
    flag(non_strict),
    flag(native_compound),
    flag(backend_precision),
    flag(non_deterministic)
)]
pub fn optimized<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(non_strict), flag(checked_compound))]
pub fn checked_boundary<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(strict), flag(native_compound))]
pub fn strict_native<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(non_strict), flag(native_compound), flag(deterministic))]
pub fn portable_native<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

#[pcu(flag(non_strict), flag(native_compound), flag(reject_subnormal_result))]
pub fn tight_native<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::matmul(left, right)
}

pub fn product<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
    left: &[[T; K]; R],
    right: &[[T; C]; K],
    optimized_precision: bool,
) -> Result<PcuTensor<T>, PcuExecutionError> {
    if optimized_precision {
        optimized(left, right)
    } else {
        preserve(left, right)
    }
}
