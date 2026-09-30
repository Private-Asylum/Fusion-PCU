//! Executed annotated source supplies the compound operation in every source measurement.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionError,
    PcuScalar,
    PcuTensor,
    pcu,
};

#[pcu(flag(non_strict), flag(native_compound))]
pub fn product<T: PcuScalar, const R: usize, const K: usize, const C: usize>(
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
