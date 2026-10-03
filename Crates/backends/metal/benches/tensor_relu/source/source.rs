//! Actual ordinary source definition; benchmark qualifies both ordinary routing and validated cold capture.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuScalar,PcuTensor,PcuExecutionError};
#[pcu(crate_path=::pcu_facade)]
pub fn relu<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu(input)
}
