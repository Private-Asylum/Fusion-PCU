//! Genuine owned staging seed and inherited exact annotated unary workloads.
#[rustfmt::skip]
use pcu_facade::{pcu,PcuScalar,PcuTensor,PcuExecutionError};
#[pcu(crate_path=::pcu_facade)]
pub fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}
#[path = "../../checked_unary/source/source.rs"]
mod unary;
pub use unary::*;
