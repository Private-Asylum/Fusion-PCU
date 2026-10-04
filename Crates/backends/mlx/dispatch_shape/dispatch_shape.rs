//! Cold bounded borrowed-IR query; inspects immediate children without recursive scanners.
use fusion_pcu::{PcuDispatchKernelIr, PcuDispatchOp};
use crate::MlxError;
pub fn require_non_nested(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), MlxError> {
    if kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::GridStrideLoop { body, .. } => body
            .iter()
            .any(|child| matches!(child, PcuDispatchOp::GridStrideLoop { .. })),
        _ => false,
    }) {
        return Err(MlxError::InvalidRequest("nested borrowed grid body".into()));
    }
    Ok(())
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
