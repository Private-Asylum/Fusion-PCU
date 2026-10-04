//! Cold bounded borrowed-IR query; inspects immediate children without recursive scanners.
use fusion_pcu::{PcuDispatchKernelIr, PcuDispatchOp};
use crate::MetalError;
pub fn require_non_nested(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), MetalError> {
    if kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::GridStrideLoop { body, .. } => body
            .iter()
            .any(|child| matches!(child, PcuDispatchOp::GridStrideLoop { .. })),
        _ => false,
    }) {
        return Err(MetalError::Unsupported);
    }
    Ok(())
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
