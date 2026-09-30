//! Cold completion selection for owned native matrix chains.
//!
//! Admission has already validated every descriptor and storage requirement. This selector
//! grants no numerical permission: it only removes intermediate host waits where stream order
//! and the existing terminal batch's retained allocation/library leases suffice.

#[rustfmt::skip]
use super::{
    NodeDescriptor,
    OpDescriptor,
    ValueId,
};

pub(super) fn batch_native_matmuls(nodes: &[NodeDescriptor<'_>], outputs: &[ValueId]) -> bool {
    let mut has_matmul = false;
    for node in nodes {
        match node.op {
            OpDescriptor::Input => {}
            OpDescriptor::MatMul { .. }
                if matches!(
                    node.scalar_type,
                    fusion_pcu::PcuScalarType::F32 | fusion_pcu::PcuScalarType::F64
                ) && node.numerical_mode == Some(fusion_pcu::PcuNumericalMode::Boundary)
                    && node.numerical_options.compound_arithmetic
                        == fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined
                    && node.numerical_options.reproducibility
                        == fusion_pcu::PcuReproducibility::Unspecified
                    && node.float_underflow_policy
                        != Some(fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult) =>
            {
                has_matmul = true;
            }
            _ => return false,
        }
    }
    // Copying a selected input output is a host boundary in the current scheduler, so leave
    // that mixed-output profile unchanged. Even one native GEMM benefits from its stream-local
    // final event rather than synchronizing unrelated work on the entire device.
    has_matmul
        && !outputs.is_empty()
        && outputs.iter().all(|output| {
            nodes
                .iter()
                .any(|node| node.value == *output && matches!(node.op, OpDescriptor::MatMul { .. }))
        })
}

#[cfg(test)]
#[path = "hardware.rs"]
mod hardware;
#[cfg(test)]
#[path = "tests.rs"]
mod tests;
