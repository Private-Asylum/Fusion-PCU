//! Immutable IEEE 754 binary128/binary256 encoding transport, without arithmetic.
//!
//! Canonical little-endian limbs preserve signed zero, subnormal and NaN payloads.
//! No host floating-point conversion or device numerical evaluation occurs here.
#[rustfmt::skip]
use fusion_pcu::{
    PcuF128Bits,
    PcuF256Bits,
    PcuScalarType,
};
#[rustfmt::skip]
use super::super::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    RocmPhysicalLayout,
    RocmTensorExecutionError,
    TensorExecutionRoute,
    TensorOperationSupport,
    TensorUnsupportedReason,
};

pub(in super::super) fn encoded(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Vec<u8>, RocmTensorExecutionError> {
    match node.scalar_type {
        PcuScalarType::F128 => super::encoded_typed::<PcuF128Bits>(node, layout),
        PcuScalarType::F256 => super::encoded_typed::<PcuF256Bits>(node, layout),
        _ => Err(RocmTensorExecutionError::InvalidPlan(node.value)),
    }
}

pub(in super::super) fn assess(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    if !super::super::is_raw_float_type(node.scalar_type)
        || !matches!(
            node.op,
            OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        )
    {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::ElementType,
        };
    }
    let count = node
        .shape
        .iter()
        .try_fold(1_usize, |count, &extent| count.checked_mul(extent));
    if !count.is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        };
    }
    if graph.node(node.value).ok() != Some(node) {
        return TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation,
        };
    }
    TensorOperationSupport::Supported {
        route: TensorExecutionRoute::Native,
        workspace_bytes: Some(0),
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
