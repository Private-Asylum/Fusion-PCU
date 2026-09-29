//! Explicit f32 bindings for the heterogeneous CPU reference evaluator.

#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Tensor,
    TensorValue,
    ValueId,
};

/// Builds enum-backed reference bindings without changing backend-facing host tensor slices.
pub fn f32_inputs(inputs: &[(ValueId, Tensor)]) -> Vec<(ValueId, TensorValue)> {
    inputs
        .iter()
        .map(|(value, tensor)| (*value, TensorValue::F32(tensor.clone())))
        .collect()
}
