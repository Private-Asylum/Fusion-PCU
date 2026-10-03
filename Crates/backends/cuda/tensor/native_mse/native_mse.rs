//! Explicitly native F32/F64 squared differences; reduction uses the selected cuBLAS contract.
//!
//! This private source never grants ordinary scalar IR unchecked arithmetic. The tensor assessor
//! admits only Boundary + `BackendDefined` MSE and owns this exact source/storage ABI together.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuOwnedBindingRequirement,
    PcuScalarType,
    PcuValueType,
};

pub fn requirements(count: u32, scalar: PcuScalarType) -> Vec<PcuOwnedBindingRequirement> {
    (0..3)
        .map(|slot| PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, slot),
            access: if slot == 2 {
                PcuBindingAccess::WriteOnly
            } else {
                PcuBindingAccess::ReadOnly
            },
            binding_type: PcuBindingType::Value(match scalar {
                PcuScalarType::F32 => PcuValueType::f32(),
                PcuScalarType::F64 => PcuValueType::f64(),
                _ => unreachable!("native MSE assessment verifies F32/F64"),
            }),
            min_required_bytes: u64::from(count) * u64::from(scalar.bit_width() / 8),
        })
        .collect()
}

pub fn source(count: u32, scalar: PcuScalarType) -> String {
    // CUDA's explicit rounding intrinsics preserve two destination-width operations. No fused
    // multiply-add, checked scalar permission or portable reduction order is inferred here.
    let (storage, sub, mul) = match scalar {
        PcuScalarType::F32 => ("float", "__fsub_rn", "__fmul_rn"),
        PcuScalarType::F64 => ("double", "__dsub_rn", "__dmul_rn"),
        _ => unreachable!("native MSE assessment verifies F32/F64"),
    };
    format!(
        r#"extern "C" __global__ void fusion_kernel(const {storage}* prediction, const {storage}* target, {storage}* squared) {{
    const unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= {count}ull) return;
    const {storage} difference = {sub}(prediction[id], target[id]);
    squared[id] = {mul}(difference, difference);
}}
"#
    )
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

/// Generate the authentic native squared-difference stage for an independent control.
///
/// ABI: three F32/F64 pointers covering the frozen prediction extent; no count argument
/// or numerical status. Follow with same-width native ASUM and a same-width reciprocal
/// scale on the same stream. IEEE 754-2019 5.4.1 specifies the separate rounded Sub/Mul;
/// the library reduction order and numerical exceptions require explicit native permission.
///
/// # Errors
/// Rejects checked boundary, strict, reproducibility, underflow, shape or scalar profiles.
pub fn lower_native_mse_to_cuda_source(
    graph: &fusion_pcu::dialect::tensor::Graph,
    value: fusion_pcu::dialect::tensor::ValueId,
) -> Result<String, super::CudaTensorExecutionError> {
    let node = graph.node(value)?;
    super::assess_native_mse_numerical_options(node, &crate::CublasEnvironmentSnapshot::capture())
        .map_err(|reason| super::CudaTensorExecutionError::Unsupported { value, reason })?;
    if let fusion_pcu::dialect::tensor::TensorOperationSupport::Unsupported { reason } =
        super::assess_tensor_node(graph, node)
    {
        return Err(super::CudaTensorExecutionError::Unsupported { value, reason });
    }
    let fusion_pcu::dialect::tensor::OpDescriptor::MeanSquaredError { prediction, .. } = node.op
    else {
        return Err(super::CudaTensorExecutionError::InvalidPlan(value));
    };
    let count = super::flattened_invocation_count(graph.shape(prediction)?)?;
    Ok(source(count, node.scalar_type))
}

#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorOperationSupport,
    TensorExecutionRoute,
    TensorUnsupportedReason,
};

pub(super) fn assess_graph(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    let unsupported = |reason| TensorOperationSupport::Unsupported { reason };
    if graph.node(node.value).ok() != Some(node) {
        return unsupported(TensorUnsupportedReason::Operation);
    }
    let OpDescriptor::MeanSquaredError { prediction, target } = node.op else {
        return unsupported(TensorUnsupportedReason::Operation);
    };
    let (Ok(left), Ok(right)) = (graph.node(prediction), graph.node(target)) else {
        return unsupported(TensorUnsupportedReason::Shape);
    };
    if left.scalar_type != node.scalar_type || right.scalar_type != node.scalar_type {
        return unsupported(TensorUnsupportedReason::ElementType);
    }
    if left.shape != right.shape || !node.shape.is_empty() {
        return unsupported(TensorUnsupportedReason::Shape);
    }
    let Some(workspace_bytes) = left
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .filter(|&n| n > 0 && i32::try_from(n).is_ok())
        .and_then(|n| n.checked_mul(usize::from(node.scalar_type.bit_width() / 8)))
    else {
        return unsupported(TensorUnsupportedReason::Shape);
    };
    TensorOperationSupport::Supported {
        route: TensorExecutionRoute::Library,
        workspace_bytes: Some(workspace_bytes),
    }
}
