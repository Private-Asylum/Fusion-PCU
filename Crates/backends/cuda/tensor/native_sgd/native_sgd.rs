//! Explicit native F32/F64 optimizer policy; ordinary scalar arithmetic remains checked.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalRequirement,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    TensorUnsupportedReason,
};

pub(super) fn assess_graph(
    graph: &Graph,
    node: NodeDescriptor<'_>,
) -> Result<(), TensorUnsupportedReason> {
    let OpDescriptor::SgdUpdate { learning_rate, .. } = node.op else {
        return Err(TensorUnsupportedReason::Operation);
    };
    let original = graph
        .node(node.value)
        .map_err(|_| TensorUnsupportedReason::Operation)?;
    let OpDescriptor::SgdUpdate {
        learning_rate: original_rate,
        ..
    } = original.op
    else {
        // A synthetic rewrite of ordinary checked Mul/Sub has no native-compound permission.
        return Err(TensorUnsupportedReason::Operation);
    };
    // Floating descriptor equality aliases +0 and -0, but the frozen parameter ABI does not.
    if original != node || original_rate.to_bits() != learning_rate.to_bits() {
        return Err(TensorUnsupportedReason::Operation);
    }
    assess(node)
}

pub(super) fn assess(node: NodeDescriptor<'_>) -> Result<(), TensorUnsupportedReason> {
    let unsupported = |requirement| TensorUnsupportedReason::NumericalPolicy {
        requirement,
        options: node.numerical_options,
    };
    if node.numerical_options.reproducibility == PcuReproducibility::PortableV1 {
        return Err(unsupported(PcuNumericalRequirement::Reproducibility));
    }
    if node.numerical_mode != Some(PcuNumericalMode::Boundary)
        || node.numerical_options.compound_arithmetic != PcuCompoundArithmeticPolicy::BackendDefined
    {
        return Err(unsupported(PcuNumericalRequirement::CompoundArithmetic));
    }
    if let Some(policy) = node.float_underflow_policy
        && policy != PcuFloatUnderflowPolicy::IeeeAfterRounding
    {
        return Err(TensorUnsupportedReason::UnderflowPolicy(policy));
    }
    if !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
        return Err(TensorUnsupportedReason::ElementType);
    }
    if !matches!(node.op, OpDescriptor::SgdUpdate { learning_rate, .. } if learning_rate.is_finite())
    {
        return Err(TensorUnsupportedReason::Other(
            "native SGD requires a finite frozen F32 learning rate".into(),
        ));
    }
    // This private CUDA kernel has no BLAS emulation/TF32 path. Vendor-library environment
    // overrides are irrelevant and must not disqualify its preserved F32 arithmetic.
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

const SGD_UPDATE_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Explicit native Preserve offer: two destination-width rounded operations.
        const float product = __fmul_rn(learning_rate, gradient[id]);
        output[id] = __fsub_rn(weights[id], product);
    }
}
"#;
const SGD_UPDATE_CONTRACTED_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update_contracted(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __fmaf_rn(-learning_rate, gradient[id], weights[id]);
}
"#;

const SGD_UPDATE_F64_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update(
    const double *weights, const double *gradient, double *output,
    double learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Explicit native Preserve offer: two destination-width rounded operations.
        const double product = __dmul_rn(learning_rate, gradient[id]);
        output[id] = __dsub_rn(weights[id], product);
    }
}
"#;
const SGD_UPDATE_F64_CONTRACTED_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update_contracted(
    const double *weights, const double *gradient, double *output,
    double learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __fma_rn(-learning_rate, gradient[id], weights[id]);
}
"#;

/// Scalar width and permitted contraction are immutable native kernel cache identities.
/// A finite F32 rate widens exactly for F64, retaining signed zero; it is never rounded
/// from a wider source parameter. IEEE 754-2019 5.4.1 separates destination rounding
/// from the fused operation in 5.4.1: native exception permission is a PCU policy.
pub(super) fn source(scalar: PcuScalarType, contracted: bool) -> &'static str {
    match (scalar, contracted) {
        (PcuScalarType::F32, false) => SGD_UPDATE_SOURCE,
        (PcuScalarType::F32, true) => SGD_UPDATE_CONTRACTED_SOURCE,
        (PcuScalarType::F64, false) => SGD_UPDATE_F64_SOURCE,
        (PcuScalarType::F64, true) => SGD_UPDATE_F64_CONTRACTED_SOURCE,
        _ => unreachable!("native SGD assessment verifies F32/F64"),
    }
}

/// Generate an authentic explicit-native optimizer kernel for an independent control.
///
/// ABI: three F32 or F64 storage pointers, a rate in the same storage width, then
/// U32 count. The finite frozen F32 graph rate widens exactly for F64; preserve
/// its signed zero when constructing the rate bytes. No numerical status is used.
///
/// # Errors
/// Rejects unsupported numerical policy, scalar width, shape or node provenance.
pub fn lower_native_sgd_to_cuda_source(
    graph: &Graph,
    value: fusion_pcu::dialect::tensor::ValueId,
) -> Result<String, super::CudaTensorExecutionError> {
    let node = graph.node(value)?;
    assess_graph(graph, node)
        .map_err(|reason| super::CudaTensorExecutionError::Unsupported { value, reason })?;
    if let fusion_pcu::dialect::tensor::TensorOperationSupport::Unsupported { reason } =
        super::assess_tensor_node(graph, node)
    {
        return Err(super::CudaTensorExecutionError::Unsupported { value, reason });
    }
    Ok(source(
        node.scalar_type,
        node.numerical_options.precision == fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
    )
    .to_owned())
}
