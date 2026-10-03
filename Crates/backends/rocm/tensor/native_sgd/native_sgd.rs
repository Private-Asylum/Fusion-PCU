//! Explicit native F32/F64 SGD. Rounded product/subtraction and permitted FMA are distinct kernels.
//! Native arithmetic permits backend special-value and subnormal behavior; scalar IR stays checked.

#[rustfmt::skip]
use super::{
    Graph,
    HipCompletionBatch,
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    NodeDescriptor,
    OpDescriptor,
    RocmMemoryResource,
    RocmTensorAssessor,
    RocmTensorExecutionError,
    TensorExecutionRoute,
    TensorOperationSupport,
    TensorUnsupportedReason,
};
use std::cell::RefCell;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuNumericalMode,
    PcuNumericalRequirement,
    PcuReproducibility,
    PcuScalarType,
};

#[derive(Default)]
pub(super) struct NativeSgdKernels {
    preserved: [RefCell<Option<(HipKernel, HipStreamHandle)>>; 2],
    contracted: [RefCell<Option<(HipKernel, HipStreamHandle)>>; 2],
}

#[derive(Clone, Copy)]
pub(super) struct SgdUpdateMode {
    pub(super) scalar: PcuScalarType,
    pub(super) learning_rate: f32,
    pub(super) contracted: bool,
}

pub(super) fn assess(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    let unsupported = |reason| TensorOperationSupport::Unsupported { reason };
    let OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate,
    } = node.op
    else {
        return unsupported(TensorUnsupportedReason::Operation);
    };
    let Ok(original) = graph.node(node.value) else {
        return unsupported(TensorUnsupportedReason::Operation);
    };
    let OpDescriptor::SgdUpdate {
        learning_rate: original_rate,
        ..
    } = original.op
    else {
        return unsupported(TensorUnsupportedReason::Operation);
    };
    // Descriptor equality alone considers +0 and -0 equal; the immutable parameter ABI does not.
    if original != node
        || original_rate.to_bits() != learning_rate.to_bits()
        || !learning_rate.is_finite()
    {
        return unsupported(TensorUnsupportedReason::Operation);
    }
    if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
        return unsupported(TensorUnsupportedReason::NumericalPolicy {
            requirement: PcuNumericalRequirement::Reproducibility,
            options: node.numerical_options,
        });
    }
    if node.numerical_mode != Some(PcuNumericalMode::Boundary)
        || node.numerical_options.compound_arithmetic != PcuCompoundArithmeticPolicy::BackendDefined
    {
        return unsupported(TensorUnsupportedReason::NumericalPolicy {
            requirement: PcuNumericalRequirement::CompoundArithmetic,
            options: node.numerical_options,
        });
    }
    if let Err(reason) = super::native_policy::assess_underflow(node.float_underflow_policy) {
        return unsupported(reason);
    }
    if !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64)
        || graph
            .node(weights)
            .map_or(true, |input| input.scalar_type != node.scalar_type)
        || graph
            .node(gradient)
            .map_or(true, |input| input.scalar_type != node.scalar_type)
    {
        return unsupported(TensorUnsupportedReason::ElementType);
    }
    match super::assess_dense_binary_shape(graph, node.shape, weights, gradient) {
        TensorOperationSupport::Supported { .. } => TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Native,
            workspace_bytes: Some(0),
        },
        unsupported @ TensorOperationSupport::Unsupported { .. } => unsupported,
    }
}

const SGD_UPDATE_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Preserve the F32 product before subtraction under explicit native permission.
        // No checked fault contract or status buffer is requested.
        volatile float product = learning_rate * gradient[id];
        output[id] = weights[id] - product;
    }
}
"#;
const SGD_UPDATE_CONTRACTED_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update_contracted(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __builtin_fmaf(-learning_rate, gradient[id], weights[id]);
}
"#;

const SGD_UPDATE_F64_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update(
    const double *weights, const double *gradient, double *output,
    double learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Preserve the F64 product before subtraction under explicit native permission.
        // No checked fault contract or status buffer is requested.
        volatile double product = learning_rate * gradient[id];
        output[id] = weights[id] - product;
    }
}
"#;
const SGD_UPDATE_F64_CONTRACTED_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update_contracted(
    const double *weights, const double *gradient, double *output,
    double learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = __builtin_fma(-learning_rate, gradient[id], weights[id]);
}
"#;

impl RocmTensorAssessor<'_> {
    pub(super) fn prepare_native_sgd(
        &self,
        scalar: PcuScalarType,
        contracted: bool,
    ) -> Result<(), RocmTensorExecutionError> {
        let cache = if contracted {
            &self.state().native_sgd.contracted[usize::from(scalar == PcuScalarType::F64)]
        } else {
            &self.state().native_sgd.preserved[usize::from(scalar == PcuScalarType::F64)]
        };
        if cache.borrow().is_none() {
            let runtime = self.session.tensor_runtime();
            let image = self
                .session
                .compile_tensor_source(source(scalar, contracted))
                .map_err(RocmTensorExecutionError::Backend)?;
            let module = runtime
                .load_module(&image)
                .map_err(RocmTensorExecutionError::Completion)?;
            let kernel = module
                .function(if contracted {
                    c"tensor_sgd_update_contracted"
                } else {
                    c"tensor_sgd_update"
                })
                .map_err(RocmTensorExecutionError::Completion)?;
            let stream = self.state().stream.clone();
            *cache.borrow_mut() = Some((kernel, stream));
        }
        Ok(())
    }

    pub(super) fn execute_sgd_update(
        &self,
        shape: &[usize],
        weights: &RocmMemoryResource,
        gradient: &RocmMemoryResource,
        mode: SgdUpdateMode,
        output: &RocmMemoryResource,
        batch: Option<&mut HipCompletionBatch>,
    ) -> Result<(), RocmTensorExecutionError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count = u32::try_from(count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let cache = if mode.contracted {
            &self.state().native_sgd.contracted[usize::from(mode.scalar == PcuScalarType::F64)]
        } else {
            &self.state().native_sgd.preserved[usize::from(mode.scalar == PcuScalarType::F64)]
        };
        let cached = cache.borrow();
        let (kernel, stream) = cached
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let rate_f32_bytes = mode.learning_rate.to_ne_bytes();
        let rate_f64_bytes = fusion_pcu::widen_f32_exact(mode.learning_rate).to_ne_bytes();
        let learning_rate_bytes = if mode.scalar == PcuScalarType::F64 {
            rate_f64_bytes.as_slice()
        } else {
            rate_f32_bytes.as_slice()
        };
        let count_bytes = count.to_ne_bytes();
        let args = [
            HipKernelArgument::Buffer(weights.device_buffer()),
            HipKernelArgument::Buffer(gradient.device_buffer()),
            HipKernelArgument::Buffer(output.device_buffer()),
            HipKernelArgument::Bytes(learning_rate_bytes),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        if let Some(batch) = batch {
            // SAFETY: the assessed shapes and live resources cover every guarded index. The
            // batch retains the launch owners before enqueue and its final event proves
            // quiescence before dependent work or storage reuse.
            #[allow(unsafe_code)]
            unsafe {
                kernel.launch_into_batch(batch, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args)
            }
            .map_err(RocmTensorExecutionError::Completion)
        } else {
            // SAFETY: the same assessed shape and resource contract applies to the direct path.
            #[allow(unsafe_code)]
            let mut completion = unsafe {
                kernel.launch(stream, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args)
            }
            .map_err(RocmTensorExecutionError::Completion)?;
            completion
                .wait()
                .map_err(RocmTensorExecutionError::Completion)
        }
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

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
pub fn lower_native_sgd_to_hip_source(
    graph: &Graph,
    value: fusion_pcu::dialect::tensor::ValueId,
) -> Result<String, super::RocmTensorExecutionError> {
    let node = graph.node(value)?;
    if let TensorOperationSupport::Unsupported { reason } = assess(graph, node) {
        return Err(RocmTensorExecutionError::Unsupported { value, reason });
    }
    Ok(source(
        node.scalar_type,
        node.numerical_options.precision == fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
    )
    .to_owned())
}
