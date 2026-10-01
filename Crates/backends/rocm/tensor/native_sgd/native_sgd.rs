//! Explicit native F32 SGD. Rounded product/subtraction and permitted FMA are distinct kernels.
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
    preserved: RefCell<Option<(HipKernel, HipStreamHandle)>>,
    contracted: RefCell<Option<(HipKernel, HipStreamHandle)>>,
}

#[derive(Clone, Copy)]
pub(super) struct SgdUpdateMode {
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
    if node.scalar_type != PcuScalarType::F32
        || graph
            .node(weights)
            .map_or(true, |input| input.scalar_type != PcuScalarType::F32)
        || graph
            .node(gradient)
            .map_or(true, |input| input.scalar_type != PcuScalarType::F32)
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

impl RocmTensorAssessor<'_> {
    pub(super) fn prepare_native_sgd(
        &self,
        contracted: bool,
    ) -> Result<(), RocmTensorExecutionError> {
        let cache = if contracted {
            &self.state().native_sgd.contracted
        } else {
            &self.state().native_sgd.preserved
        };
        if cache.borrow().is_none() {
            let runtime = self.session.tensor_runtime();
            let image = self
                .session
                .compile_tensor_source(if contracted {
                    SGD_UPDATE_CONTRACTED_SOURCE
                } else {
                    SGD_UPDATE_SOURCE
                })
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
            &self.state().native_sgd.contracted
        } else {
            &self.state().native_sgd.preserved
        };
        let cached = cache.borrow();
        let (kernel, stream) = cached
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let learning_rate_bytes = mode.learning_rate.to_ne_bytes();
        let count_bytes = count.to_ne_bytes();
        let args = [
            HipKernelArgument::Buffer(weights.device_buffer()),
            HipKernelArgument::Buffer(gradient.device_buffer()),
            HipKernelArgument::Buffer(output.device_buffer()),
            HipKernelArgument::Bytes(&learning_rate_bytes),
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
