//! Explicit native F32 loss arithmetic; ordinary scalar IR keeps its checked contract.
//!
//! A native MSE rounds the F32 difference and square, reduces squared magnitudes through
//! rocBLAS ASUM, then scales by a rounded F32 reciprocal count. Reduction order, subnormal
//! handling and special-value propagation are backend-defined under explicit permission.
//! This is not IEEE exception classification or a portable reproducibility profile.
//! rocBLAS ASUM and device-result pointer mode are documented at:
//! <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-1.html>

#[rustfmt::skip]
use super::{
    Graph,
    HipKernelArgument,
    NodeDescriptor,
    OpDescriptor,
    RocmMemoryResource,
    RocmTensorAssessor,
    RocmTensorExecutionError,
    TensorExecutionRoute,
    TensorOperationSupport,
    TensorUnsupportedReason,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalRequirement,
    PcuReproducibility,
    PcuScalarType,
};

const SOURCE: &str = r#"
extern "C" __global__ void tensor_native_mse_squared_difference(
    const float *prediction, const float *target, float *squared, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        // Preserve a F32 difference before the F32 square. No checked status is requested.
        volatile float difference = prediction[id] - target[id];
        squared[id] = difference * difference;
    }
}
"#;

pub(super) fn assess(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    let unsupported = |reason| TensorOperationSupport::Unsupported { reason };
    if graph.node(node.value).ok() != Some(node) {
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
    if node.float_underflow_policy == Some(PcuFloatUnderflowPolicy::RejectSubnormalResult) {
        return unsupported(TensorUnsupportedReason::UnderflowPolicy(
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ));
    }
    if node.scalar_type != PcuScalarType::F32 {
        return unsupported(TensorUnsupportedReason::ElementType);
    }
    let OpDescriptor::MeanSquaredError { prediction, target } = node.op else {
        return unsupported(TensorUnsupportedReason::Operation);
    };
    let Ok(prediction_node) = graph.node(prediction) else {
        return unsupported(TensorUnsupportedReason::Shape);
    };
    let Ok(target_node) = graph.node(target) else {
        return unsupported(TensorUnsupportedReason::Shape);
    };
    if prediction_node.scalar_type != PcuScalarType::F32
        || target_node.scalar_type != PcuScalarType::F32
    {
        return unsupported(TensorUnsupportedReason::ElementType);
    }
    if prediction_node.shape != target_node.shape || !node.shape.is_empty() {
        return unsupported(TensorUnsupportedReason::Shape);
    }
    let Some(workspace_bytes) = prediction_node
        .shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .filter(|count| *count > 0 && i32::try_from(*count).is_ok())
        .and_then(|count| count.checked_mul(size_of::<f32>()))
    else {
        return unsupported(TensorUnsupportedReason::Shape);
    };
    // Preserve uses F32 storage/arithmetic; BackendOptimized permission may select this stronger
    // implementation. No reduced storage type, approximation or hidden fault scan is added.
    TensorOperationSupport::Supported {
        route: TensorExecutionRoute::Library,
        workspace_bytes: Some(workspace_bytes),
    }
}

impl RocmTensorAssessor<'_> {
    pub(super) fn prepare_native_mse(&self) -> Result<(), RocmTensorExecutionError> {
        if self.state().native_mse.borrow().is_some() {
            return Ok(());
        }
        let image = self
            .session
            .compile_tensor_source(SOURCE)
            .map_err(RocmTensorExecutionError::Backend)?;
        let module = self
            .session
            .tensor_runtime()
            .load_module(&image)
            .map_err(RocmTensorExecutionError::Completion)?;
        let kernel = module
            .function(c"tensor_native_mse_squared_difference")
            .map_err(RocmTensorExecutionError::Completion)?;
        *self.state().native_mse.borrow_mut() = Some(kernel);
        Ok(())
    }

    pub(super) fn execute_native_mse_squared(
        &self,
        count: u32,
        prediction: &RocmMemoryResource,
        target: &RocmMemoryResource,
        squared: &RocmMemoryResource,
    ) -> Result<(), RocmTensorExecutionError> {
        let cached = self.state().native_mse.borrow();
        let kernel = cached
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count_bytes = count.to_ne_bytes();
        let arguments = [
            HipKernelArgument::Buffer(prediction.device_buffer()),
            HipKernelArgument::Buffer(target.device_buffer()),
            HipKernelArgument::Buffer(squared.device_buffer()),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: cold admission and scheduler binding checks prove F32 extents for all guarded
        // indices. The synchronous completion retains every launch owner through terminal
        // quiescence or quarantine before ASUM may read the private squared scratch buffer.
        #[allow(unsafe_code)]
        let mut completion = unsafe {
            kernel.launch(
                &self.state().stream,
                [count.div_ceil(256), 1, 1],
                [256, 1, 1],
                0,
                &arguments,
            )
        }
        .map_err(RocmTensorExecutionError::Completion)?;
        completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
