//! Opt-in tensor operation assessment for an explicitly selected `ROCm` session.
//!
//! This adapter executes dense row-major f32 matrix multiplication through rocBLAS and elementwise
//! addition and `ReLU` through owned PCU Dispatch. Other tensor operations remain unsupported.

use std::{
    cell::RefCell,
    collections::{
        HashMap,
        VecDeque,
    },
    error::Error,
    fmt,
    mem::{
        align_of,
        size_of,
    },
    time::{
        Duration,
        Instant,
    },
};

use fusion_pcu::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuInvocationShape,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryResource,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryPoolId,
    PcuOwnedDispatchMemorySession,
    PcuOwnedCompletion,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
};
use std::num::NonZeroU32;
use fusion_pcu_tensor::{
    Graph,
    NodeDescriptor,
    OpDescriptor,
    Tensor,
    TensorError,
    TensorExecutionRoute,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorSynchronousF32MatMulBackend,
    TensorUnsupportedReason,
    ValueId,
};

use crate::{
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    Rocblas,
    RocblasError,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmPreparedDispatch,
};

const ADD_DISPATCH_CACHE_CAPACITY: usize = 8;
const RELU_BACKWARD_SOURCE: &str = r#"
extern "C" __global__ void tensor_relu_backward(
    const float *input, const float *upstream, float *output, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = input[id] > 0.0f ? upstream[id] : 0.0f;
}
"#;
const SGD_UPDATE_SOURCE: &str = r#"
extern "C" __global__ void tensor_sgd_update(
    const float *weights, const float *gradient, float *output,
    float learning_rate, unsigned int n) {
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) output[id] = weights[id] - learning_rate * gradient[id];
}
"#;

const ADD_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("left"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("right"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ),
];

const ADD_LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const ADD_RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const ADD_SUM: PcuDispatchValueId = PcuDispatchValueId(3);
const ADD_LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const ADD_RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const ADD_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);

const ADD_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_LEFT,
        binding: ADD_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_RIGHT,
        binding: ADD_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        result: ADD_SUM,
        op: fusion_pcu::PcuDispatchAluOp::Add,
        lhs: ADD_LEFT,
        rhs: ADD_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: ADD_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: ADD_SUM,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

const SUB_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_LEFT,
        binding: ADD_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_RIGHT,
        binding: ADD_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        result: ADD_SUM,
        op: fusion_pcu::PcuDispatchAluOp::Sub,
        lhs: ADD_LEFT,
        rhs: ADD_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: ADD_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: ADD_SUM,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

const MUL_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_LEFT,
        binding: ADD_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: ADD_RIGHT,
        binding: ADD_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        result: ADD_SUM,
        op: fusion_pcu::PcuDispatchAluOp::Mul,
        lhs: ADD_LEFT,
        rhs: ADD_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: ADD_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: ADD_SUM,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

const RELU_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ),
];

const RELU_INPUT: PcuDispatchValueId = PcuDispatchValueId(2);
const RELU_ZERO: PcuDispatchValueId = PcuDispatchValueId(1);
const RELU_RESULT: PcuDispatchValueId = PcuDispatchValueId(3);
const RELU_INPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const RELU_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const RELU_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
        result: RELU_ZERO,
        value: PcuParameterValue::F32(0.0_f32.to_bits()),
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: RELU_INPUT,
        binding: RELU_INPUT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        result: RELU_RESULT,
        op: fusion_pcu::PcuDispatchAluOp::Max,
        lhs: RELU_INPUT,
        rhs: RELU_ZERO,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: RELU_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: RELU_RESULT,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

const MSE_LEFT: PcuDispatchValueId = PcuDispatchValueId(1);
const MSE_RIGHT: PcuDispatchValueId = PcuDispatchValueId(2);
const MSE_DIFF: PcuDispatchValueId = PcuDispatchValueId(3);
const MSE_SQUARE: PcuDispatchValueId = PcuDispatchValueId(4);
const MSE_LEFT_REF: PcuBindingRef = PcuBindingRef::new(0, 0);
const MSE_RIGHT_REF: PcuBindingRef = PcuBindingRef::new(0, 1);
const MSE_OUTPUT_REF: PcuBindingRef = PcuBindingRef::new(0, 2);
const MSE_BINDINGS: &[PcuBinding<'static>] = &[
    PcuBinding::value(
        Some("prediction"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("target"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::f32(),
    ),
    PcuBinding::value(
        Some("squared_error"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::f32(),
    ),
];
const MSE_OPS: &[PcuDispatchOp<'static>] = &[
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: MSE_LEFT,
        binding: MSE_LEFT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: MSE_RIGHT,
        binding: MSE_RIGHT_REF,
        index: PcuDispatchIndex::InvocationId,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        result: MSE_DIFF,
        op: fusion_pcu::PcuDispatchAluOp::Sub,
        lhs: MSE_LEFT,
        rhs: MSE_RIGHT,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
        result: MSE_SQUARE,
        op: fusion_pcu::PcuDispatchAluOp::Mul,
        lhs: MSE_DIFF,
        rhs: MSE_DIFF,
    }),
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: MSE_OUTPUT_REF,
        index: PcuDispatchIndex::InvocationId,
        value: MSE_SQUARE,
    }),
    PcuDispatchOp::Control(PcuDispatchControlOp::Return),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TensorDispatchKind {
    Add,
    Sub,
    Mul,
    Relu,
    SquaredDifference,
}

impl TensorDispatchKind {
    fn kernel(self, logical_count: u32) -> PcuDispatchKernelIr<'static> {
        match self {
            Self::Add => add_kernel(logical_count),
            Self::Sub => sub_kernel(logical_count),
            Self::Mul => mul_kernel(logical_count),
            Self::Relu => relu_kernel(logical_count),
            Self::SquaredDifference => mse_kernel(logical_count),
        }
    }
}

fn mse_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0x4d53_4521),
        entry: PcuDispatchEntryPoint {
            name: "tensor_mse_squared_difference",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: MSE_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: MSE_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn add_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0x5445_4e53),
        entry: PcuDispatchEntryPoint {
            name: "tensor_add",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: ADD_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn sub_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0x5355_4221),
        entry: PcuDispatchEntryPoint {
            name: "tensor_sub",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: SUB_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn mul_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0x4d55_4c21),
        entry: PcuDispatchEntryPoint {
            name: "tensor_mul",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: ADD_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: MUL_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

fn relu_kernel(logical_count: u32) -> PcuDispatchKernelIr<'static> {
    PcuDispatchKernelIr {
        id: fusion_pcu::PcuKernelId(0x5245_4c55),
        entry: PcuDispatchEntryPoint {
            name: "tensor_relu",
            logical_shape: [logical_count, 1, 1],
        },
        bindings: RELU_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: RELU_OPS,
        type_caps: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        feature_caps: PcuDispatchFeatureCaps::default(),
    }
}

/// Errors from the bounded `ROCm` tensor `MatMul` operation.
#[derive(Debug)]
pub enum RocmTensorError {
    Rocblas(RocblasError),
    InvalidShape,
    InvalidMemoryAccess,
    DimensionOverflow,
}

impl fmt::Display for RocmTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Rocblas(error) => error.fmt(f),
            Self::InvalidShape => {
                f.write_str("ROCm tensor MatMul requires nonempty rank-two matrices")
            }
            Self::InvalidMemoryAccess => {
                f.write_str("ROCm tensor MatMul memory access does not match its role")
            }
            Self::DimensionOverflow => {
                f.write_str("ROCm tensor MatMul byte size or dimension overflows")
            }
        }
    }
}

impl Error for RocmTensorError {}

impl From<RocblasError> for RocmTensorError {
    fn from(error: RocblasError) -> Self {
        Self::Rocblas(error)
    }
}

/// Failure while executing a requested output from a graph on the selected `ROCm` session.
#[derive(Debug)]
pub enum RocmTensorExecutionError {
    Graph(TensorError),
    Unsupported {
        value: ValueId,
        reason: TensorUnsupportedReason,
    },
    Memory(PcuMemoryProviderError),
    Operation(RocmTensorError),
    Backend(crate::RocmOwnedDispatchError),
    Completion(crate::HipError),
    FailedCompletion,
    SizeOverflow,
    EmptyOutputs,
    DuplicateOutput(ValueId),
    InvalidPlan(ValueId),
    MissingResource(ValueId),
    InputResourceMismatch,
    InputPoolMismatch,
    OutputResourceMismatch,
    ScratchMismatch,
}

impl fmt::Display for RocmTensorExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Error for RocmTensorExecutionError {}

impl From<TensorError> for RocmTensorExecutionError {
    fn from(error: TensorError) -> Self {
        Self::Graph(error)
    }
}

impl From<PcuMemoryProviderError> for RocmTensorExecutionError {
    fn from(error: PcuMemoryProviderError) -> Self {
        Self::Memory(error)
    }
}

impl From<RocmTensorError> for RocmTensorExecutionError {
    fn from(error: RocmTensorError) -> Self {
        Self::Operation(error)
    }
}

/// Assessor and bounded tensor executor tied to an explicitly opened `ROCm` dispatch session.
///
/// Construction loads rocBLAS, creates a handle for the selected device, and verifies the SGEMM
/// entry point. Therefore a `MatMul` node is reported as `Library` only when the route is prepared.
pub struct RocmTensorAssessor<'session> {
    rocblas: Rocblas,
    session: &'session RocmOwnedDispatchBackend,
    add_dispatches: RefCell<VecDeque<((TensorDispatchKind, u32), RocmPreparedDispatch<'static>)>>,
    relu_backward: RefCell<Option<(HipKernel, HipStreamHandle)>>,
    sgd_update: RefCell<Option<(HipKernel, HipStreamHandle)>>,
}

/// Caller-owned device allocation reusable as an immutable graph input.
///
/// The session borrow prevents use after backend destruction; execution also checks backend and
/// pool identity before launching work. Contents change only through [`RocmTensorAssessor::update_input`].
pub struct RocmTensorInput<'session> {
    session: &'session RocmOwnedDispatchBackend,
    shape: Vec<usize>,
    resource: RocmMemoryResource,
}

impl RocmTensorInput<'_> {
    fn validate_session(
        &self,
        session: &RocmOwnedDispatchBackend,
    ) -> Result<(), RocmTensorExecutionError> {
        if !std::ptr::eq(self.session, session) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        if !self.resource.belongs_to_runtime(session.tensor_runtime()) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        Ok(())
    }

    fn resource_ref(&self) -> RocmMemoryResource {
        self.resource.clone_for_tensor_input()
    }
}

impl<'session> RocmTensorAssessor<'session> {
    /// Prepare rocBLAS for the backend's already selected device.
    ///
    /// # Errors
    ///
    /// Returns an error if rocBLAS, its SGEMM symbol, or the selected-device handle is unavailable.
    pub fn new(session: &'session RocmOwnedDispatchBackend) -> Result<Self, RocblasError> {
        Ok(Self {
            rocblas: Rocblas::new(session.tensor_runtime())?,
            session,
            add_dispatches: RefCell::new(VecDeque::new()),
            relu_backward: RefCell::new(None),
            sgd_update: RefCell::new(None),
        })
    }

    /// Executes the dependency closure of `output` using this selected `ROCm` tensor route.
    ///
    /// Inputs and constants use the supplied provider, dense `MatMul` nodes use rocBLAS, and
    /// same-shape nonempty `Add` and `ReLU` nodes use owned PCU Dispatch. Elementwise executables
    /// are cached by operation and flattened element count in a bounded per-assessor cache. The
    /// entire requested closure and its host inputs are validated before the first allocation. No
    /// reference or other-backend fallback is attempted.
    ///
    /// # Errors
    ///
    /// Returns a graph/input error, an explicit unsupported-node error, a memory-provider error,
    /// or a `ROCm` operation error.
    pub fn execute_graph<P>(
        &self,
        graph: &Graph,
        inputs: &[(ValueId, Tensor)],
        output: ValueId,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let prepared = self.prepare_graph(graph, output)?;
        self.execute_prepared(&prepared, inputs, pool, memory)
    }

    /// Prepare the requested output dependency closure for repeated execution.
    ///
    /// The returned value borrows `graph`, which must remain unchanged for its lifetime. It
    /// contains the selected node order and resource use counts; host input validation and device
    /// allocation are deferred until [`Self::execute_prepared`].
    ///
    /// # Errors
    ///
    /// Returns an error if the output is unknown, its dependency closure contains an operation
    /// this adapter cannot execute, or the selected node shapes cannot be represented.
    pub fn prepare_graph<'graph>(
        &self,
        graph: &'graph Graph,
        output: ValueId,
    ) -> Result<RocmPreparedTensorGraph<'graph>, RocmTensorExecutionError> {
        self.prepare_graph_outputs(graph, &[output])
    }

    /// Prepare the union of the dependency closures for requested outputs.
    ///
    /// Outputs retain the caller's order. Empty and duplicate lists are rejected.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or duplicate output list, unknown outputs, unsupported
    /// operations in any dependency closure, or unrepresentable node shapes.
    pub fn prepare_graph_outputs<'graph>(
        &self,
        graph: &'graph Graph,
        outputs: &[ValueId],
    ) -> Result<RocmPreparedTensorGraph<'graph>, RocmTensorExecutionError> {
        let plan = prepare_graph_outputs_plan(graph, outputs, self)?;
        Ok(RocmPreparedTensorGraph {
            graph,
            output: outputs[0],
            outputs: outputs.to_vec(),
            nodes: plan.nodes,
            index_by_value: plan.index_by_value,
            use_counts: plan.use_counts,
        })
    }

    /// Execute a previously prepared output using new host inputs.
    ///
    /// Structural graph analysis is not repeated. Inputs are checked before the first device
    /// allocation, and the selected rocBLAS handle is checked before execution begins. The first
    /// Add of a particular flattened extent prepares its PCU executable; later calls reuse it.
    ///
    /// # Errors
    ///
    /// Returns an input error, memory-provider error, or a `ROCm` operation error.
    #[allow(clippy::too_many_lines)] // Keeps preflighted graph scheduling and resource release visible together.
    pub fn execute_prepared<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let output = self.execute_prepared_resident(prepared, inputs, pool, memory)?;
        download_tensor(memory, &output)
    }

    /// Execute a prepared graph and retain its output in device memory.
    ///
    /// The returned resource is tied to this selected backend and can be supplied as an input to
    /// another prepared graph through [`Self::execute_prepared_with_resources_resident`]. This
    /// avoids both output readback and a subsequent host-to-device upload between graph stages.
    ///
    /// # Errors
    ///
    /// Returns an input error, memory-provider error, or a `ROCm` operation error.
    #[allow(clippy::too_many_lines)] // Keeps preflighted graph scheduling and resource release visible together.
    pub fn execute_prepared_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut timings = NoopNodeTiming;
        let mut outputs = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[],
            pool,
            memory,
            None,
            None,
            &mut timings,
        )?;
        Ok(outputs.remove(0))
    }

    /// Prepare persistent device storage for constants and non-output nodes in `prepared`.
    ///
    /// The scratch borrows both this assessor's selected session and the exact prepared plan.
    /// It is mutable during execution, which prevents overlapping use of its intermediate
    /// allocations. The requested output always gets a fresh allocation for each execution.
    ///
    /// # Errors
    ///
    /// Returns an allocation, upload, or selected-session resource mismatch error.
    pub fn prepare_scratch<'plan, 'graph, P>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorScratch<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut resources = Vec::with_capacity(prepared.nodes.len());
        let max_mse_count = mse_scratch_element_count(prepared.graph, &prepared.nodes)?;
        for node in &prepared.nodes {
            let resource = if scratch_stores_node(node.op, node.value, &prepared.outputs) {
                match node.op {
                    OpDescriptor::Constant(tensor) => Some(upload_tensor(memory, pool, tensor)?),
                    OpDescriptor::Input => None,
                    _ => Some(allocate_tensor(memory, pool, node.shape)?),
                }
            } else {
                None
            };
            if let Some(resource) = &resource
                && (resource.pool() != pool
                    || !resource.belongs_to_runtime(self.session.tensor_runtime()))
            {
                return Err(RocmTensorExecutionError::ScratchMismatch);
            }
            resources.push(resource);
        }
        let (mse_squared, mse_ones) = if max_mse_count == 0 {
            (None, None)
        } else {
            let squared = allocate_tensor(memory, pool, &[max_mse_count])?;
            let ones = upload_tensor(
                memory,
                pool,
                &Tensor::new(vec![max_mse_count], vec![1.0; max_mse_count])?,
            )?;
            (Some(squared), Some(ones))
        };
        Ok(RocmTensorScratch {
            prepared,
            session: self.session,
            pool,
            resources,
            mse_squared,
            mse_ones,
            poisoned: false,
        })
    }

    /// Allocate reusable caller-owned storage for every output in the prepared plan.
    ///
    /// # Errors
    ///
    /// Returns an allocation, shape, or selected-session resource mismatch error.
    pub fn prepare_output_bank<'plan, 'graph, P>(
        &'session self,
        prepared: &'plan RocmPreparedTensorGraph<'graph>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorOutputBank<'plan, 'graph, 'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut outputs = Vec::with_capacity(prepared.outputs.len());
        for &value in &prepared.outputs {
            let shape = prepared.graph.shape(value)?.to_vec();
            let resource = allocate_tensor(memory, pool, &shape)?;
            if resource.pool() != pool
                || !resource.belongs_to_runtime(self.session.tensor_runtime())
                || resource.device_buffer().len() < byte_len(&shape)?
                || !matches!(
                    resource.access(),
                    PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
                )
                || outputs.iter().any(|prior: &RocmTensorInput<'session>| {
                    prior.resource.same_allocation(&resource)
                })
            {
                return Err(RocmTensorExecutionError::OutputResourceMismatch);
            }
            outputs.push(RocmTensorInput {
                session: self.session,
                shape,
                resource,
            });
        }
        Ok(RocmTensorOutputBank {
            prepared,
            session: self.session,
            pool,
            outputs,
            poisoned: false,
        })
    }

    /// Execute into reusable output storage. Outputs remain in the bank in prepared output order.
    ///
    /// # Errors
    ///
    /// Returns a plan, session, pool, shape, or allocation-alias mismatch, or an execution error.
    pub fn execute_prepared_outputs_into_bank<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        output_bank: &mut RocmTensorOutputBank<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if output_bank.poisoned
            || !std::ptr::eq(output_bank.prepared, prepared)
            || !std::ptr::eq(output_bank.session, self.session)
            || output_bank.pool != scratch.pool
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            output_bank.pool,
            memory,
            Some(&mut *scratch),
            Some(output_bank),
            &mut NoopNodeTiming,
        );
        if result.is_err() {
            scratch.poisoned = true;
            output_bank.poisoned = true;
        }
        result.map(drop)
    }

    /// Execute a prepared graph with caller-owned reusable constants and intermediate storage.
    ///
    /// # Errors
    ///
    /// Returns an input or scratch identity error, memory-provider error, or operation error.
    pub fn execute_prepared_with_scratch_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, Tensor)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        if scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let mut timings = NoopNodeTiming;
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            inputs,
            &[],
            scratch.pool,
            memory,
            Some(&mut *scratch),
            None,
            &mut timings,
        );
        if result.is_err() {
            scratch.poisoned = true;
        }
        result.map(|mut outputs| outputs.remove(0))
    }

    /// Execute a prepared graph with reusable scratch and persistent device inputs.
    ///
    /// # Errors
    ///
    /// Returns an input or scratch identity error, memory-provider error, or operation error.
    pub fn execute_prepared_with_resources_and_scratch_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut outputs = self.execute_prepared_outputs_with_resources_and_scratch_resident(
            prepared, inputs, scratch, memory,
        )?;
        Ok(outputs.remove(0))
    }

    /// Execute a prepared multi-output graph with persistent inputs and reusable scratch.
    ///
    /// Returned outputs follow the order supplied to [`Self::prepare_graph_outputs`].
    ///
    /// # Errors
    ///
    /// Returns an input or scratch identity error, memory-provider error, or operation error.
    pub fn execute_prepared_outputs_with_resources_and_scratch_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<Vec<RocmTensorInput<'session>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        self.execute_prepared_outputs_with_resources_and_scratch_resident_timed(
            prepared,
            inputs,
            scratch,
            memory,
            &mut NoopNodeTiming,
        )
    }

    /// Diagnostic variant of [`Self::execute_prepared_outputs_with_resources_and_scratch_resident`]
    /// that also returns synchronous wall time for each prepared node in execution order.
    ///
    /// Timing collection is deliberately isolated to this entry point because reading the clock
    /// and retaining one record per node perturb short kernels.
    ///
    /// # Errors
    ///
    /// Returns the same validation, memory-provider, or operation errors as the unprofiled
    /// execution entry point.
    pub fn execute_prepared_outputs_with_resources_and_scratch_resident_profiled<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
    ) -> Result<(Vec<RocmTensorInput<'session>>, Vec<RocmTensorNodeTiming>), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut timings = CollectNodeTimings::default();
        let outputs = self.execute_prepared_outputs_with_resources_and_scratch_resident_timed(
            prepared,
            inputs,
            scratch,
            memory,
            &mut timings,
        )?;
        Ok((outputs, timings.0))
    }

    fn execute_prepared_outputs_with_resources_and_scratch_resident_timed<P, T>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        scratch: &mut RocmTensorScratch<'_, '_, 'session>,
        memory: &mut P,
        timings: &mut T,
    ) -> Result<Vec<RocmTensorInput<'session>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
        T: NodeTimingSink,
    {
        if scratch.poisoned
            || !std::ptr::eq(scratch.prepared, prepared)
            || !std::ptr::eq(scratch.session, self.session)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        let result = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            scratch.pool,
            memory,
            Some(&mut *scratch),
            None,
            timings,
        );
        if result.is_err() {
            scratch.poisoned = true;
        }
        result
    }

    /// Upload a tensor once for reuse as an immutable graph input on this selected `ROCm` session.
    ///
    /// The returned input owns the allocation and is bound to this backend. Call
    /// [`Self::update_input`] to synchronously replace its contents before a later execution.
    ///
    /// # Errors
    ///
    /// Returns an allocation or host-to-device transfer error.
    pub fn upload_input<P>(
        &self,
        tensor: &Tensor,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let resource = upload_tensor(memory, pool, tensor)?;
        if !resource.belongs_to_runtime(self.session.tensor_runtime()) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
        Ok(RocmTensorInput {
            session: self.session,
            shape: tensor.shape().to_vec(),
            resource,
        })
    }

    /// Replace the contents of a reusable input with a synchronous host-to-device transfer.
    ///
    /// Shape changes are rejected; create a new input when the graph input shape changes.
    ///
    /// # Errors
    ///
    /// Returns a session mismatch, shape mismatch, or host-to-device transfer error.
    pub fn update_input<P>(
        &self,
        input: &mut RocmTensorInput<'session>,
        tensor: &Tensor,
        memory: &mut P,
    ) -> Result<(), RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        input.validate_session(self.session)?;
        if tensor.shape() != input.shape {
            return Err(TensorError::ShapeMismatch {
                left: input.shape.clone(),
                right: tensor.shape().to_vec(),
            }
            .into());
        }
        transfer_tensor(memory, &mut input.resource, tensor)
    }

    /// Copy a resident device tensor to a host [`Tensor`] after validating its session and pool.
    ///
    /// Keep outputs resident between graph stages and call this only when host access is needed.
    ///
    /// # Errors
    ///
    /// Returns a session, device, or pool mismatch, or a device-to-host transfer error.
    pub fn download_output<P>(
        &self,
        output: &RocmTensorInput<'_>,
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        output.validate_session(self.session)?;
        if output.resource.pool() != pool {
            return Err(RocmTensorExecutionError::InputPoolMismatch);
        }
        download_tensor(memory, output)
    }

    /// Execute a prepared graph using persistent device inputs, reusing their allocations.
    ///
    /// Every graph input must have exactly one matching resource. Resources must belong to this
    /// selected backend and match the graph shape. No host upload or backend fallback occurs.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, resource identity error, or execution error.
    pub fn execute_prepared_with_resources<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<Tensor, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let output =
            self.execute_prepared_with_resources_resident(prepared, inputs, pool, memory)?;
        download_tensor(memory, &output)
    }

    /// Execute a prepared graph from persistent inputs and return a reusable resident output.
    ///
    /// The output may be passed directly as an input to a later graph on this same selected
    /// backend and pool. The executor validates every input and the output's device identity
    /// before dispatch; it performs no host readback.
    ///
    /// # Errors
    ///
    /// Returns an input validation error, resource identity error, or execution error.
    pub fn execute_prepared_with_resources_resident<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
    ) -> Result<RocmTensorInput<'session>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        let mut timings = NoopNodeTiming;
        let mut outputs = self.execute_prepared_outputs_with_input_sources_and_scratch(
            prepared,
            &[],
            inputs,
            pool,
            memory,
            None,
            None,
            &mut timings,
        )?;
        Ok(outputs.remove(0))
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    fn execute_prepared_outputs_with_input_sources_and_scratch<P>(
        &self,
        prepared: &RocmPreparedTensorGraph<'_>,
        host_inputs: &[(ValueId, Tensor)],
        resource_inputs: &[(ValueId, &RocmTensorInput<'_>)],
        pool: PcuMemoryPoolId,
        memory: &mut P,
        mut scratch: Option<&mut RocmTensorScratch<'_, '_, 'session>>,
        output_bank: Option<&RocmTensorOutputBank<'_, '_, 'session>>,
        timings: &mut impl NodeTimingSink,
    ) -> Result<Vec<RocmTensorInput<'session>>, RocmTensorExecutionError>
    where
        P: PcuMemoryProvider<Resource = RocmMemoryResource>,
    {
        validate_graph_input_sources(
            &prepared.nodes,
            host_inputs,
            resource_inputs,
            self.session,
            pool,
        )?;
        if let Some(scratch) = scratch.as_ref()
            && (!std::ptr::eq(scratch.prepared, prepared)
                || !std::ptr::eq(scratch.session, self.session)
                || scratch.poisoned
                || scratch.pool != pool)
        {
            return Err(RocmTensorExecutionError::ScratchMismatch);
        }
        if let Some(bank) = output_bank {
            validate_output_bank(
                bank,
                prepared,
                self.session,
                pool,
                resource_inputs,
                scratch.as_deref(),
            )?;
        }
        if prepared.nodes.iter().any(|node| {
            matches!(
                node.op,
                OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. }
            )
        }) && !self.rocblas.is_usable()
        {
            return Err(RocmTensorExecutionError::Unsupported {
                value: prepared.output,
                reason: TensorUnsupportedReason::Other(
                    "rocBLAS completion is uncertain; selected handle cannot run more work".into(),
                ),
            });
        }
        let graph = prepared.graph;
        let plan = prepared;
        let outputs = &plan.outputs;
        let node_count = plan.nodes.len();
        let mut resources: Vec<Option<RocmMemoryResource>> =
            std::iter::repeat_with(|| None).take(node_count).collect();
        let mut remaining_uses = plan.use_counts.clone();

        for (index, node) in plan.nodes.iter().enumerate() {
            let timing_mark = timings.begin();
            match node.op {
                OpDescriptor::Input => {
                    if let Some(output) = output_bank.and_then(|bank| bank.output(node.value)) {
                        let mut destination = output.resource.clone_for_tensor_input();
                        if let Some((_, input)) =
                            resource_inputs.iter().find(|(id, _)| *id == node.value)
                        {
                            let bytes = byte_len(&output.shape)?;
                            destination
                                .copy_from_resource(&input.resource, bytes)
                                .map_err(RocmTensorExecutionError::Completion)?;
                        } else {
                            let (_, tensor) = host_inputs
                                .iter()
                                .find(|(id, _)| *id == node.value)
                                .ok_or(TensorError::MissingInput(node.value))?;
                            transfer_tensor(memory, &mut destination, tensor)?;
                        }
                        resources[index] = Some(destination);
                        timings.finish(timing_mark, node);
                        continue;
                    }
                    if let Some((_, input)) =
                        resource_inputs.iter().find(|(id, _)| *id == node.value)
                    {
                        resources[index] = Some(input.resource_ref());
                        timings.finish(timing_mark, node);
                        continue;
                    }
                    let (_, tensor) = host_inputs
                        .iter()
                        .find(|(id, _)| *id == node.value)
                        .ok_or(TensorError::MissingInput(node.value))?;
                    resources[index] = Some(upload_tensor(memory, pool, tensor)?);
                }
                OpDescriptor::Constant(tensor) => {
                    resources[index] = Some(
                        if let Some(output) = output_bank.and_then(|bank| bank.output(node.value)) {
                            let mut resource = output.resource_ref();
                            transfer_tensor(memory, &mut resource, tensor)?;
                            resource
                        } else if outputs.contains(&node.value) {
                            upload_tensor(memory, pool, tensor)?
                        } else if let Some(scratch) = scratch.as_deref_mut() {
                            scratch.lease(index)?
                        } else {
                            upload_tensor(memory, pool, tensor)?
                        },
                    );
                }
                OpDescriptor::MatMul {
                    left,
                    right,
                    transpose_left,
                    transpose_right,
                } => {
                    let left_index = plan
                        .index_of(left)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
                    let right_index = plan
                        .index_of(right)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
                    let left_shape: [usize; 2] = graph
                        .shape(left)?
                        .try_into()
                        .map_err(|_| RocmTensorExecutionError::InvalidPlan(left))?;
                    let right_shape: [usize; 2] = graph
                        .shape(right)?
                        .try_into()
                        .map_err(|_| RocmTensorExecutionError::InvalidPlan(right))?;
                    let result = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_matmul_row_major_flags(
                        resources[left_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                        resources[right_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                        &result,
                        left_shape,
                        right_shape,
                        transpose_left,
                        transpose_right,
                    )?;
                    resources[index] = Some(result);
                    release_after_read(&mut resources, &mut remaining_uses, left_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, right_index)?;
                }
                OpDescriptor::Add { left, right }
                | OpDescriptor::Sub { left, right }
                | OpDescriptor::Mul { left, right } => {
                    let kind = match node.op {
                        OpDescriptor::Add { .. } => TensorDispatchKind::Add,
                        OpDescriptor::Sub { .. } => TensorDispatchKind::Sub,
                        OpDescriptor::Mul { .. } => TensorDispatchKind::Mul,
                        _ => unreachable!("binary branch selects only Add, Sub, or Mul"),
                    };
                    let left_index = plan
                        .index_of(left)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(left))?;
                    let right_index = plan
                        .index_of(right)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(right))?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_binary(
                        kind,
                        graph.shape(node.value)?,
                        resources[left_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(left))?,
                        resources[right_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(right))?,
                        &output,
                    )?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, left_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, right_index)?;
                }
                OpDescriptor::Relu { input } => {
                    let input_index = plan
                        .index_of(input)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(input))?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_relu(
                        node.shape,
                        resources[input_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(input))?,
                        &output,
                    )?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, input_index)?;
                }
                OpDescriptor::ReluBackward { input, upstream } => {
                    let input_index = plan
                        .index_of(input)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(input))?;
                    let upstream_index = plan
                        .index_of(upstream)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(upstream))?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_relu_backward(
                        node.shape,
                        resources[input_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(input))?,
                        resources[upstream_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(upstream))?,
                        &output,
                    )?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, input_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, upstream_index)?;
                }
                OpDescriptor::SgdUpdate {
                    weights,
                    gradient,
                    learning_rate,
                } => {
                    let weights_index = plan
                        .index_of(weights)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(weights))?;
                    let gradient_index = plan
                        .index_of(gradient)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(gradient))?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.execute_sgd_update(
                        node.shape,
                        resources[weights_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(weights))?,
                        resources[gradient_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(gradient))?,
                        learning_rate,
                        &output,
                    )?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, weights_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, gradient_index)?;
                }
                OpDescriptor::MeanSquaredError { prediction, target } => {
                    let prediction_index = plan
                        .index_of(prediction)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(prediction))?;
                    let target_index = plan
                        .index_of(target)
                        .ok_or(RocmTensorExecutionError::InvalidPlan(target))?;
                    let shape = graph.shape(prediction)?;
                    let count = shape
                        .iter()
                        .try_fold(1usize, |n, d| n.checked_mul(*d))
                        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
                    if count == 0 || count > i32::MAX as usize {
                        return Err(RocmTensorExecutionError::Unsupported {
                            value: node.value,
                            reason: TensorUnsupportedReason::Shape,
                        });
                    }
                    let (squared, ones) = if let Some(scratch) = scratch.as_ref() {
                        let squared = scratch
                            .mse_squared
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::ScratchMismatch)?
                            .clone_for_tensor_input();
                        let ones = scratch
                            .mse_ones
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::ScratchMismatch)?
                            .clone_for_tensor_input();
                        if !mse_scratch_lengths_fit(
                            count,
                            squared.device_buffer().len(),
                            ones.device_buffer().len(),
                        )? {
                            return Err(RocmTensorExecutionError::ScratchMismatch);
                        }
                        (squared, ones)
                    } else {
                        let squared = allocate_tensor(memory, pool, shape)?;
                        let ones = Tensor::new(vec![count], vec![1.0; count])?;
                        (squared, upload_tensor(memory, pool, &ones)?)
                    };
                    self.execute_elementwise(
                        TensorDispatchKind::SquaredDifference,
                        shape,
                        resources[prediction_index]
                            .as_ref()
                            .ok_or(RocmTensorExecutionError::MissingResource(prediction))?,
                        Some(
                            resources[target_index]
                                .as_ref()
                                .ok_or(RocmTensorExecutionError::MissingResource(target))?,
                        ),
                        &squared,
                    )?;
                    let output = execution_resource(
                        memory,
                        pool,
                        node.shape,
                        index,
                        node.value,
                        &mut scratch,
                        output_bank,
                    )?;
                    self.rocblas
                        .sdot_scaled(
                            count,
                            squared.device_buffer(),
                            1,
                            ones.device_buffer(),
                            1,
                            mse_scale(count),
                            output.device_buffer(),
                        )
                        .map_err(RocmTensorError::from)?;
                    resources[index] = Some(output);
                    release_after_read(&mut resources, &mut remaining_uses, prediction_index)?;
                    release_after_read(&mut resources, &mut remaining_uses, target_index)?;
                }
            }
            timings.finish(timing_mark, node);
        }

        outputs
            .iter()
            .map(|&output| {
                let output_index = plan
                    .index_of(output)
                    .ok_or(RocmTensorExecutionError::InvalidPlan(output))?;
                let resource = resources[output_index]
                    .take()
                    .ok_or(RocmTensorExecutionError::MissingResource(output))?;
                if resource.pool() != pool
                    || !resource.belongs_to_runtime(self.session.tensor_runtime())
                    || !matches!(
                        resource.access(),
                        PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
                    )
                {
                    return Err(RocmTensorExecutionError::OutputResourceMismatch);
                }
                Ok(RocmTensorInput {
                    session: self.session,
                    shape: graph.shape(output)?.to_vec(),
                    resource,
                })
            })
            .collect()
    }

    fn execute_binary(
        &self,
        kind: TensorDispatchKind,
        shape: &[usize],
        left: &RocmMemoryResource,
        right: &RocmMemoryResource,
        output: &RocmMemoryResource,
    ) -> Result<(), RocmTensorExecutionError> {
        self.execute_elementwise(kind, shape, left, Some(right), output)
    }

    fn execute_relu(
        &self,
        shape: &[usize],
        input: &RocmMemoryResource,
        output: &RocmMemoryResource,
    ) -> Result<(), RocmTensorExecutionError> {
        self.execute_elementwise(TensorDispatchKind::Relu, shape, input, None, output)
    }

    fn execute_relu_backward(
        &self,
        shape: &[usize],
        input: &RocmMemoryResource,
        upstream: &RocmMemoryResource,
        output: &RocmMemoryResource,
    ) -> Result<(), RocmTensorExecutionError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count = u32::try_from(count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        if self.relu_backward.borrow().is_none() {
            let runtime = self.session.tensor_runtime();
            let image = self
                .session
                .compile_tensor_source(RELU_BACKWARD_SOURCE)
                .map_err(RocmTensorExecutionError::Backend)?;
            let module = runtime
                .load_module(&image)
                .map_err(RocmTensorExecutionError::Completion)?;
            let kernel = module
                .function(c"tensor_relu_backward")
                .map_err(RocmTensorExecutionError::Completion)?;
            let stream = runtime
                .create_stream()
                .map_err(RocmTensorExecutionError::Completion)?;
            *self.relu_backward.borrow_mut() = Some((kernel, stream));
        }
        let cache = self.relu_backward.borrow();
        let (kernel, stream) = cache
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count_bytes = count.to_ne_bytes();
        let args = [
            HipKernelArgument::Buffer(input.device_buffer()),
            HipKernelArgument::Buffer(upstream.device_buffer()),
            HipKernelArgument::Buffer(output.device_buffer()),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: The compiled kernel takes three f32 buffers and a u32 count. Assessed shapes
        // and allocation sizes cover all guarded indices; the completion retains resources.
        #[allow(unsafe_code)]
        let mut completion =
            unsafe { kernel.launch(stream, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }
                .map_err(RocmTensorExecutionError::Completion)?;
        completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)
    }

    fn execute_sgd_update(
        &self,
        shape: &[usize],
        weights: &RocmMemoryResource,
        gradient: &RocmMemoryResource,
        learning_rate: f32,
        output: &RocmMemoryResource,
    ) -> Result<(), RocmTensorExecutionError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let count = u32::try_from(count)
            .ok()
            .filter(|count| *count > 0)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        if self.sgd_update.borrow().is_none() {
            let runtime = self.session.tensor_runtime();
            let image = self
                .session
                .compile_tensor_source(SGD_UPDATE_SOURCE)
                .map_err(RocmTensorExecutionError::Backend)?;
            let module = runtime
                .load_module(&image)
                .map_err(RocmTensorExecutionError::Completion)?;
            let kernel = module
                .function(c"tensor_sgd_update")
                .map_err(RocmTensorExecutionError::Completion)?;
            let stream = runtime
                .create_stream()
                .map_err(RocmTensorExecutionError::Completion)?;
            *self.sgd_update.borrow_mut() = Some((kernel, stream));
        }
        let cache = self.sgd_update.borrow();
        let (kernel, stream) = cache
            .as_ref()
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let learning_rate_bytes = learning_rate.to_ne_bytes();
        let count_bytes = count.to_ne_bytes();
        let args = [
            HipKernelArgument::Buffer(weights.device_buffer()),
            HipKernelArgument::Buffer(gradient.device_buffer()),
            HipKernelArgument::Buffer(output.device_buffer()),
            HipKernelArgument::Bytes(&learning_rate_bytes),
            HipKernelArgument::Bytes(&count_bytes),
        ];
        // SAFETY: assessed equal nonempty f32 shapes fit u32, and execution resources cover all
        // indices. The completion retains the buffers until the one launch has finished.
        #[allow(unsafe_code)]
        let mut completion =
            unsafe { kernel.launch(stream, [count.div_ceil(256), 1, 1], [256, 1, 1], 0, &args) }
                .map_err(RocmTensorExecutionError::Completion)?;
        completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)
    }

    #[allow(clippy::too_many_lines)] // Keeps the cache, owned bindings, and completion lifetime explicit.
    fn execute_elementwise(
        &self,
        kind: TensorDispatchKind,
        shape: &[usize],
        left: &RocmMemoryResource,
        right: Option<&RocmMemoryResource>,
        output: &RocmMemoryResource,
    ) -> Result<(), RocmTensorExecutionError> {
        let count = shape
            .iter()
            .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let invocations = u32::try_from(count)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        let logical_count = invocations.get();
        let cache_key = (kind, logical_count);
        {
            let mut cache = self.add_dispatches.borrow_mut();
            if !cache.iter().any(|(key, _)| *key == cache_key) {
                let prepared = self
                    .session
                    .prepare_dispatch_owned_kernel(
                        kind.kernel(logical_count),
                        PcuInvocationShape::invocations(invocations),
                    )
                    .map_err(RocmTensorExecutionError::Backend)?;
                if cache.len() == ADD_DISPATCH_CACHE_CAPACITY {
                    cache.pop_front();
                }
                cache.push_back((cache_key, prepared));
            }
        }
        let left_binding = self
            .session
            .bind(
                match kind {
                    TensorDispatchKind::Add | TensorDispatchKind::Sub | TensorDispatchKind::Mul => {
                        ADD_LEFT_REF
                    }
                    TensorDispatchKind::Relu => RELU_INPUT_REF,
                    TensorDispatchKind::SquaredDifference => MSE_LEFT_REF,
                },
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                left,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        let mut bindings = vec![left_binding];
        if let Some(right) = right {
            let right_binding = self
                .session
                .bind(
                    match kind {
                        TensorDispatchKind::Add
                        | TensorDispatchKind::Sub
                        | TensorDispatchKind::Mul => ADD_RIGHT_REF,
                        TensorDispatchKind::Relu => {
                            return Err(RocmTensorExecutionError::SizeOverflow);
                        }
                        TensorDispatchKind::SquaredDifference => MSE_RIGHT_REF,
                    },
                    PcuBindingAccess::ReadOnly,
                    PcuBindingType::Value(PcuValueType::f32()),
                    right,
                )
                .map_err(RocmTensorExecutionError::Backend)?;
            bindings.push(right_binding);
        }
        let output_binding = self
            .session
            .bind(
                match kind {
                    TensorDispatchKind::Add | TensorDispatchKind::Sub | TensorDispatchKind::Mul => {
                        ADD_OUTPUT_REF
                    }
                    TensorDispatchKind::Relu => RELU_OUTPUT_REF,
                    TensorDispatchKind::SquaredDifference => MSE_OUTPUT_REF,
                },
                PcuBindingAccess::WriteOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                output,
            )
            .map_err(RocmTensorExecutionError::Backend)?;
        bindings.push(output_binding);
        let mut completion = {
            let cache = self.add_dispatches.borrow();
            let prepared = cache
                .iter()
                .find(|(key, _)| *key == cache_key)
                .map(|(_, prepared)| prepared)
                .ok_or(RocmTensorExecutionError::SizeOverflow)?;
            prepared
                .submit(&bindings)
                .map_err(RocmTensorExecutionError::Backend)?
        };
        match completion
            .wait()
            .map_err(RocmTensorExecutionError::Completion)?
        {
            PcuCompletionOutcome::Succeeded => Ok(()),
            PcuCompletionOutcome::Failed => Err(RocmTensorExecutionError::FailedCompletion),
        }
    }

    #[allow(clippy::many_single_char_names)] // BLAS' A/B/C and m/n/k notation is standardized.
    fn execute_matmul_row_major(
        &self,
        a: &RocmMemoryResource,
        b: &RocmMemoryResource,
        c: &RocmMemoryResource,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<(), RocmTensorError> {
        self.execute_matmul_row_major_flags(a, b, c, [rows, inner], [inner, columns], false, false)
    }

    /// Interpret row-major operands with optional logical transposition; rocBLAS consumes the
    /// same backing through its column-major view, swapping operand order and transpose flags.
    #[allow(clippy::too_many_arguments, clippy::many_single_char_names)]
    fn execute_matmul_row_major_flags(
        &self,
        a: &RocmMemoryResource,
        b: &RocmMemoryResource,
        c: &RocmMemoryResource,
        a_shape: [usize; 2],
        b_shape: [usize; 2],
        transpose_a: bool,
        transpose_b: bool,
    ) -> Result<(), RocmTensorError> {
        let rows = a_shape[usize::from(transpose_a)];
        let inner = a_shape[usize::from(!transpose_a)];
        let right_inner = b_shape[usize::from(transpose_b)];
        let columns = b_shape[usize::from(!transpose_b)];
        if inner != right_inner {
            return Err(RocmTensorError::InvalidShape);
        }
        if rows == 0 || inner == 0 || columns == 0 {
            return Err(RocmTensorError::InvalidShape);
        }
        let a_bytes = a_shape[0]
            .checked_mul(a_shape[1])
            .and_then(|n| n.checked_mul(4));
        let b_bytes = b_shape[0]
            .checked_mul(b_shape[1])
            .and_then(|n| n.checked_mul(4));
        let c_bytes = rows.checked_mul(columns).and_then(|n| n.checked_mul(4));
        if a_bytes.is_none() || b_bytes.is_none() || c_bytes.is_none() {
            return Err(RocmTensorError::DimensionOverflow);
        }
        if !matches!(
            a.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) || !matches!(
            b.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) || !matches!(
            c.access(),
            PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(RocmTensorError::InvalidMemoryAccess);
        }
        let (m, n) = (columns, rows);
        self.rocblas.sgemm(
            transpose_b,
            transpose_a,
            m,
            n,
            inner,
            1.0,
            b.device_buffer(),
            b_shape[1],
            a.device_buffer(),
            a_shape[1],
            0.0,
            c.device_buffer(),
            m,
        )?;
        Ok(())
    }
}

fn mse_scratch_lengths_fit(
    count: usize,
    squared_bytes: usize,
    ones_bytes: usize,
) -> Result<bool, RocmTensorExecutionError> {
    let required_bytes = count
        .checked_mul(size_of::<f32>())
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    Ok(squared_bytes >= required_bytes && ones_bytes >= required_bytes)
}

/// A structurally validated `ROCm` execution plan for one graph output.
///
/// The graph is borrowed and must remain alive and unchanged while the plan is used.
pub struct RocmPreparedTensorGraph<'graph> {
    graph: &'graph Graph,
    output: ValueId,
    outputs: Vec<ValueId>,
    nodes: Vec<NodeDescriptor<'graph>>,
    index_by_value: HashMap<ValueId, usize>,
    use_counts: Vec<usize>,
}

/// Synchronous wall time spent executing one prepared graph node in diagnostic mode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RocmTensorNodeTiming {
    /// Graph value produced by this node.
    pub value: ValueId,
    /// Operation kind, such as `MatMul`, `ReluBackward`, or `Input`.
    pub operation: &'static str,
    /// Elapsed wall time, including synchronous device completion.
    pub elapsed: Duration,
}

trait NodeTimingSink {
    type Mark;

    fn begin(&mut self) -> Self::Mark;
    fn finish(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>);
}

struct NoopNodeTiming;

impl NodeTimingSink for NoopNodeTiming {
    type Mark = ();

    #[inline(always)]
    fn begin(&mut self) {}

    #[inline(always)]
    fn finish(&mut self, (): Self::Mark, _: &NodeDescriptor<'_>) {}
}

#[derive(Default)]
struct CollectNodeTimings(Vec<RocmTensorNodeTiming>);

impl NodeTimingSink for CollectNodeTimings {
    type Mark = Instant;

    fn begin(&mut self) -> Self::Mark {
        Instant::now()
    }

    fn finish(&mut self, mark: Self::Mark, node: &NodeDescriptor<'_>) {
        self.0.push(RocmTensorNodeTiming {
            value: node.value,
            operation: match node.op {
                OpDescriptor::Input => "Input",
                OpDescriptor::Constant(_) => "Constant",
                OpDescriptor::MatMul { .. } => "MatMul",
                OpDescriptor::Add { .. } => "Add",
                OpDescriptor::Sub { .. } => "Sub",
                OpDescriptor::Mul { .. } => "Mul",
                OpDescriptor::Relu { .. } => "Relu",
                OpDescriptor::ReluBackward { .. } => "ReluBackward",
                OpDescriptor::SgdUpdate { .. } => "SgdUpdate",
                OpDescriptor::MeanSquaredError { .. } => "MeanSquaredError",
            },
            elapsed: mark.elapsed(),
        });
    }
}

/// Reusable device allocations for one exact prepared graph, session, and pool.
///
/// Mutably borrowing this value for execution excludes concurrent reuse of its buffers.
pub struct RocmTensorScratch<'plan, 'graph, 'session> {
    prepared: &'plan RocmPreparedTensorGraph<'graph>,
    session: &'session RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    resources: Vec<Option<RocmMemoryResource>>,
    mse_squared: Option<RocmMemoryResource>,
    mse_ones: Option<RocmMemoryResource>,
    poisoned: bool,
}

/// Caller-owned output allocations reusable for repeated execution of one exact prepared plan.
pub struct RocmTensorOutputBank<'plan, 'graph, 'session> {
    prepared: &'plan RocmPreparedTensorGraph<'graph>,
    session: &'session RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    outputs: Vec<RocmTensorInput<'session>>,
    poisoned: bool,
}

impl<'session> RocmTensorOutputBank<'_, '_, 'session> {
    /// Outputs in the order requested when the graph was prepared.
    #[must_use]
    pub fn outputs(&self) -> &[RocmTensorInput<'session>] {
        &self.outputs
    }

    fn output(&self, value: ValueId) -> Option<&RocmTensorInput<'session>> {
        output_position(&self.prepared.outputs, value).and_then(|index| self.outputs.get(index))
    }
}

fn output_position(outputs: &[ValueId], value: ValueId) -> Option<usize> {
    outputs.iter().position(|&output| output == value)
}

fn validate_output_bank(
    bank: &RocmTensorOutputBank<'_, '_, '_>,
    prepared: &RocmPreparedTensorGraph<'_>,
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    inputs: &[(ValueId, &RocmTensorInput<'_>)],
    scratch: Option<&RocmTensorScratch<'_, '_, '_>>,
) -> Result<(), RocmTensorExecutionError> {
    if bank.poisoned
        || !std::ptr::eq(bank.prepared, prepared)
        || !std::ptr::eq(bank.session, session)
        || bank.pool != pool
        || bank.outputs.len() != prepared.outputs.len()
    {
        return Err(RocmTensorExecutionError::OutputResourceMismatch);
    }
    for (index, (&value, output)) in prepared.outputs.iter().zip(&bank.outputs).enumerate() {
        let expected_shape = prepared.graph.shape(value)?;
        if !std::ptr::eq(output.session, session)
            || output.shape != expected_shape
            || output.resource.pool() != pool
            || !output.resource.belongs_to_runtime(session.tensor_runtime())
            || output.resource.device_buffer().len() < byte_len(expected_shape)?
            || !matches!(
                output.resource.access(),
                PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
            )
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
        if bank.outputs[..index]
            .iter()
            .any(|prior| prior.resource.same_allocation(&output.resource))
            || inputs
                .iter()
                .any(|(_, input)| input.resource.same_allocation(&output.resource))
            || scratch.is_some_and(|scratch| {
                scratch
                    .resources
                    .iter()
                    .flatten()
                    .any(|resource| resource.same_allocation(&output.resource))
                    || scratch
                        .mse_squared
                        .as_ref()
                        .is_some_and(|resource| resource.same_allocation(&output.resource))
                    || scratch
                        .mse_ones
                        .as_ref()
                        .is_some_and(|resource| resource.same_allocation(&output.resource))
            })
        {
            return Err(RocmTensorExecutionError::OutputResourceMismatch);
        }
    }
    Ok(())
}

impl RocmTensorScratch<'_, '_, '_> {
    fn lease(&self, index: usize) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
        self.resources
            .get(index)
            .and_then(Option::as_ref)
            .map(RocmMemoryResource::clone_for_tensor_input)
            .ok_or_else(|| {
                RocmTensorExecutionError::MissingResource(
                    self.prepared
                        .nodes
                        .get(index)
                        .map_or(self.prepared.output, |node| node.value),
                )
            })
    }
}

fn execution_resource<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
    index: usize,
    value: ValueId,
    scratch: &mut Option<&mut RocmTensorScratch<'_, '_, '_>>,
    output_bank: Option<&RocmTensorOutputBank<'_, '_, '_>>,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    if let Some(bank) = output_bank
        && let Some(output) = bank.output(value)
    {
        return Ok(output.resource_ref());
    }
    if let Some(scratch) = scratch.as_deref_mut()
        && !scratch.prepared.outputs.contains(&value)
    {
        return scratch.lease(index);
    }
    allocate_tensor(memory, pool, shape)
}

fn scratch_stores_node(op: OpDescriptor<'_>, value: ValueId, outputs: &[ValueId]) -> bool {
    !outputs.contains(&value) && !matches!(op, OpDescriptor::Input)
}

fn mse_scratch_element_count(
    graph: &Graph,
    nodes: &[NodeDescriptor<'_>],
) -> Result<usize, RocmTensorExecutionError> {
    nodes.iter().try_fold(0usize, |max_count, node| {
        let OpDescriptor::MeanSquaredError { prediction, .. } = node.op else {
            return Ok(max_count);
        };
        let shape = graph.shape(prediction)?;
        // The MSE output is scalar; size internal scratch from its prediction shape.
        let count = shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        if count == 0 || count > i32::MAX as usize {
            return Err(RocmTensorExecutionError::Unsupported {
                value: node.value,
                reason: TensorUnsupportedReason::Shape,
            });
        }
        Ok(max_count.max(count))
    })
}

impl RocmPreparedTensorGraph<'_> {
    fn index_of(&self, value: ValueId) -> Option<usize> {
        self.index_by_value.get(&value).copied()
    }
}

#[derive(Debug)]
struct GraphExecutionPreflight<'a> {
    nodes: Vec<NodeDescriptor<'a>>,
    index_by_value: HashMap<ValueId, usize>,
    use_counts: Vec<usize>,
}

#[cfg(test)]
fn prepare_graph<'a, A: TensorOperationAssessor>(
    graph: &'a Graph,
    output: ValueId,
    assessor: &A,
) -> Result<GraphExecutionPreflight<'a>, RocmTensorExecutionError> {
    prepare_graph_outputs_plan(graph, &[output], assessor)
}

fn prepare_graph_outputs_plan<'a, A: TensorOperationAssessor>(
    graph: &'a Graph,
    outputs: &[ValueId],
    assessor: &A,
) -> Result<GraphExecutionPreflight<'a>, RocmTensorExecutionError> {
    if outputs.is_empty() {
        return Err(RocmTensorExecutionError::EmptyOutputs);
    }
    for (index, output) in outputs.iter().enumerate() {
        if outputs[..index].contains(output) {
            return Err(RocmTensorExecutionError::DuplicateOutput(*output));
        }
    }
    let nodes: Vec<_> = graph.nodes().collect();
    let graph_indices: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(index, node)| (node.value, index))
        .collect();
    let mut needed = vec![false; nodes.len()];
    for output in outputs {
        let output_index = graph_indices
            .get(output)
            .copied()
            .ok_or(TensorError::UnknownValue(*output))?;
        for (needed, closure_needed) in
            needed
                .iter_mut()
                .zip(dependency_closure(&nodes, &graph_indices, output_index)?)
        {
            *needed |= closure_needed;
        }
    }

    let mut selected = Vec::new();
    for (index, node) in nodes.into_iter().enumerate() {
        if needed[index] {
            let expected_route = match node.op {
                OpDescriptor::Input
                | OpDescriptor::Constant(_)
                | OpDescriptor::ReluBackward { .. }
                | OpDescriptor::SgdUpdate { .. } => TensorExecutionRoute::Native,
                OpDescriptor::MatMul { .. } => TensorExecutionRoute::Library,
                OpDescriptor::Add { .. }
                | OpDescriptor::Sub { .. }
                | OpDescriptor::Mul { .. }
                | OpDescriptor::Relu { .. }
                | OpDescriptor::MeanSquaredError { .. } => TensorExecutionRoute::Synthesized,
            };
            match assessor.assess_node(graph, node) {
                TensorOperationSupport::Supported { route, .. } if route == expected_route => {}
                TensorOperationSupport::Supported { .. } => {
                    return Err(RocmTensorExecutionError::Unsupported {
                        value: node.value,
                        reason: TensorUnsupportedReason::Other(
                            "assessor selected a route this executor does not implement".into(),
                        ),
                    });
                }
                TensorOperationSupport::Unsupported { reason } => {
                    return Err(RocmTensorExecutionError::Unsupported {
                        value: node.value,
                        reason,
                    });
                }
            }
            selected.push(node);
        }
    }

    let index_by_value: HashMap<_, _> = selected
        .iter()
        .enumerate()
        .map(|(index, node)| (node.value, index))
        .collect();
    let mut use_counts = value_use_counts(&selected, &index_by_value)?;
    for output in outputs {
        let index = index_by_value
            .get(output)
            .copied()
            .ok_or(RocmTensorExecutionError::InvalidPlan(*output))?;
        use_counts[index] = use_counts[index]
            .checked_add(1)
            .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    }
    for node in &selected {
        let _ = byte_len(node.shape)?;
    }
    Ok(GraphExecutionPreflight {
        nodes: selected,
        index_by_value,
        use_counts,
    })
}

fn dependency_closure(
    nodes: &[NodeDescriptor<'_>],
    graph_indices: &HashMap<ValueId, usize>,
    output_index: usize,
) -> Result<Vec<bool>, TensorError> {
    let mut needed = vec![false; nodes.len()];
    let mut pending = vec![output_index];
    while let Some(index) = pending.pop() {
        if needed[index] {
            continue;
        }
        needed[index] = true;
        for operand in op_operands(nodes[index].op) {
            let operand_index = graph_indices
                .get(&operand)
                .copied()
                .ok_or(TensorError::UnknownValue(operand))?;
            pending.push(operand_index);
        }
    }
    Ok(needed)
}

fn value_use_counts(
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &HashMap<ValueId, usize>,
) -> Result<Vec<usize>, RocmTensorExecutionError> {
    let mut use_counts = vec![0_usize; nodes.len()];
    for node in nodes {
        for operand in op_operands(node.op) {
            let index = index_by_value
                .get(&operand)
                .copied()
                .ok_or(RocmTensorExecutionError::InvalidPlan(operand))?;
            use_counts[index] = use_counts[index]
                .checked_add(1)
                .ok_or(RocmTensorExecutionError::SizeOverflow)?;
        }
    }
    Ok(use_counts)
}

fn op_operands(op: OpDescriptor<'_>) -> impl Iterator<Item = ValueId> {
    let (first, second) = match op {
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right }
        | OpDescriptor::MatMul { left, right, .. } => (Some(left), Some(right)),
        OpDescriptor::ReluBackward { input, upstream } => (Some(input), Some(upstream)),
        OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } => (Some(weights), Some(gradient)),
        OpDescriptor::Relu { input } => (Some(input), None),
        OpDescriptor::MeanSquaredError { prediction, target } => (Some(prediction), Some(target)),
        OpDescriptor::Input | OpDescriptor::Constant(_) => (None, None),
    };
    first.into_iter().chain(second)
}

#[cfg(test)]
fn validate_graph_inputs(
    nodes: &[NodeDescriptor<'_>],
    inputs: &[(ValueId, Tensor)],
) -> Result<(), TensorError> {
    validate_graph_inputs_inner(nodes, inputs, true)
}

fn validate_graph_inputs_inner(
    nodes: &[NodeDescriptor<'_>],
    inputs: &[(ValueId, Tensor)],
    require_all: bool,
) -> Result<(), TensorError> {
    for (index, (id, tensor)) in inputs.iter().enumerate() {
        if inputs[..index].iter().any(|(previous, _)| previous == id) {
            return Err(TensorError::DuplicateInput(*id));
        }
        let Some(node) = nodes.iter().find(|node| node.value == *id) else {
            return Err(TensorError::ExtraInput(*id));
        };
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorError::ExtraInput(*id));
        }
        if tensor.shape() != node.shape {
            return Err(TensorError::ShapeMismatch {
                left: tensor.shape().to_vec(),
                right: node.shape.to_vec(),
            });
        }
    }
    for node in nodes {
        if require_all
            && matches!(node.op, OpDescriptor::Input)
            && !inputs.iter().any(|(id, _)| *id == node.value)
        {
            return Err(TensorError::MissingInput(node.value));
        }
    }
    Ok(())
}

fn validate_graph_input_sources(
    nodes: &[NodeDescriptor<'_>],
    host_inputs: &[(ValueId, Tensor)],
    resource_inputs: &[(ValueId, &RocmTensorInput<'_>)],
    session: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
) -> Result<(), RocmTensorExecutionError> {
    validate_graph_inputs_inner(nodes, host_inputs, false)?;
    for (index, (id, input)) in resource_inputs.iter().enumerate() {
        if resource_inputs[..index]
            .iter()
            .any(|(previous, _)| previous == id)
            || host_inputs.iter().any(|(previous, _)| previous == id)
        {
            return Err(TensorError::DuplicateInput(*id).into());
        }
        let Some(node) = nodes.iter().find(|node| node.value == *id) else {
            return Err(TensorError::ExtraInput(*id).into());
        };
        if !matches!(node.op, OpDescriptor::Input) {
            return Err(TensorError::ExtraInput(*id).into());
        }
        if node.shape != input.shape {
            return Err(TensorError::ShapeMismatch {
                left: node.shape.to_vec(),
                right: input.shape.clone(),
            }
            .into());
        }
        input.validate_session(session)?;
        if input.resource.pool() != pool {
            return Err(RocmTensorExecutionError::InputPoolMismatch);
        }
        if !matches!(
            input.resource.access(),
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(RocmTensorExecutionError::InputResourceMismatch);
        }
    }
    for node in nodes {
        if matches!(node.op, OpDescriptor::Input)
            && !host_inputs.iter().any(|(id, _)| *id == node.value)
            && !resource_inputs.iter().any(|(id, _)| *id == node.value)
        {
            return Err(TensorError::MissingInput(node.value).into());
        }
    }
    Ok(())
}

fn byte_len(shape: &[usize]) -> Result<usize, RocmTensorExecutionError> {
    let bytes = shape
        .iter()
        .try_fold(size_of::<f32>(), |bytes, dimension| {
            bytes
                .checked_mul(*dimension)
                .ok_or(RocmTensorExecutionError::SizeOverflow)
        })?;
    if bytes > isize::MAX as usize {
        return Err(RocmTensorExecutionError::SizeOverflow);
    }
    Ok(bytes)
}

#[allow(clippy::cast_precision_loss)] // Matches the tensor dialect's f32 mean divisor semantics.
fn mse_scale(element_count: usize) -> f32 {
    1.0 / element_count as f32
}

fn allocate_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    shape: &[usize],
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    let size_bytes =
        u64::try_from(byte_len(shape)?).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    Ok(memory.allocate(PcuMemoryAllocationRequest {
        pool,
        size_bytes,
        alignment_bytes: u64::try_from(align_of::<f32>())
            .map_err(|_| RocmTensorExecutionError::SizeOverflow)?,
        access: PcuMemoryAccess::ReadWrite,
        host_access: PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    })?)
}

fn upload_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    pool: PcuMemoryPoolId,
    tensor: &Tensor,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    let mut resource = allocate_tensor(memory, pool, tensor.shape())?;
    transfer_tensor(memory, &mut resource, tensor)?;
    Ok(resource)
}

fn download_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    output: &RocmTensorInput<'_>,
) -> Result<Tensor, RocmTensorExecutionError> {
    let bytes_len = byte_len(&output.shape)?;
    let mut data = vec![0.0_f32; bytes_len / size_of::<f32>()];
    // SAFETY: `data` is initialized, contiguous f32 storage. Every f32 bit pattern is valid;
    // u8 has alignment one, and its byte length was checked from the output shape.
    let output_bytes =
        unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast::<u8>(), bytes_len) };
    memory.transfer_from(&output.resource, 0, output_bytes)?;
    Ok(Tensor::new(output.shape.clone(), data)?)
}

fn transfer_tensor<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    memory: &mut P,
    resource: &mut RocmMemoryResource,
    tensor: &Tensor,
) -> Result<(), RocmTensorExecutionError> {
    let bytes_len = byte_len(tensor.shape())?;
    // SAFETY: Tensor construction checks that shape and initialized f32 data agree. f32 has
    // exactly four bytes with no padding, u8 has alignment one, and the byte extent is checked.
    // The provider's synchronous transfer ends before the Tensor can be released by this call.
    let bytes =
        unsafe { std::slice::from_raw_parts(tensor.data().as_ptr().cast::<u8>(), bytes_len) };
    memory.transfer_to(resource, 0, bytes)?;
    Ok(())
}

fn release_after_read(
    resources: &mut [Option<RocmMemoryResource>],
    remaining_uses: &mut [usize],
    index: usize,
) -> Result<(), RocmTensorExecutionError> {
    let Some(remaining) = remaining_uses.get_mut(index) else {
        return Err(RocmTensorExecutionError::SizeOverflow);
    };
    *remaining = remaining
        .checked_sub(1)
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    if *remaining == 0 {
        resources[index] = None;
    }
    Ok(())
}

// SAFETY: `Rocblas::sgemm` acquires shared exclusive allocation leases for A/B/C and performs a
// HIP device synchronization before returning. If synchronization fails, it poisons the handle
// and deliberately forgets all leases, retaining the allocations indefinitely. All validation
// errors occur before rocBLAS is called, so no device work can outlive a returned error.
unsafe impl TensorSynchronousF32MatMulBackend for RocmTensorAssessor<'_> {
    type Resource = RocmMemoryResource;
    type Error = RocmTensorError;

    fn matmul_row_major(
        &self,
        a: &Self::Resource,
        b: &Self::Resource,
        c: &Self::Resource,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<(), Self::Error> {
        self.execute_matmul_row_major(a, b, c, rows, inner, columns)
    }
}

impl TensorOperationAssessor for RocmTensorAssessor<'_> {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        if matches!(
            node.op,
            OpDescriptor::MatMul { .. } | OpDescriptor::MeanSquaredError { .. }
        ) && !self.rocblas.is_usable()
        {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Other(
                    "rocBLAS completion is uncertain; selected handle cannot run more work".into(),
                ),
            };
        }
        assess_tensor_node(graph, node)
    }
}

fn assess_tensor_node(graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
    match node.op {
        OpDescriptor::Input | OpDescriptor::Constant(_) => TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Native,
            workspace_bytes: Some(0),
        },
        OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } => {
            if matmul_shape_supported(
                graph,
                node.shape,
                left,
                right,
                transpose_left,
                transpose_right,
            ) {
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Library,
                    workspace_bytes: None,
                }
            } else {
                TensorOperationSupport::Unsupported {
                    reason: TensorUnsupportedReason::Shape,
                }
            }
        }
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right } => match (graph.shape(left), graph.shape(right)) {
            (Ok(left), Ok(right))
                if left == right
                    && left
                        .iter()
                        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                        .is_some_and(|count| count > 0 && u32::try_from(count).is_ok())
                    && node.shape == left =>
            {
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Synthesized,
                    workspace_bytes: Some(0),
                }
            }
            _ => TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            },
        },
        OpDescriptor::Relu { input } => match graph.shape(input) {
            Ok(input_shape)
                if input_shape == node.shape
                    && input_shape
                        .iter()
                        .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                        .is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) =>
            {
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Synthesized,
                    workspace_bytes: Some(0),
                }
            }
            _ => TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            },
        },
        OpDescriptor::ReluBackward { input, upstream } => {
            assess_relu_backward(graph, node.shape, input, upstream)
        }
        OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } => assess_sgd_update(graph, node.shape, weights, gradient),
        OpDescriptor::MeanSquaredError { prediction, target } => {
            match (graph.shape(prediction), graph.shape(target)) {
                (Ok(left), Ok(right)) if left == right && node.shape.is_empty() => {
                    let count = left.iter().try_fold(1usize, |n, d| n.checked_mul(*d));
                    if count.is_some_and(|n| n > 0 && i32::try_from(n).is_ok()) {
                        TensorOperationSupport::Supported {
                            route: TensorExecutionRoute::Synthesized,
                            workspace_bytes: None,
                        }
                    } else {
                        TensorOperationSupport::Unsupported {
                            reason: TensorUnsupportedReason::Shape,
                        }
                    }
                }
                _ => TensorOperationSupport::Unsupported {
                    reason: TensorUnsupportedReason::Shape,
                },
            }
        }
    }
}

fn assess_relu_backward(
    graph: &Graph,
    output_shape: &[usize],
    input: ValueId,
    upstream: ValueId,
) -> TensorOperationSupport {
    match (graph.shape(input), graph.shape(upstream)) {
        (Ok(input_shape), Ok(upstream_shape))
            if input_shape == upstream_shape
                && input_shape == output_shape
                && input_shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) =>
        {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        }
        _ => TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        },
    }
}

fn assess_sgd_update(
    graph: &Graph,
    output_shape: &[usize],
    weights: ValueId,
    gradient: ValueId,
) -> TensorOperationSupport {
    match (graph.shape(weights), graph.shape(gradient)) {
        (Ok(weights_shape), Ok(gradient_shape))
            if weights_shape == gradient_shape
                && weights_shape == output_shape
                && weights_shape
                    .iter()
                    .try_fold(1usize, |count, dimension| count.checked_mul(*dimension))
                    .is_some_and(|count| count > 0 && u32::try_from(count).is_ok()) =>
        {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        }
        _ => TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        },
    }
}

fn matmul_shape_supported(
    graph: &Graph,
    output_shape: &[usize],
    left: ValueId,
    right: ValueId,
    transpose_left: bool,
    transpose_right: bool,
) -> bool {
    let (Ok([left_rows, left_columns]), Ok([right_rows, right_columns])) =
        (graph.shape(left), graph.shape(right))
    else {
        return false;
    };
    let rows = if transpose_left {
        *left_columns
    } else {
        *left_rows
    };
    let inner = if transpose_left {
        *left_rows
    } else {
        *left_columns
    };
    let right_inner = if transpose_right {
        *right_columns
    } else {
        *right_rows
    };
    let columns = if transpose_right {
        *right_rows
    } else {
        *right_columns
    };
    rows > 0
        && inner > 0
        && right_inner == inner
        && columns > 0
        && [
            *left_rows,
            *left_columns,
            *right_rows,
            *right_columns,
            rows,
            inner,
            columns,
        ]
        .into_iter()
        .all(|dimension| i32::try_from(dimension).is_ok())
        && [
            (*left_rows, *left_columns),
            (*right_rows, *right_columns),
            (rows, columns),
        ]
        .into_iter()
        .all(|(a, b)| a.checked_mul(b).and_then(|n| n.checked_mul(4)).is_some())
        && output_shape == [rows, columns]
}

#[cfg(test)]
mod tests {
    use fusion_pcu_tensor::{
        Graph,
        NodeDescriptor,
        Tensor,
        TensorError,
        TensorOperationAssessor,
        TensorOperationSupport,
    };

    use super::{
        add_kernel,
        assess_tensor_node,
        mse_kernel,
        mse_scratch_lengths_fit,
        mse_scratch_element_count,
        output_position,
        prepare_graph,
        prepare_graph_outputs_plan,
        relu_kernel,
        scratch_stores_node,
        validate_graph_inputs,
        CollectNodeTimings,
        NodeTimingSink,
        RocmTensorExecutionError,
        TensorExecutionRoute,
        TensorUnsupportedReason,
    };

    #[test]
    fn diagnostic_node_timings_preserve_execution_order_and_count() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let sum = graph.add(left, right).unwrap();
        let left_node = NodeDescriptor {
            value: left,
            op: graph.nodes().find(|node| node.value == left).unwrap().op,
            shape: graph.shape(left).unwrap(),
        };
        let sum_node = NodeDescriptor {
            value: sum,
            op: graph.nodes().find(|node| node.value == sum).unwrap().op,
            shape: graph.shape(sum).unwrap(),
        };
        let mut timings = CollectNodeTimings::default();

        let mark = timings.begin();
        timings.finish(mark, &left_node);
        let mark = timings.begin();
        timings.finish(mark, &sum_node);

        assert_eq!(timings.0.len(), 2);
        assert_eq!(timings.0[0].value, left);
        assert_eq!(timings.0[0].operation, "Input");
        assert_eq!(timings.0[1].value, sum);
        assert_eq!(timings.0[1].operation, "Add");
    }

    #[test]
    fn add_dispatch_kernel_lowers_with_valid_ssa_values() {
        let kernel = add_kernel(17);

        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();

        assert!(source.contains("fusion_kernel"));
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    #[test]
    fn relu_dispatch_kernel_lowers_max_against_zero() {
        let kernel = relu_kernel(17);

        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();

        assert!(source.contains("fmaxf(v2, v1)"));
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    #[test]
    fn mse_dispatch_kernel_lowers_squared_difference() {
        let kernel = mse_kernel(17);
        let source = crate::lower_dispatch_to_hip_source(&kernel).unwrap();
        assert!(source.contains(" - "));
        assert!(source.contains(" * "));
        assert_eq!(kernel.entry.logical_shape, [17, 1, 1]);
    }

    struct PureRocmAssessor;

    impl TensorOperationAssessor for PureRocmAssessor {
        fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
            assess_tensor_node(graph, node)
        }
    }

    #[test]
    fn preflight_selects_only_requested_output_dependencies() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let unrelated_left = graph.input([2, 2]).unwrap();
        let unrelated_right = graph.input([2, 2]).unwrap();
        let _unsupported_but_unreachable = graph.add(unrelated_left, unrelated_right).unwrap();
        let inputs = [
            (left, Tensor::new([2, 2], vec![1.0; 4]).unwrap()),
            (right, Tensor::new([2, 2], vec![2.0; 4]).unwrap()),
        ];

        let plan = prepare_graph(&graph, result, &PureRocmAssessor).unwrap();
        validate_graph_inputs(&plan.nodes, &inputs).unwrap();

        assert_eq!(plan.nodes.len(), 3);
        assert_eq!(plan.use_counts, [1, 1, 1]);
        assert!(plan.index_by_value.contains_key(&result));
    }

    #[test]
    fn multi_output_preflight_unions_dependencies_and_pins_requested_values() {
        let mut graph = Graph::default();
        let input = graph.input([2]).unwrap();
        let first_constant = graph.constant(Tensor::new([2], vec![1.0; 2]).unwrap());
        let first = graph.add(input, first_constant).unwrap();
        let second_constant = graph.constant(Tensor::new([2], vec![2.0; 2]).unwrap());
        let second = graph.mul(first, second_constant).unwrap();

        let plan = prepare_graph_outputs_plan(&graph, &[second, first], &PureRocmAssessor)
            .expect("both output dependency closures are supported");

        assert_eq!(plan.nodes.len(), 5);
        assert_eq!(plan.nodes.last().unwrap().value, second);
        let first_index = plan.index_by_value[&first];
        assert_eq!(plan.use_counts[first_index], 2);
        let output_order = [second, first];
        assert_eq!(output_position(&output_order, first), Some(1));
        assert_eq!(output_position(&output_order, second), Some(0));
        assert_eq!(output_position(&output_order, input), None);
    }

    #[test]
    fn multi_output_preflight_rejects_empty_and_duplicate_outputs() {
        let mut graph = Graph::default();
        let input = graph.input([2]).unwrap();

        assert!(matches!(
            prepare_graph_outputs_plan(&graph, &[], &PureRocmAssessor),
            Err(RocmTensorExecutionError::EmptyOutputs)
        ));
        assert!(matches!(
            prepare_graph_outputs_plan(&graph, &[input, input], &PureRocmAssessor),
            Err(RocmTensorExecutionError::DuplicateOutput(value)) if value == input
        ));
    }

    #[test]
    fn preflight_accepts_supported_mse_reduction() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let output = graph.mean_squared_error(left, right).unwrap();

        let prepared = prepare_graph(&graph, output, &PureRocmAssessor).unwrap();
        assert_eq!(prepared.nodes.last().unwrap().value, output);
        assert_eq!(prepared.nodes.last().unwrap().shape, &[]);
    }

    #[test]
    fn mse_scratch_capacity_covers_largest_reduction_in_multi_output_plan() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3]).unwrap();
        let right = graph.input([2, 3]).unwrap();
        let first_loss = graph.mean_squared_error(left, right).unwrap();
        let next_left = graph.input([4]).unwrap();
        let next_right = graph.input([4]).unwrap();
        let second_loss = graph.mean_squared_error(next_left, next_right).unwrap();
        let plan =
            prepare_graph_outputs_plan(&graph, &[first_loss, second_loss], &PureRocmAssessor)
                .unwrap();

        assert_eq!(mse_scratch_element_count(&graph, &plan.nodes).unwrap(), 6);
    }

    #[test]
    fn mse_scratch_validates_buffer_capacity_in_bytes() {
        assert!(mse_scratch_lengths_fit(3, 12, 12).unwrap());
        assert!(!mse_scratch_lengths_fit(3, 11, 12).unwrap());
        assert!(!mse_scratch_lengths_fit(3, 12, 11).unwrap());
        assert!(matches!(
            mse_scratch_lengths_fit(usize::MAX, usize::MAX, usize::MAX),
            Err(RocmTensorExecutionError::SizeOverflow)
        ));
    }

    #[test]
    fn sgd_update_preflight_is_native_and_tracks_both_operands() {
        let mut graph = Graph::default();
        let weights = graph.input([3]).unwrap();
        let gradient = graph.input([3]).unwrap();
        let updated = graph.sgd_update(weights, gradient, 0.25).unwrap();

        let plan = prepare_graph(&graph, updated, &PureRocmAssessor).unwrap();
        let node = plan.nodes.last().unwrap();
        assert!(
            matches!(node.op, fusion_pcu_tensor::OpDescriptor::SgdUpdate {
            weights: value_weights, gradient: value_gradient, learning_rate
        } if value_weights == weights
            && value_gradient == gradient
            && (learning_rate - 0.25).abs() < f32::EPSILON)
        );
        assert_eq!(plan.use_counts, [1, 1, 1]);
        assert_eq!(
            assess_tensor_node(&graph, *node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Native,
                workspace_bytes: Some(0),
            }
        );
    }

    #[test]
    fn scratch_retains_constants_and_non_output_computations_only() {
        let mut graph = Graph::default();
        let input = graph.input([2]).unwrap();
        let constant = graph.constant(Tensor::new([2], vec![1.0; 2]).unwrap());
        let computed = graph.add(input, constant).unwrap();
        let input_node = graph.nodes().find(|node| node.value == input).unwrap();
        let constant_node = graph.nodes().find(|node| node.value == constant).unwrap();
        let computed_node = graph.nodes().find(|node| node.value == computed).unwrap();

        assert!(!scratch_stores_node(input_node.op, input, &[computed]));
        assert!(scratch_stores_node(constant_node.op, constant, &[computed]));
        assert!(!scratch_stores_node(
            computed_node.op,
            computed,
            &[computed]
        ));
        assert!(scratch_stores_node(computed_node.op, computed, &[input]));
    }

    #[test]
    fn preflight_checks_all_reachable_inputs_before_execution() {
        let mut graph = Graph::default();
        let left = graph.input([2, 2]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let output = graph.matmul(left, right).unwrap();
        let inputs = [(left, Tensor::new([2, 2], vec![1.0; 4]).unwrap())];

        assert!(matches!(
            prepare_graph(&graph, output, &PureRocmAssessor)
                .and_then(|plan| validate_graph_inputs(&plan.nodes, &inputs).map_err(Into::into)),
            Err(RocmTensorExecutionError::Graph(TensorError::MissingInput(id))) if id == right
        ));
    }

    #[test]
    fn preflight_rejects_wrong_shaped_input() {
        let mut graph = Graph::default();
        let input = graph.input([2, 2]).unwrap();
        let inputs = [(input, Tensor::new([4], vec![1.0; 4]).unwrap())];

        assert!(matches!(
            prepare_graph(&graph, input, &PureRocmAssessor).and_then(|plan| validate_graph_inputs(
                &plan.nodes,
                &inputs
            )
            .map_err(Into::into)),
            Err(RocmTensorExecutionError::Graph(
                TensorError::ShapeMismatch { .. }
            ))
        ));
    }

    #[test]
    fn assessor_selects_library_for_dense_rank_two_matmul() {
        let mut graph = Graph::default();
        let left = graph.input([2, 3]).unwrap();
        let right = graph.input([3, 4]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Library,
                workspace_bytes: None,
            }
        );
    }

    #[test]
    fn assessor_synthesizes_add_relu_and_mse() {
        let mut graph = Graph::default();
        let first = graph.input([2, 2]).unwrap();
        let second = graph.input([2, 2]).unwrap();
        let add = graph.add(first, second).unwrap();
        let relu = graph.relu(first).unwrap();
        let loss = graph.mean_squared_error(first, second).unwrap();

        let add_node = graph.nodes().find(|node| node.value == add).unwrap();
        assert_eq!(
            assess_tensor_node(&graph, add_node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        );

        let relu_node = graph.nodes().find(|node| node.value == relu).unwrap();
        assert_eq!(
            assess_tensor_node(&graph, relu_node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: Some(0),
            }
        );

        let node = graph.nodes().find(|node| node.value == loss).unwrap();
        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Synthesized,
                workspace_bytes: None,
            }
        );
    }

    #[test]
    fn assessor_rejects_zero_sized_matmul_output() {
        let mut graph = Graph::default();
        let left = graph.input([0, 3]).unwrap();
        let right = graph.input([3, 2]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        );
    }

    #[test]
    fn assessor_rejects_zero_inner_dimension_even_with_nonzero_output() {
        let mut graph = Graph::default();
        let left = graph.input([2, 0]).unwrap();
        let right = graph.input([0, 3]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        );
    }

    #[test]
    fn assessor_rejects_dimensions_outside_rocblas_integer_range() {
        let mut graph = Graph::default();
        let left = graph.input([i32::MAX as usize + 1, 1]).unwrap();
        let right = graph.input([1, 1]).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let node = graph.nodes().find(|node| node.value == result).unwrap();

        assert_eq!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Shape,
            }
        );
    }
}
