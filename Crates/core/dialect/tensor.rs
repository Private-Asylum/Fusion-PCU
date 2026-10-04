//! Feature-gated heterogeneous tensor graphs and typed host storage.
//!
//! Tensor semantics remain isolated from generic coprocessor contracts. This module uses
//! `core` and `alloc`, never `std`. Graph inputs, constants, and uniforms retain scalar identity
//! in one erased graph; [`TensorValueId`](crate::dialect::tensor::TensorValueId) provides checked typed handles, with no implicit type
//! promotion. Dense host storage round-trips each [`TensorValue`](crate::dialect::tensor::TensorValue) scalar variant, but that fact
//! does not claim that every executor implements its type or arithmetic. The reference evaluator
//! transports typed leaf values and implements checked native/wide integer and F16/BF16/FP8/F32/F64 elementwise arithmetic, `ReLU`,
//! and strict ordered matrix multiplication. The legacy boundary-mode reference compounds
//! retain raw arithmetic; they are not evidence of checked conformance. Use
//! [`Graph::evaluate_checked`] or [`TensorCheckedReferenceAssessor`] for checked admission.
//! Mean-squared error and graph reverse-mode differentiation support f32/f64;
//! SGD supports f32/f64 with a finite f32 rate; the legacy evaluated-value gradient API remains f32-only. Packed arithmetic has no assigned
//! semantics. Broadcasting, batching, convolution, views, optimizers, and serialization remain
//! outside this dialect.

#[rustfmt::skip]
use alloc::{
    string::String,
    vec,
    vec::Vec,
};
#[rustfmt::skip]
use crate::{
    PcuScalar,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuNumericalRequirement,
    PcuReproducibility,
    core::PcuScalarType,
};
use core::fmt;
use core::sync::atomic::Ordering;
#[cfg(target_has_atomic = "64")]
use core::sync::atomic::AtomicU64 as GraphIdentityCounter;
#[cfg(all(not(target_has_atomic = "64"), target_has_atomic = "32"))]
use core::sync::atomic::AtomicU32 as GraphIdentityCounter;
#[cfg(not(any(target_has_atomic = "64", target_has_atomic = "32")))]
compile_error!("PCU tensor graph identities require 32-bit or 64-bit atomics");

#[path = "tensor/constants/constants.rs"]
pub mod constants;

#[path = "tensor/storage.rs"]
mod storage;
#[rustfmt::skip]
pub use storage::{
    TerminalBinaryDonorProof,
    TensorBinaryOperand,
    TensorBinaryOperation,
    TensorGraphRequirements,
    TensorInputReuseProof,
    TensorScratchStorageAssignment,
    TensorScratchStoragePlan,
    TensorScratchStorageSlot,
    TensorStorageConstraint,
    TensorStorageReuseError,
    TensorStorageValidationError,
    TensorValueLiveness,
    TensorValueRequirement,
    TensorValueStorageRequirement,
};
#[rustfmt::skip]
use storage::{
    dense_scalar_layout,
    node_output_bytes,
};

#[path = "tensor/feedback.rs"]
mod feedback;
#[rustfmt::skip]
pub use feedback::{
    TensorFeedbackBinding,
    TensorFeedbackInput,
    TensorFeedbackPlan,
};

#[path = "tensor/value.rs"]
mod value;
#[rustfmt::skip]
pub use value::{
    TensorElement,
    TensorScalarValue,
    TensorValue,
    TensorValueTypeMismatch,
};

#[path = "tensor/execution.rs"]
mod execution;
pub use execution::TensorExecution;

#[path = "tensor/reference.rs"]
mod reference;
#[rustfmt::skip]
use reference::{
    binary_value,
    execution_f32_value,
    matmul_value,
    mean_squared_error_value,
    relu_backward_value,
    relu_value,
    sgd_value,
};

#[path = "tensor/gradients/gradients.rs"]
mod gradients;

static NEXT_GRAPH_ID: GraphIdentityCounter = GraphIdentityCounter::new(1);

fn next_graph_id(counter: &GraphIdentityCounter) -> Result<u64, TensorError> {
    let identity = counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
            value.checked_add(1)
        })
        .map_err(|_| TensorError::GraphIdentityExhausted)?;
    #[cfg(target_has_atomic = "64")]
    {
        Ok(identity)
    }
    #[cfg(not(target_has_atomic = "64"))]
    {
        Ok(u64::from(identity))
    }
}

#[derive(Clone, Debug)]
pub struct Tensor<T: PcuScalar = f32> {
    shape: Vec<usize>,
    data: Vec<T>,
    known_uniform_value: Option<T>,
}

impl<T: PcuScalar + PartialEq> PartialEq for Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl<T: PcuScalar> Tensor<T> {
    /// Creates a tensor after checking that its data length matches its shape.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeOverflow`] if the element count overflows, or
    /// [`TensorError::DataLength`] if `data` has the wrong number of elements.
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<T>) -> Result<Self, TensorError> {
        let shape = shape.into();
        let len = element_count(&shape)?;
        if len != data.len() {
            return Err(TensorError::DataLength {
                expected: len,
                actual: data.len(),
            });
        }
        Ok(Self {
            shape,
            data,
            known_uniform_value: None,
        })
    }

    /// Creates a tensor whose elements are all `value` and records that fact for dialect
    /// optimizations that can use uniform data without rescanning a potentially large buffer.
    ///
    /// The returned tensor still owns ordinary contiguous data, so evaluators and backends see
    /// the same representation and upload behavior as for [`Self::new`]. Empty tensors have no
    /// known element value.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeOverflow`] if the element count overflows.
    pub fn splat(shape: impl Into<Vec<usize>>, value: T) -> Result<Self, TensorError> {
        let shape = shape.into();
        let len = element_count(&shape)?;
        Ok(Self {
            shape,
            data: vec![value; len],
            known_uniform_value: (len != 0).then_some(value),
        })
    }

    #[must_use]
    pub fn scalar(value: T) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
            known_uniform_value: Some(value),
        }
    }
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Scalar element type associated with this sealed host-storage type.
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        T::TYPE
    }

    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Returns a uniform element value when the tensor was constructed with a uniformity-aware
    /// constructor. General tensors created with [`Self::new`] remain unknown until an
    /// optimization elects to inspect their data.
    #[must_use]
    pub const fn known_uniform_value(&self) -> Option<T> {
        self.known_uniform_value
    }

    /// Consumes the tensor and returns its contiguous data without copying it.
    #[must_use]
    pub fn into_data(self) -> Vec<T> {
        self.data
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    /// Checked scalar arithmetic failed at the first affected tensor element.
    /// A cold arithmetic constant uses element zero of its originating value.
    ArithmeticFault {
        value: ValueId,
        element_index: usize,
        kind: PcuExecutionFaultKind,
    },
    /// A strict compound operation failed before publishing its output.
    CompoundArithmeticFault {
        value: ValueId,
        /// Row-major output cell index.
        element_index: usize,
        /// Increasing reduction index of the first failing operation; zero for elementwise SGD.
        reduction_index: usize,
        step: TensorArithmeticStep,
        kind: PcuExecutionFaultKind,
    },
    /// The reference route has not implemented this compound numerical contract.
    UnsupportedNumericalMode {
        value: ValueId,
        mode: PcuNumericalMode,
    },
    /// The evaluator has no certified implementation for these independent requirements.
    UnsupportedNumericalOptions {
        value: ValueId,
        options: PcuNumericalOptions,
    },
    ShapeOverflow,
    /// The process exhausted its monotonic graph identity domain; IDs never wrap or repeat.
    GraphIdentityExhausted,
    InvalidStorageAlignment {
        value: ValueId,
        alignment_bytes: usize,
    },
    UnsupportedScalarType {
        value: ValueId,
        scalar_type: PcuScalarType,
    },
    ScalarTypeMismatch {
        value: ValueId,
        expected: PcuScalarType,
        actual: PcuScalarType,
    },
    DataLength {
        expected: usize,
        actual: usize,
    },
    UnknownValue(ValueId),
    ShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    MatMulShape {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    LossMustBeScalar(Vec<usize>),
    MissingInput(ValueId),
    DuplicateInput(ValueId),
    ExtraInput(ValueId),
    EmptyOutputs,
    DuplicateOutput(ValueId),
    FeedbackRequiresMultipleBanks(usize),
    WrongGraph,
    GraphChanged,
    InvalidSeed,
    InvalidLearningRate,
    GradientRootNotMse(ValueId),
    UnsupportedGradient(ValueId),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl core::error::Error for TensorError {}

fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    shape.iter().try_fold(1usize, |n, &d| {
        n.checked_mul(d).ok_or(TensorError::ShapeOverflow)
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ValueId {
    graph_id: u64,
    index: usize,
}

/// A graph value whose scalar representation was checked against `T` when the handle was made.
///
/// The graph ID and scalar type stay private. Values can only be created by typed graph methods
/// or by [`Graph::typed_view`], which checks the referenced node before constructing this handle.
#[derive(Clone, Copy, Debug)]
pub struct TensorValueId<T: PcuScalar> {
    value: ValueId,
    marker: core::marker::PhantomData<fn() -> T>,
}

impl<T: PcuScalar> PartialEq for TensorValueId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: PcuScalar> Eq for TensorValueId<T> {}

impl<T: PcuScalar> core::hash::Hash for TensorValueId<T> {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::hash::Hash::hash(&self.value, state);
    }
}

impl<T: PcuScalar> TensorValueId<T> {
    const fn new(value: ValueId) -> Self {
        Self {
            value,
            marker: core::marker::PhantomData,
        }
    }

    /// Erases the compile-time scalar marker for dynamic graph/model boundaries.
    #[must_use]
    pub const fn erase(self) -> ValueId {
        self.value
    }

    /// Scalar type proven when this handle was constructed.
    #[must_use]
    pub const fn scalar_type(self) -> PcuScalarType {
        T::TYPE
    }
}

/// A read-only description of one operation in a graph.
///
/// Constants and uniform scalars are borrowed from the graph, so inspecting a plan does not
/// clone tensor data or copy wide scalar payloads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpDescriptor<'a> {
    Input,
    Constant(&'a TensorValue),
    /// A compact graph-level constant whose logical output is a dense tensor filled with `value`.
    /// Storage and liveness metadata continue to describe the full logical tensor extent.
    Uniform {
        value: &'a TensorScalarValue,
    },
    Add {
        left: ValueId,
        right: ValueId,
    },
    Sub {
        left: ValueId,
        right: ValueId,
    },
    Mul {
        left: ValueId,
        right: ValueId,
    },
    Div {
        left: ValueId,
        right: ValueId,
    },
    SgdUpdate {
        weights: ValueId,
        gradient: ValueId,
        learning_rate: f32,
    },
    MatMul {
        left: ValueId,
        right: ValueId,
        transpose_left: bool,
        transpose_right: bool,
    },
    Relu {
        input: ValueId,
    },
    ReluBackward {
        input: ValueId,
        upstream: ValueId,
    },
    MeanSquaredError {
        prediction: ValueId,
        target: ValueId,
    },
}

/// Constituent arithmetic boundary inside an ordered compound operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorArithmeticStep {
    Multiply,
    Add,
    Subtract,
    Divide,
}

#[path = "tensor/fault_domain/fault_domain.rs"]
mod fault_domain;
#[rustfmt::skip]
pub use fault_domain::{
    TensorStrictFaultDomain,
    TensorStrictFaultLocation,
};

/// Read-only metadata for one graph value, yielded in stable append order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeDescriptor<'a> {
    pub value: ValueId,
    pub op: OpDescriptor<'a>,
    pub shape: &'a [usize],
    pub scalar_type: PcuScalarType,
    /// Exception detection granularity for compound nodes; absent for scalar operations.
    pub numerical_mode: Option<PcuNumericalMode>,
    /// Independent arithmetic/precision/reproducibility requirements frozen at capture.
    pub numerical_options: PcuNumericalOptions,
    /// Independent checked-float underflow policy for binary and compound nodes.
    pub float_underflow_policy: Option<PcuFloatUnderflowPolicy>,
}

/// Execution route selected by an explicit tensor operation assessor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorExecutionRoute {
    Native,
    Library,
    Synthesized,
    Reference,
}

/// Physical representation offered for an input operand of a tensor operation.
///
/// This does not change the operand's logical shape or the graph's value semantics. An adapter
/// must affirm a compact representation for every consumer before allocating less than the
/// logical tensor extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorOperandRepresentation {
    Dense,
    UniformScalar,
}

/// Structured reason why an assessor cannot support a graph operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorUnsupportedReason {
    Operation,
    Layout,
    ElementType,
    Shape,
    /// A numerical implementation cannot honor this independently selected underflow policy.
    UnderflowPolicy(PcuFloatUnderflowPolicy),
    /// No implementation satisfies this numerical requirement and option combination.
    NumericalPolicy {
        requirement: PcuNumericalRequirement,
        options: PcuNumericalOptions,
    },
    Other(String),
}

/// Backend or adapter decision for one borrowed graph node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorOperationSupport {
    Supported {
        route: TensorExecutionRoute,
        /// Required scratch workspace when known. `None` means the assessor did not report it.
        workspace_bytes: Option<usize>,
    },
    Unsupported {
        reason: TensorUnsupportedReason,
    },
}

/// Adapter contract for assessing one operation on an explicitly selected target.
pub trait TensorOperationAssessor {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport;

    /// Whether `node` can consume `operand` in the given physical representation.
    ///
    /// Dense operands retain the existing assessment contract. Compact forms require an
    /// explicit backend affirmation; the conservative default prevents accidental compression.
    fn supports_operand_representation(
        &self,
        _graph: &Graph,
        _node: NodeDescriptor<'_>,
        _operand: ValueId,
        representation: TensorOperandRepresentation,
    ) -> bool {
        representation == TensorOperandRepresentation::Dense
    }
}

/// Synchronous execution of one dense row-major f32 matrix multiplication.
///
/// This narrow operation contract lets a selected backend prove a tensor-library route before a
/// general graph executor exists. `a` has shape `[rows, inner]`, `b` has shape
/// `[inner, columns]`, and `c` has shape `[rows, columns]`. The implementation checks dimensions,
/// byte extents, access permissions, aliasing, and device/session identity before touching memory.
///
/// # Safety
///
/// Implementors must establish that device access to all three resources has ended before this
/// method returns, on success and on every error path. If completion cannot be established, they
/// must retain every possibly accessed resource and relevant runtime objects for as long as work
/// might remain in flight.
pub unsafe trait TensorSynchronousF32MatMulBackend: TensorOperationAssessor {
    type Resource;
    type Error;

    /// Computes `c = a * b`, with no implicit host transfers or fallback.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes/resources or backend execution failure.
    fn matmul_row_major(
        &self,
        a: &Self::Resource,
        b: &Self::Resource,
        c: &Self::Resource,
        rows: usize,
        inner: usize,
        columns: usize,
    ) -> Result<(), Self::Error>;
}

/// Explicit assessor for this crate's deterministic host reference evaluator.
///
/// Boundary-mode compounds retain historical raw arithmetic in this legacy route. Support
/// here describes reference executability, not checked numerical conformance. For a checked
/// reference execution use [`Graph::evaluate_checked`].
///
/// Applications must select this route deliberately; assessing another backend never causes an
/// automatic retry through the reference evaluator. Its temporary workspace is not yet bounded.
pub struct TensorReferenceAssessor;

impl TensorOperationAssessor for TensorReferenceAssessor {
    fn assess_node(&self, _graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::NumericalPolicy {
                    requirement: PcuNumericalRequirement::Reproducibility,
                    options: node.numerical_options,
                },
            };
        }
        if node.numerical_mode == Some(PcuNumericalMode::Strict)
            && !matches!(
                node.op,
                OpDescriptor::MatMul { .. }
                    | OpDescriptor::SgdUpdate { .. }
                    | OpDescriptor::MeanSquaredError { .. }
                    | OpDescriptor::ReluBackward { .. }
            )
        {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation,
            };
        }
        let leaf = matches!(
            node.op,
            OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
        );
        let integer_binary = matches!(
            node.op,
            OpDescriptor::Add { .. } | OpDescriptor::Sub { .. } | OpDescriptor::Mul { .. }
        ) && matches!(
            node.scalar_type,
            PcuScalarType::U8
                | PcuScalarType::U16
                | PcuScalarType::U32
                | PcuScalarType::U64
                | PcuScalarType::I8
                | PcuScalarType::I16
                | PcuScalarType::I32
                | PcuScalarType::I64
                | PcuScalarType::U128
                | PcuScalarType::I128
                | PcuScalarType::U256
                | PcuScalarType::I256
                | PcuScalarType::U512
                | PcuScalarType::I512
        );
        let low_elementwise = matches!(
            node.scalar_type,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
        ) && matches!(
            node.op,
            OpDescriptor::Add { .. }
                | OpDescriptor::Sub { .. }
                | OpDescriptor::Mul { .. }
                | OpDescriptor::Div { .. }
                | OpDescriptor::Relu { .. }
                | OpDescriptor::ReluBackward { .. }
        );
        if leaf
            || integer_binary
            || low_elementwise
            || matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64)
        {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Reference,
                workspace_bytes: None,
            }
        } else {
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::ElementType,
            }
        }
    }
}

/// Assessor admitting only implemented checked host-reference numerical contracts.
///
/// Unlike the legacy reference assessor, this rejects raw boundary compounds. Selection of
/// this assessor never triggers implicit fallback from another backend.
/// Backend-defined arithmetic and optimized precision are permissions, so the exact checked
/// reference route may retain its stronger ordered, precision-preserving behavior. Portable
/// reproducibility still needs its own independently specified executor profile.
pub struct TensorCheckedReferenceAssessor;

impl TensorOperationAssessor for TensorCheckedReferenceAssessor {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
            return TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::NumericalPolicy {
                    requirement: PcuNumericalRequirement::Reproducibility,
                    options: node.numerical_options,
                },
            };
        }
        let unsupported_compound = node.numerical_mode.is_some_and(|mode| {
            !matches!(node.op, OpDescriptor::ReluBackward { .. })
                && (mode != PcuNumericalMode::Strict
                    || !matches!(
                        node.op,
                        OpDescriptor::MatMul { .. }
                            | OpDescriptor::SgdUpdate { .. }
                            | OpDescriptor::MeanSquaredError { .. }
                            | OpDescriptor::ReluBackward { .. }
                    ))
        });
        if unsupported_compound {
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation,
            }
        } else {
            TensorReferenceAssessor.assess_node(graph, node)
        }
    }
}

/// One node's selected-target assessment, retaining a borrowed descriptor and shape.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorNodeAssessment<'a> {
    pub node: NodeDescriptor<'a>,
    pub support: TensorOperationSupport,
    /// Required bytes for this node's output, if representable.
    pub output_bytes: Option<usize>,
}

/// Assessment for every graph node in stable append order.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorGraphAssessment<'a> {
    pub nodes: Vec<TensorNodeAssessment<'a>>,
}

impl TensorGraphAssessment<'_> {
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.nodes
            .iter()
            .all(|node| matches!(node.support, TensorOperationSupport::Supported { .. }))
    }

    pub fn unsupported_nodes(&self) -> impl Iterator<Item = &TensorNodeAssessment<'_>> {
        self.nodes
            .iter()
            .filter(|node| matches!(node.support, TensorOperationSupport::Unsupported { .. }))
    }
}

/// Stable CPU reference execution order and checked storage/liveness facts for a graph.
///
/// Nodes are currently appended only after their operands, so append order is a valid stable
/// topological order. Lifetimes are inclusive: an operand remains live through its final reader.
/// `peak_live_bytes` is `None` when any value extent or accumulation overflows `usize`.
#[derive(Clone, Debug)]
pub struct TensorExecutionPlan<'a> {
    graph: &'a Graph,
    node_order: Vec<ValueId>,
    input_values: Vec<ValueId>,
    output_values: Vec<ValueId>,
    value_liveness: Vec<TensorValueLiveness>,
    index_by_value: Vec<usize>,
    use_counts: Vec<usize>,
    peak_live_bytes: Option<usize>,
}

/// Controls whether a backend-neutral lowering plan may replace a multiply followed by
/// subtraction with the fused SGD operation.
///
/// `AllowContractedArithmetic` explicitly permits the target to contract `weight - rate * grad`
/// and therefore change intermediate f32 rounding. It never changes graph construction or the
/// CPU reference evaluator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TensorArithmeticRewritePolicy {
    /// Preserve the graph's operation boundaries in every lowering plan.
    #[default]
    Disabled,
    /// Permit SGD recognition when the selected target declares support for contracted arithmetic.
    AllowContractedArithmetic,
}

/// Numerical behavior the selected target explicitly accepts for one lowered operation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TensorArithmeticCapability {
    /// The target has not proved that contracted arithmetic is acceptable.
    #[default]
    Strict,
    /// The selected target accepts a fused multiply-add with one final rounding.
    ContractedMultiplyAdd,
}

/// Controls whether selected-plan metadata may group a narrow pointwise operation chain.
///
/// This policy is independent from arithmetic rewriting: grouping Add followed by `ReLU` retains
/// the Add rounding point and does not permit reassociation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TensorPointwiseGroupingPolicy {
    /// Keep each selected graph operation as a separate scheduled operation.
    #[default]
    Disabled,
    /// Group a same-shape Add -> `ReLU` chain when Add has exactly one reader, that `ReLU`.
    SingleUseAddRelu,
    /// Group a same-shape Add/Sub expression with one to eight single-use arithmetic nodes and
    /// at most four external operands, followed by a terminal `ReLU`.
    BoundedAddSubRelu,
    /// Group a same-shape Add/Sub expression with one to eight single-use arithmetic nodes and
    /// at most four external operands, retaining the terminal arithmetic result as its output.
    BoundedAddSubIdentity,
    /// Group a same-shape Mul expression with one to eight single-use nodes and at most four
    /// external operands, retaining its terminal multiplication result.
    BoundedMulIdentity,
}

/// Epilogue applied after an ordered Add/Sub pointwise expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorPointwiseEpilogue {
    /// Return the final arithmetic result without another operation.
    Identity,
    /// Apply `max(value, 0)` after the final arithmetic result.
    Relu,
}

/// Arithmetic operation in a bounded pointwise expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorPointwiseArithmeticOp {
    /// IEEE-754 addition in source order.
    Add,
    /// IEEE-754 subtraction in source order.
    Sub,
}

/// Operand reference in a bounded pointwise expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorPointwiseOperand {
    /// Index into [`TensorBoundedPointwiseFusionGroup::leaves`].
    Leaf(usize),
    /// Index into the preceding entries in [`TensorBoundedPointwiseFusionGroup::steps`].
    Step(usize),
}

/// One ordered arithmetic instruction in a bounded pointwise expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorPointwiseStep {
    /// Result produced by the corresponding source graph operation.
    pub output: ValueId,
    /// Source arithmetic operation.
    pub op: TensorPointwiseArithmeticOp,
    /// First source operand, preserving operand order.
    pub left: TensorPointwiseOperand,
    /// Second source operand, preserving operand order.
    pub right: TensorPointwiseOperand,
}

/// Bounded straight-line Add/Sub expression with an explicit terminal epilogue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorBoundedPointwiseFusionGroup {
    /// Value produced by the selected terminal operation.
    pub output: ValueId,
    /// Operation applied after the ordered arithmetic steps.
    pub epilogue: TensorPointwiseEpilogue,
    /// Unique external values in stable first-use order. Uniforms remain ordinary leaves.
    pub leaves: Vec<ValueId>,
    /// Arithmetic nodes in source topological order. Each node keeps its original operand order.
    pub steps: Vec<TensorPointwiseStep>,
    /// Common logical shape of every leaf, arithmetic node, and terminal `ReLU`.
    pub shape: Vec<usize>,
}

/// One ordered multiplication instruction in a bounded Mul expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorPointwiseMulStep {
    /// Result produced by the corresponding source graph operation.
    pub output: ValueId,
    /// First source operand.
    pub left: TensorPointwiseOperand,
    /// Second source operand.
    pub right: TensorPointwiseOperand,
}

/// Bounded straight-line Mul expression retaining the terminal result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorBoundedMulFusionGroup {
    /// Value produced by the selected terminal operation.
    pub output: ValueId,
    /// Unique external values in stable first-use order. Uniforms remain ordinary leaves.
    pub leaves: Vec<ValueId>,
    /// Multiplication nodes in source topological order.
    pub steps: Vec<TensorPointwiseMulStep>,
    /// Common logical shape of every leaf and multiplication node.
    pub shape: Vec<usize>,
}

/// Source provenance for a grouped Add -> `ReLU` selected operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorPointwiseFusionGroup {
    /// Add result suppressed as a separate scheduled operation.
    pub add_output: ValueId,
    /// `ReLU` result and selected output of the group.
    pub relu_output: ValueId,
    /// First Add operand.
    pub left: ValueId,
    /// Second Add operand.
    pub right: ValueId,
    /// Common logical shape of both Add inputs, Add result, and `ReLU` result.
    pub shape: Vec<usize>,
}

/// An operation in an opt-in selected lowering schedule.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorSelectedOperation<'a> {
    /// A selected source operation, including any separately authorized arithmetic rewrite.
    Node(NodeDescriptor<'a>),
    /// Add followed by `ReLU`. The Add result does not require a scheduled output allocation.
    FusedAddRelu {
        add_output: ValueId,
        relu_output: ValueId,
        left: ValueId,
        right: ValueId,
        shape: &'a [usize],
    },
    /// A bounded ordered Add/Sub expression with an identity or `ReLU` epilogue.
    FusedAddSub {
        /// Group description with source provenance and external operands.
        group: TensorBoundedPointwiseFusionGroup,
    },
    /// A bounded same-shape Mul expression retaining the source operation order.
    FusedMul { group: TensorBoundedMulFusionGroup },
}

/// Lifetime-free selected operation stored by an owned tensor program.
///
/// Source nodes remain in the owned [`Graph`] and are addressed by `ValueId`, so constants and
/// shapes are not duplicated when a reusable program is prepared.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorOwnedSelectedOperation {
    /// A selected source graph node.
    Node { value: ValueId },
    /// Add followed by `ReLU`; the Add intermediate has no separate allocation.
    FusedAddRelu {
        add_output: ValueId,
        relu_output: ValueId,
        left: ValueId,
        right: ValueId,
    },
    /// A bounded ordered Add/Sub chain and optional `ReLU` epilogue.
    FusedAddSub {
        group: TensorBoundedPointwiseFusionGroup,
    },
    /// A bounded ordered Mul chain.
    FusedMul { group: TensorBoundedMulFusionGroup },
}

/// Owned graph plus its selected dependency and lowering metadata.
///
/// Unlike [`TensorExecutionPlan`] and [`TensorSelectedLoweringPlan`], this value has no borrows
/// into the graph it retains. It is suitable as the immutable source for a reusable backend
/// prepared template. The graph is consumed to preserve its `ValueId` identity without cloning or
/// remapping constants.
pub struct TensorOwnedSelectedProgram {
    graph: Graph,
    node_order: Vec<ValueId>,
    input_values: Vec<ValueId>,
    output_values: Vec<ValueId>,
    value_liveness: Vec<TensorValueLiveness>,
    peak_live_bytes: Option<usize>,
    source_storage_constraints: Vec<TensorStorageConstraint>,
    selected_nodes: Vec<ValueId>,
    selected_use_counts: Vec<usize>,
    rewrites: Vec<TensorSgdRewriteCandidate>,
    suppressed_values: Vec<ValueId>,
    operations: Vec<TensorOwnedSelectedOperation>,
    operation_index_by_value: Vec<usize>,
    operation_use_counts: Vec<usize>,
    operation_liveness: Vec<TensorValueLiveness>,
    operation_read_only: Vec<bool>,
    operation_storage_constraints: Vec<TensorStorageConstraint>,
}

#[path = "tensor/fragment/fragment.rs"]
mod fragment;
#[rustfmt::skip]
pub use fragment::{
    TensorFragmentError,
    TensorFragmentInput,
    TensorOperationFragment,
};

/// An inspectable lowering-only replacement for `Sub(weights, Mul(rate, gradient))`.
///
/// Consuming a candidate authorizes a different floating-point result whenever contraction
/// changes rounding. It does not modify the graph or authorize in-place storage reuse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TensorSgdRewriteCandidate {
    /// Value produced by the subtraction in the original graph.
    pub output: ValueId,
    /// The multiplication intermediate that the lowering may omit.
    pub multiply: ValueId,
    pub weights: ValueId,
    pub gradient: ValueId,
    pub learning_rate: f32,
}

/// Backend-neutral, inspectable operation schedule selected for a tensor execution plan.
///
/// This is a description only. It does not mutate the graph or cause a backend to execute a
/// rewrite. Values and output pins remain identified by their original [`ValueId`]s.
#[derive(Clone, Debug)]
pub struct TensorSelectedLoweringPlan<'a> {
    graph_id: u64,
    nodes: Vec<NodeDescriptor<'a>>,
    input_values: Vec<ValueId>,
    output_values: Vec<ValueId>,
    index_by_value: Vec<usize>,
    use_counts: Vec<usize>,
    rewritten: Vec<TensorSgdRewriteCandidate>,
    suppressed_values: Vec<ValueId>,
    pointwise_groups: Vec<TensorPointwiseFusionGroup>,
    bounded_pointwise_groups: Vec<TensorBoundedPointwiseFusionGroup>,
    bounded_mul_groups: Vec<TensorBoundedMulFusionGroup>,
    operations: Vec<TensorSelectedOperation<'a>>,
    operation_index_by_value: Vec<usize>,
    operation_use_counts: Vec<usize>,
}

impl<'a> TensorSelectedLoweringPlan<'a> {
    /// Node descriptors after arithmetic rewriting and before optional pointwise grouping.
    ///
    /// For an execution schedule that applies pointwise grouping, use [`Self::operations`].
    #[must_use]
    pub fn nodes(&self) -> &[NodeDescriptor<'a>] {
        &self.nodes
    }

    /// Selected graph inputs, preserved from the source plan.
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_values
    }

    /// Requested outputs, preserved and pinned from the source plan.
    #[must_use]
    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }

    /// Operand-reader counts aligned with [`Self::nodes`], including output pins.
    #[must_use]
    pub fn use_counts(&self) -> &[usize] {
        &self.use_counts
    }

    /// Index in [`Self::nodes`], or `None` for values outside that schedule or removed by an
    /// arithmetic rewrite. For the opt-in pointwise schedule, use [`Self::operation_index_of`].
    #[must_use]
    pub fn index_of(&self, value: ValueId) -> Option<usize> {
        if value.graph_id != self.graph_id {
            return None;
        }
        self.index_by_value
            .get(value.index)
            .copied()
            .filter(|index| *index != usize::MAX)
    }

    /// Rewrites actually selected by the lowering policy.
    #[must_use]
    pub fn rewrites(&self) -> &[TensorSgdRewriteCandidate] {
        &self.rewritten
    }

    /// Values whose operation was removed from this schedule.
    #[must_use]
    pub fn suppressed_values(&self) -> &[ValueId] {
        &self.suppressed_values
    }

    /// Opt-in operation schedule, including selected pointwise groups.
    ///
    /// `nodes()` remains the arithmetic-selected graph-node view. This typed schedule is the
    /// execution view when pointwise grouping is enabled; its own use counts, indices, and
    /// storage requirements are exposed separately below.
    #[must_use]
    pub fn operations(&self) -> &[TensorSelectedOperation<'a>] {
        &self.operations
    }

    /// Pointwise groups selected by the independent grouping policy.
    #[must_use]
    pub fn pointwise_fusion_groups(&self) -> &[TensorPointwiseFusionGroup] {
        &self.pointwise_groups
    }

    /// Bounded ordered Add/Sub expressions with a terminal `ReLU` selected for this plan.
    #[must_use]
    pub fn bounded_pointwise_fusion_groups(&self) -> &[TensorBoundedPointwiseFusionGroup] {
        &self.bounded_pointwise_groups
    }

    /// Bounded ordered Mul expressions with an identity epilogue selected for this plan.
    #[must_use]
    pub fn bounded_mul_fusion_groups(&self) -> &[TensorBoundedMulFusionGroup] {
        &self.bounded_mul_groups
    }

    /// Reader counts aligned with [`Self::operations`], including requested-output pins.
    #[must_use]
    pub fn operation_use_counts(&self) -> &[usize] {
        &self.operation_use_counts
    }

    /// Index in [`Self::operations`] for a selected value. Grouped Add intermediates return None.
    #[must_use]
    pub fn operation_index_of(&self, value: ValueId) -> Option<usize> {
        if value.graph_id != self.graph_id {
            return None;
        }
        self.operation_index_by_value
            .get(value.index)
            .copied()
            .filter(|index| *index != usize::MAX)
    }

    /// Storage disjointness constraints for the typed operation schedule.
    ///
    /// The grouped Add intermediate is omitted and the fused operation reads both original Add
    /// operands, so this reflects the storage the selected schedule actually requires.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` if any scheduled value's byte extent cannot be represented.
    pub fn operation_storage_constraints(
        &self,
    ) -> Result<Vec<TensorStorageConstraint>, TensorError> {
        let mut last_use: Vec<usize> = (0..self.operations.len()).collect();
        for (position, operation) in self.operations.iter().enumerate() {
            for_each_selected_operand(operation, |operand| {
                if let Some(index) = self.operation_index_of(operand) {
                    last_use[index] = position;
                }
            });
        }
        let final_position = self.operations.len().saturating_sub(1);
        for output in &self.output_values {
            if let Some(index) = self.operation_index_of(*output) {
                last_use[index] = final_position;
            }
        }
        let mut constraints = Vec::new();
        for (left_index, left) in self.operations.iter().enumerate() {
            for (right_index, right) in self.operations.iter().enumerate().skip(left_index + 1) {
                let left_read_only = selected_operation_is_read_only(left);
                let right_read_only = selected_operation_is_read_only(right);
                if (last_use[left_index] < right_index && !left_read_only && !right_read_only)
                    || (left_read_only && right_read_only)
                {
                    continue;
                }
                let left_bytes = selected_operation_bytes(left, &self.nodes)?;
                let right_bytes = selected_operation_bytes(right, &self.nodes)?;
                constraints.push(TensorStorageConstraint {
                    left: selected_operation_output(left),
                    right: selected_operation_output(right),
                    left_bytes,
                    right_bytes,
                });
            }
        }
        Ok(constraints)
    }

    /// Plans scratch slots against this selected operation schedule's operand lifetimes.
    ///
    /// Liveness is recomputed from selected operations rather than filtered from source-graph
    /// liveness, because rewrites and fusion may change which operations read a value. The
    /// lowering adapter supplies only values it will materialize; read-only inputs/constants/
    /// uniforms and requested outputs are excluded here.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` when a selected value extent or total scratch capacity cannot be
    /// represented.
    pub fn scratch_storage_plan(
        &self,
        eligible_values: &[ValueId],
    ) -> Result<TensorScratchStoragePlan, TensorError> {
        if let Some(&value) = eligible_values
            .iter()
            .find(|value| value.graph_id != self.graph_id)
        {
            return Err(TensorError::UnknownValue(value));
        }
        let liveness = self.operation_liveness()?;
        let eligible: Vec<_> = self
            .operations
            .iter()
            .filter_map(|operation| {
                let value = selected_operation_output(operation);
                (eligible_values.contains(&value)
                    && !selected_operation_is_read_only(operation)
                    && !self.output_values.contains(&value))
                .then_some(value)
            })
            .collect();
        storage::plan_scratch_storage(&liveness, &eligible)
    }

    fn operation_liveness(&self) -> Result<Vec<TensorValueLiveness>, TensorError> {
        let mut last_use: Vec<usize> = (0..self.operations.len()).collect();
        for (position, operation) in self.operations.iter().enumerate() {
            for_each_selected_operand(operation, |operand| {
                if let Some(index) = self.operation_index_of(operand) {
                    last_use[index] = position;
                }
            });
        }
        let final_position = self.operations.len().saturating_sub(1);
        for &output in &self.output_values {
            if let Some(index) = self.operation_index_of(output) {
                last_use[index] = final_position;
            }
        }
        self.operations
            .iter()
            .enumerate()
            .map(|(position, operation)| {
                let output = selected_operation_output(operation);
                let scalar_type = self
                    .nodes
                    .iter()
                    .find(|node| node.value == output)
                    .map(|node| node.scalar_type)
                    .ok_or(TensorError::UnknownValue(output))?;
                Ok(TensorValueLiveness {
                    value: output,
                    scalar_type,
                    output_bytes: Some(checked_node_output_bytes(
                        output,
                        selected_operation_shape(operation),
                        scalar_type,
                    )?),
                    alignment_bytes: dense_scalar_layout(scalar_type)
                        .map(|(_, alignment)| alignment),
                    first_live_node: position,
                    last_live_node: last_use[position],
                })
            })
            .collect()
    }

    /// Storage disjointness required by the arithmetic-selected node schedule in [`Self::nodes`].
    ///
    /// Rewrites can extend an operand's lifetime as well as remove intermediates, so these
    /// constraints are derived from the selected operands and output pins rather than filtered
    /// from the source plan. For the optional pointwise schedule, use
    /// [`Self::operation_storage_constraints`].
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` when a surviving value's byte extent cannot be represented.
    pub fn storage_constraints(&self) -> Result<Vec<TensorStorageConstraint>, TensorError> {
        let mut last_use: Vec<usize> = (0..self.nodes.len()).collect();
        for (position, node) in self.nodes.iter().enumerate() {
            for_each_descriptor_operand(node.op, |operand| {
                if let Some(index) = self.index_of(operand) {
                    last_use[index] = position;
                }
            });
        }
        let final_position = self.nodes.len().saturating_sub(1);
        for output in &self.output_values {
            if let Some(index) = self.index_of(*output) {
                last_use[index] = final_position;
            }
        }
        let mut constraints = Vec::new();
        for (left_index, left) in self.nodes.iter().enumerate() {
            for (right_index, right) in self.nodes.iter().enumerate().skip(left_index + 1) {
                let left_read_only = is_read_only_descriptor(left.op);
                let right_read_only = is_read_only_descriptor(right.op);
                if (last_use[left_index] < right_index && !left_read_only && !right_read_only)
                    || (left_read_only && right_read_only)
                {
                    continue;
                }
                let left_bytes = u64::try_from(checked_node_output_bytes(
                    left.value,
                    left.shape,
                    left.scalar_type,
                )?)
                .map_err(|_| TensorError::ShapeOverflow)?;
                let right_bytes = u64::try_from(checked_node_output_bytes(
                    right.value,
                    right.shape,
                    right.scalar_type,
                )?)
                .map_err(|_| TensorError::ShapeOverflow)?;
                constraints.push(TensorStorageConstraint {
                    left: left.value,
                    right: right.value,
                    left_bytes,
                    right_bytes,
                });
            }
        }
        Ok(constraints)
    }
}

impl TensorOwnedSelectedProgram {
    /// Original owned graph. Values retain the graph IDs captured by the selected schedules.
    #[must_use]
    pub const fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Selected dependency closure in stable topological order.
    #[must_use]
    pub fn node_order(&self) -> &[ValueId] {
        &self.node_order
    }

    /// Input values required by the selected outputs.
    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_values
    }

    /// Requested and pinned outputs in caller order.
    #[must_use]
    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }

    /// Proves that a consumed external input can supply the storage for a selected `ReLU` output.
    ///
    /// This is deliberately limited to a direct, same-index `ReLU` of an external input. It
    /// rejects input fanout, output-pinned inputs, and `ReLU` outputs that feed another selected
    /// operation. The returned proof describes graph legality
    /// only; a backend must still prove physical exclusivity, quiescence, and an in-place binding
    /// contract before reusing a resource. Ordinary storage constraints remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns a typed reuse error when the selected graph does not meet the narrow contract, or
    /// a shape/layout error when dense extents cannot be represented.
    pub fn prove_consumed_relu_reuse(
        &self,
        input: ValueId,
        output: ValueId,
    ) -> Result<TensorInputReuseProof, TensorStorageReuseError> {
        if input.graph_id != self.graph.id || self.graph.nodes.get(input.index).is_none() {
            return Err(TensorStorageReuseError::ValueNotInProgram(input));
        }
        if output.graph_id != self.graph.id || self.graph.nodes.get(output.index).is_none() {
            return Err(TensorStorageReuseError::ValueNotInProgram(output));
        }
        if !self.input_values.contains(&input)
            || !matches!(self.graph.nodes[input.index].op, Op::Input)
        {
            return Err(TensorStorageReuseError::InputIsNotGraphInput(input));
        }
        let Some(output_operation_index) = self.operation_index_of(output) else {
            return Err(TensorStorageReuseError::OutputIsNotSelectedOutput(output));
        };
        if !self.output_values.contains(&output) {
            return Err(TensorStorageReuseError::OutputIsNotSelectedOutput(output));
        }
        if self.output_values.contains(&input) {
            return Err(TensorStorageReuseError::InputIsSelectedOutput(input));
        }

        let output_uses = self.operation_use_counts[output_operation_index];
        if output_uses != 1 {
            return Err(TensorStorageReuseError::OutputHasSelectedConsumers {
                value: output,
                actual: output_uses.saturating_sub(1),
            });
        }

        let Some(input_operation_index) = self.operation_index_of(input) else {
            return Err(TensorStorageReuseError::ValueNotInProgram(input));
        };
        let actual_uses = self.operation_use_counts[input_operation_index];
        if actual_uses != 1 {
            return Err(TensorStorageReuseError::InputUseCount {
                value: input,
                actual: actual_uses,
            });
        }

        if !matches!(self.graph.nodes[output.index].op, Op::Relu(actual) if actual == input)
            || !matches!(
                self.operations.get(output_operation_index),
                Some(TensorOwnedSelectedOperation::Node { value }) if *value == output
            )
        {
            return Err(TensorStorageReuseError::NotDirectRelu { input, output });
        }

        let input_node = &self.graph.nodes[input.index];
        let output_node = &self.graph.nodes[output.index];
        if input_node.shape != output_node.shape {
            return Err(TensorStorageReuseError::ShapeMismatch { input, output });
        }
        if input_node.scalar_type != output_node.scalar_type {
            return Err(TensorStorageReuseError::ScalarTypeMismatch { input, output });
        }
        let scalar_type = input_node.scalar_type;
        let Some((_, alignment_bytes)) = dense_scalar_layout(scalar_type) else {
            return Err(TensorStorageReuseError::UnsupportedScalarType {
                value: input,
                scalar_type,
            });
        };
        let input_bytes = checked_node_output_bytes(input, &input_node.shape, scalar_type)
            .map_err(|_| TensorStorageReuseError::ShapeOverflow(input))?;
        let output_bytes = checked_node_output_bytes(output, &output_node.shape, scalar_type)
            .map_err(|_| TensorStorageReuseError::ShapeOverflow(output))?;
        if input_bytes != output_bytes {
            return Err(TensorStorageReuseError::LayoutMismatch { input, output });
        }
        let bytes = u64::try_from(input_bytes)
            .map_err(|_| TensorStorageReuseError::ShapeOverflow(input))?;
        let alignment_bytes = u64::try_from(alignment_bytes)
            .map_err(|_| TensorStorageReuseError::ShapeOverflow(input))?;
        Ok(TensorInputReuseProof::new(
            self.graph.id,
            input,
            output,
            scalar_type,
            bytes,
            alignment_bytes,
        ))
    }

    /// Proves that a terminal Add, Sub, or Mul can overwrite one designated input.
    ///
    /// The selected program must contain exactly two distinct external inputs and one binary
    /// operation, with the result as its sole requested output. Both operands must have one
    /// selected use. This proves graph legality only; a backend must independently validate
    /// resource ownership, overlap, quiescence, and its in-place binding contract.
    ///
    /// # Errors
    ///
    /// Returns a typed reuse error when the selected graph does not meet this profile or its
    /// dense storage facts cannot be represented.
    pub fn prove_consumed_binary_donor(
        &self,
        donor: ValueId,
        other: ValueId,
        output: ValueId,
    ) -> Result<TerminalBinaryDonorProof, TensorStorageReuseError> {
        self.validate_binary_donor_selection(donor, other, output)?;
        let (operation, donor_operand) =
            binary_donor_operation(&self.graph.nodes[output.index].op, donor, other).ok_or(
                TensorStorageReuseError::NotTerminalBinary {
                    donor,
                    other,
                    output,
                },
            )?;
        let (scalar_type, bytes, alignment_bytes) =
            self.binary_donor_layout(donor, other, output)?;
        Ok(TerminalBinaryDonorProof::new(
            self.graph.id,
            donor,
            other,
            output,
            operation,
            donor_operand,
            scalar_type,
            bytes,
            alignment_bytes,
        ))
    }

    fn validate_binary_donor_selection(
        &self,
        donor: ValueId,
        other: ValueId,
        output: ValueId,
    ) -> Result<(), TensorStorageReuseError> {
        for value in [donor, other, output] {
            if value.graph_id != self.graph.id || self.graph.nodes.get(value.index).is_none() {
                return Err(TensorStorageReuseError::ValueNotInProgram(value));
            }
        }
        if donor == other {
            return Err(TensorStorageReuseError::DonorAndOtherAreSameValue(donor));
        }
        if !self.input_values.contains(&donor)
            || !matches!(self.graph.nodes[donor.index].op, Op::Input)
        {
            return Err(TensorStorageReuseError::InputIsNotGraphInput(donor));
        }
        if !self.input_values.contains(&other)
            || !matches!(self.graph.nodes[other.index].op, Op::Input)
        {
            return Err(TensorStorageReuseError::InputIsNotGraphInput(other));
        }
        if self.input_values.len() != 2
            || !self.input_values.contains(&donor)
            || !self.input_values.contains(&other)
            || self.output_values.as_slice() != [output]
        {
            return Err(TensorStorageReuseError::NotTerminalBinary {
                donor,
                other,
                output,
            });
        }

        let Some(output_operation_index) = self.operation_index_of(output) else {
            return Err(TensorStorageReuseError::OutputIsNotSelectedOutput(output));
        };
        let output_uses = self.operation_use_counts[output_operation_index];
        if output_uses != 1 {
            return Err(TensorStorageReuseError::OutputHasSelectedConsumers {
                value: output,
                actual: output_uses.saturating_sub(1),
            });
        }
        let donor_operation_index = self
            .operation_index_of(donor)
            .ok_or(TensorStorageReuseError::ValueNotInProgram(donor))?;
        let other_operation_index = self
            .operation_index_of(other)
            .ok_or(TensorStorageReuseError::ValueNotInProgram(other))?;
        for (value, index) in [
            (donor, donor_operation_index),
            (other, other_operation_index),
        ] {
            let actual = self.operation_use_counts[index];
            if actual != 1 {
                return Err(TensorStorageReuseError::InputUseCount { value, actual });
            }
        }

        if self.operations.len() != 3
            || !matches!(
                self.operations.get(donor_operation_index),
                Some(TensorOwnedSelectedOperation::Node { value }) if *value == donor
            )
            || !matches!(
                self.operations.get(other_operation_index),
                Some(TensorOwnedSelectedOperation::Node { value }) if *value == other
            )
            || !matches!(
                self.operations.get(output_operation_index),
                Some(TensorOwnedSelectedOperation::Node { value }) if *value == output
            )
        {
            return Err(TensorStorageReuseError::NotTerminalBinary {
                donor,
                other,
                output,
            });
        }
        Ok(())
    }

    fn binary_donor_layout(
        &self,
        donor: ValueId,
        other: ValueId,
        output: ValueId,
    ) -> Result<(PcuScalarType, u64, u64), TensorStorageReuseError> {
        let donor_node = &self.graph.nodes[donor.index];
        let other_node = &self.graph.nodes[other.index];
        let output_node = &self.graph.nodes[output.index];
        if donor_node.shape != other_node.shape || donor_node.shape != output_node.shape {
            return Err(TensorStorageReuseError::ShapeMismatch {
                input: donor,
                output,
            });
        }
        if donor_node.scalar_type != other_node.scalar_type
            || donor_node.scalar_type != output_node.scalar_type
        {
            return Err(TensorStorageReuseError::ScalarTypeMismatch {
                input: donor,
                output,
            });
        }
        let scalar_type = donor_node.scalar_type;
        let Some((_, alignment_bytes)) = dense_scalar_layout(scalar_type) else {
            return Err(TensorStorageReuseError::UnsupportedScalarType {
                value: donor,
                scalar_type,
            });
        };
        let donor_bytes = checked_node_output_bytes(donor, &donor_node.shape, scalar_type)
            .map_err(|_| TensorStorageReuseError::ShapeOverflow(donor))?;
        let output_bytes = checked_node_output_bytes(output, &output_node.shape, scalar_type)
            .map_err(|_| TensorStorageReuseError::ShapeOverflow(output))?;
        if donor_bytes != output_bytes {
            return Err(TensorStorageReuseError::LayoutMismatch {
                input: donor,
                output,
            });
        }
        Ok((
            scalar_type,
            u64::try_from(donor_bytes)
                .map_err(|_| TensorStorageReuseError::ShapeOverflow(donor))?,
            u64::try_from(alignment_bytes)
                .map_err(|_| TensorStorageReuseError::ShapeOverflow(donor))?,
        ))
    }

    /// Source-graph inclusive liveness facts aligned with [`Self::node_order`].
    #[must_use]
    pub fn value_liveness(&self) -> &[TensorValueLiveness] {
        &self.value_liveness
    }

    /// Peak live source-graph bytes, or `None` when extents overflow.
    #[must_use]
    pub const fn peak_live_bytes(&self) -> Option<usize> {
        self.peak_live_bytes
    }

    /// Source graph storage-disjointness constraints for this selected dependency closure.
    #[must_use]
    pub fn source_storage_constraints(&self) -> &[TensorStorageConstraint] {
        &self.source_storage_constraints
    }

    /// Nodes retained by arithmetic lowering before pointwise grouping.
    #[must_use]
    pub fn selected_nodes(&self) -> &[ValueId] {
        &self.selected_nodes
    }

    /// Reader counts aligned with [`Self::selected_nodes`], including output pins.
    #[must_use]
    pub fn selected_use_counts(&self) -> &[usize] {
        &self.selected_use_counts
    }

    /// Arithmetic rewrites selected under the policy used to build this program.
    #[must_use]
    pub fn rewrites(&self) -> &[TensorSgdRewriteCandidate] {
        &self.rewrites
    }

    /// Values omitted by the selected arithmetic or pointwise lowering.
    #[must_use]
    pub fn suppressed_values(&self) -> &[ValueId] {
        &self.suppressed_values
    }

    /// Lifetime-free operation sequence. Source nodes are addressed through [`Self::graph`].
    #[must_use]
    pub fn operations(&self) -> &[TensorOwnedSelectedOperation] {
        &self.operations
    }

    /// Selected operation index for a value, or `None` if it was removed by lowering.
    #[must_use]
    pub fn operation_index_of(&self, value: ValueId) -> Option<usize> {
        if value.graph_id != self.graph.id {
            return None;
        }
        self.operation_index_by_value
            .get(value.index)
            .copied()
            .filter(|&index| index != usize::MAX)
    }

    /// Reader counts aligned with [`Self::operations`], including requested-output pins.
    #[must_use]
    pub fn operation_use_counts(&self) -> &[usize] {
        &self.operation_use_counts
    }

    /// Selected operation liveness, derived from the actual rewritten/fused schedule.
    #[must_use]
    pub fn operation_liveness(&self) -> &[TensorValueLiveness] {
        &self.operation_liveness
    }

    /// Storage-disjointness constraints for the exact selected operation sequence.
    #[must_use]
    pub fn operation_storage_constraints(&self) -> &[TensorStorageConstraint] {
        &self.operation_storage_constraints
    }

    /// Plans scratch slots for selected computed values in a serial same-queue schedule.
    ///
    /// The operation liveness was captured when this owned program was constructed, so this
    /// does not rebuild the graph plan. Inputs, constants, uniforms, requested outputs, and
    /// values not materialized by the caller remain ineligible.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an eligible value from another graph and
    /// [`TensorError::ShapeOverflow`] when a selected output extent cannot be represented.
    pub fn scratch_storage_plan(
        &self,
        eligible_values: &[ValueId],
    ) -> Result<TensorScratchStoragePlan, TensorError> {
        if let Some(&value) = eligible_values
            .iter()
            .find(|value| value.graph_id != self.graph.id)
        {
            return Err(TensorError::UnknownValue(value));
        }
        let eligible: Vec<_> = self
            .operation_liveness
            .iter()
            .zip(&self.operation_read_only)
            .filter_map(|(life, read_only)| {
                (eligible_values.contains(&life.value)
                    && !read_only
                    && !self.output_values.contains(&life.value))
                .then_some(life.value)
            })
            .collect();
        storage::plan_scratch_storage(&self.operation_liveness, &eligible)
    }

    /// Runs the original graph's deterministic CPU reference evaluator.
    ///
    /// Numerical rewrites in this program affect backend lowering only; the reference result
    /// intentionally remains the source graph's result.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, extra, or wrongly shaped graph inputs.
    pub fn execute_reference(
        &self,
        inputs: &[(ValueId, TensorValue)],
    ) -> Result<Execution, TensorError> {
        self.graph.evaluate(inputs)
    }
}

impl<'a> TensorExecutionPlan<'a> {
    #[must_use]
    pub fn node_order(&self) -> &[ValueId] {
        &self.node_order
    }

    /// Borrowed descriptors for the nodes in this plan's selected dependency closure.
    #[must_use]
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = NodeDescriptor<'a>> + use<'a, '_> {
        self.node_order
            .iter()
            .map(|value| self.graph.node_descriptor(value.index))
    }

    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_values
    }

    #[must_use]
    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }

    /// Returns the selected-plan index for a graph value, if it is in the dependency closure.
    #[must_use]
    pub fn index_of(&self, value: ValueId) -> Option<usize> {
        if value.graph_id != self.graph.id {
            return None;
        }
        self.index_by_value
            .get(value.index)
            .copied()
            .filter(|&index| index != usize::MAX)
    }

    /// Operand-reader counts aligned with [`Self::node_order`]. Requested outputs each receive
    /// one extra pin count so a backend can keep them alive after their final graph consumer.
    #[must_use]
    pub fn use_counts(&self) -> &[usize] {
        &self.use_counts
    }

    /// Finds safe-to-select SGD rewrite candidates in this plan's dependency closure.
    ///
    /// Candidates are returned only when `policy` opts into altered f32 rounding,
    /// `target_supports_contracted_arithmetic` is true, the multiply has exactly one reader,
    /// and its rate operand is a finite, elementwise-uniform constant matching the gradient
    /// shape. The target flag is the selected backend's explicit declaration that fused
    /// multiply-add semantics are accepted for this lowering. This method only describes
    /// replacements; it never mutates the graph or evaluator. A backend that cannot verify that
    /// arithmetic contract must pass `false` and preserve both operations.
    #[must_use]
    pub fn sgd_rewrite_candidates(
        &self,
        policy: TensorArithmeticRewritePolicy,
        target_arithmetic: TensorArithmeticCapability,
    ) -> Vec<TensorSgdRewriteCandidate> {
        if policy != TensorArithmeticRewritePolicy::AllowContractedArithmetic
            || target_arithmetic != TensorArithmeticCapability::ContractedMultiplyAdd
        {
            return Vec::new();
        }
        let mut candidates = Vec::new();
        for output in self.nodes() {
            let OpDescriptor::Sub {
                left: weights,
                right,
            } = output.op
            else {
                continue;
            };
            let Some(multiply_index) = self.index_of(right) else {
                continue;
            };
            // A requested intermediate is pinned and therefore has an additional use count.
            if self.use_counts[multiply_index] != 1 {
                continue;
            }
            let Some(multiply) = self.graph.nodes.get(right.index) else {
                continue;
            };
            let Op::Mul(rate_value, gradient) = &multiply.op else {
                continue;
            };
            let (rate_value, gradient) = (*rate_value, *gradient);
            if self.graph.shape(weights).ok() != self.graph.shape(gradient).ok()
                || self.graph.shape(right).ok() != self.graph.shape(gradient).ok()
            {
                continue;
            }
            let Some(rate_node) = self.graph.nodes.get(rate_value.index) else {
                continue;
            };
            let (learning_rate, rate_shape, rate_is_uniform) = match &rate_node.op {
                Op::Constant(rate_tensor) => {
                    let Ok(rate_tensor) = rate_tensor.as_typed::<f32>() else {
                        continue;
                    };
                    let Some(value) = rate_tensor
                        .known_uniform_value
                        .or_else(|| rate_tensor.data.first().copied())
                    else {
                        continue;
                    };
                    (
                        value,
                        rate_tensor.shape.as_slice(),
                        rate_tensor.known_uniform_value.is_some()
                            || rate_tensor
                                .data
                                .iter()
                                .all(|rate| rate.to_bits() == value.to_bits()),
                    )
                }
                Op::Uniform(value) => {
                    let Ok(value) = value.as_typed::<f32>() else {
                        continue;
                    };
                    (value, rate_node.shape.as_slice(), true)
                }
                _ => continue,
            };
            if !learning_rate.is_finite()
                || rate_shape != rate_node.shape
                || rate_shape != self.graph.nodes[gradient.index].shape
                || !rate_is_uniform
            {
                continue;
            }
            candidates.push(TensorSgdRewriteCandidate {
                output: output.value,
                multiply: right,
                weights,
                gradient,
                learning_rate,
            });
        }
        candidates
    }

    /// Selects an inspectable operation schedule without modifying the source graph.
    ///
    /// Rewrites are enabled only when both the user policy and the target's declared arithmetic
    /// capability permit them. The schedule retains requested outputs and recomputes value-use
    /// counts after suppressing the uniquely consumed multiply node and any orphaned rate
    /// constant. Shared or explicitly requested constants remain in the schedule.
    #[must_use]
    pub fn select_lowering(
        &self,
        policy: TensorArithmeticRewritePolicy,
        target_arithmetic: TensorArithmeticCapability,
    ) -> TensorSelectedLoweringPlan<'a> {
        self.select_lowering_with_grouping(
            policy,
            target_arithmetic,
            TensorPointwiseGroupingPolicy::Disabled,
        )
    }

    /// Selects a lowering and, independently, an optional pointwise grouping schedule.
    ///
    /// Grouping metadata does not change graph construction or CPU reference execution. The
    /// returned typed operation schedule suppresses the single-use Add intermediate and carries
    /// the original Add operands to the fused Add-ReLU operation. Arithmetic rewrite policy
    /// remains independent, so the Add rounding point is preserved in the grouped operation.
    #[allow(clippy::too_many_lines)] // Keep policy selection and schedule assembly together.
    #[must_use]
    pub fn select_lowering_with_grouping(
        &self,
        policy: TensorArithmeticRewritePolicy,
        target_arithmetic: TensorArithmeticCapability,
        grouping_policy: TensorPointwiseGroupingPolicy,
    ) -> TensorSelectedLoweringPlan<'a> {
        let rewritten = self.sgd_rewrite_candidates(policy, target_arithmetic);
        let mut suppressed_values: Vec<_> = rewritten
            .iter()
            .map(|candidate| candidate.multiply)
            .collect();
        let mut nodes = Vec::with_capacity(
            self.node_order
                .len()
                .saturating_sub(suppressed_values.len()),
        );
        for mut node in self.nodes() {
            if suppressed_values.contains(&node.value) {
                continue;
            }
            if let Some(candidate) = rewritten
                .iter()
                .find(|candidate| candidate.output == node.value)
            {
                node.op = OpDescriptor::SgdUpdate {
                    weights: candidate.weights,
                    gradient: candidate.gradient,
                    learning_rate: candidate.learning_rate,
                };
            }
            nodes.push(node);
        }
        let count_uses = |nodes: &[NodeDescriptor<'_>]| {
            let mut index_by_value = vec![usize::MAX; self.graph.nodes.len()];
            for (index, node) in nodes.iter().enumerate() {
                index_by_value[node.value.index] = index;
            }
            let mut use_counts = vec![0usize; nodes.len()];
            for node in nodes {
                for_each_descriptor_operand(node.op, |operand| {
                    if let Some(index) = index_by_value.get(operand.index).copied()
                        && index != usize::MAX
                    {
                        use_counts[index] += 1;
                    }
                });
            }
            for output in &self.output_values {
                if let Some(index) = index_by_value.get(output.index).copied()
                    && index != usize::MAX
                {
                    use_counts[index] += 1;
                }
            }
            (index_by_value, use_counts)
        };
        let (mut index_by_value, mut use_counts) = count_uses(&nodes);
        // A rewrite can orphan its uniform rate constant. Keep shared or requested constants,
        // but do not make a backend upload a value that the selected schedule never reads.
        if !rewritten.is_empty() {
            let mut removed = Vec::new();
            nodes.retain(|node| {
                let unused_constant = matches!(
                    node.op,
                    OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
                ) && use_counts[index_by_value[node.value.index]] == 0;
                if unused_constant {
                    removed.push(node.value);
                }
                !unused_constant
            });
            if !removed.is_empty() {
                suppressed_values.extend(removed);
                (index_by_value, use_counts) = count_uses(&nodes);
            }
        }
        let (pointwise_groups, bounded_pointwise_groups, bounded_mul_groups) =
            select_pointwise_groups(
                &nodes,
                &index_by_value,
                &use_counts,
                &self.output_values,
                grouping_policy,
            );
        let (operations, operation_index_by_value, operation_use_counts) =
            build_selected_operations(
                &nodes,
                &pointwise_groups,
                &bounded_pointwise_groups,
                &bounded_mul_groups,
                self.graph.nodes.len(),
                &self.output_values,
            );
        TensorSelectedLoweringPlan {
            graph_id: self.graph.id,
            nodes,
            input_values: self.input_values.clone(),
            output_values: self.output_values.clone(),
            index_by_value,
            use_counts,
            rewritten,
            suppressed_values,
            pointwise_groups,
            bounded_pointwise_groups,
            bounded_mul_groups,
            operations,
            operation_index_by_value,
            operation_use_counts,
        }
    }

    #[must_use]
    pub fn value_liveness(&self) -> &[TensorValueLiveness] {
        &self.value_liveness
    }

    /// Plans reusable scratch slots for selected transient values in a serial same-queue schedule.
    ///
    /// `eligible_values` comes from the selected lowering so fused or storage-free values can be
    /// omitted. Inputs, constants, uniforms, and requested outputs are always excluded. A slot is
    /// reused only when the previous value's inclusive lifetime ends before the new value is
    /// produced; this does not permit same-operation in-place aliasing.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for an ID outside this execution plan and `ShapeOverflow` when a
    /// selected value extent or total scratch capacity cannot be represented.
    pub fn scratch_storage_plan(
        &self,
        eligible_values: &[ValueId],
    ) -> Result<TensorScratchStoragePlan, TensorError> {
        for &value in eligible_values {
            if self.index_of(value).is_none() {
                return Err(TensorError::UnknownValue(value));
            }
        }

        let eligible: Vec<_> = self
            .value_liveness
            .iter()
            .filter(|life| {
                eligible_values.contains(&life.value)
                    && self.is_writable_value(life.value)
                    && !self.output_values.contains(&life.value)
            })
            .map(|life| life.value)
            .collect();
        storage::plan_scratch_storage(&self.value_liveness, &eligible)
    }

    /// Returns all pairwise storage disjointness requirements implied by live writable values.
    ///
    /// Lifetimes are inclusive, so values whose intervals touch at a node are both considered
    /// live there. Read-only input/constant pairs are omitted; computed values remain disjoint
    /// under these default constraints. A consumed-ReLU graph proof is separate evidence and
    /// does not relax this validation path.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` if a selected value's byte extent cannot be represented.
    pub fn storage_constraints(&self) -> Result<Vec<TensorStorageConstraint>, TensorError> {
        let mut constraints = Vec::new();
        for (i, left) in self.value_liveness.iter().enumerate() {
            for right in &self.value_liveness[i + 1..] {
                let left_read_only = !self.is_writable_value(left.value);
                let right_read_only = !self.is_writable_value(right.value);
                let disjoint_lifetimes = left.last_live_node < right.first_live_node
                    || right.last_live_node < left.first_live_node;
                if (disjoint_lifetimes && !left_read_only && !right_read_only)
                    || (left_read_only && right_read_only)
                {
                    continue;
                }
                let left_bytes = u64::try_from(checked_node_output_bytes(
                    left.value,
                    &self.graph.nodes[left.value.index].shape,
                    left.scalar_type,
                )?)
                .map_err(|_| TensorError::ShapeOverflow)?;
                let right_bytes = u64::try_from(checked_node_output_bytes(
                    right.value,
                    &self.graph.nodes[right.value.index].shape,
                    right.scalar_type,
                )?)
                .map_err(|_| TensorError::ShapeOverflow)?;
                constraints.push(TensorStorageConstraint {
                    left: left.value,
                    right: right.value,
                    left_bytes,
                    right_bytes,
                });
            }
        }
        Ok(constraints)
    }

    fn is_writable_value(&self, value: ValueId) -> bool {
        !matches!(
            self.graph.nodes[value.index].op,
            Op::Input | Op::Constant(_) | Op::Uniform(_)
        )
    }

    /// Returns dense-layout resource extents and required access for selected values.
    ///
    /// Inputs and constants are read-only; computed values may be produced and consumed by
    /// selected operations, so their resources must support both reads and writes. The byte
    /// extents are logical dense extents. A backend using compact uniforms, views, or another
    /// physical layout must physicalize these requirements before validating actual resources.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` if a selected value's byte extent cannot be represented.
    pub fn value_storage_requirements(
        &self,
    ) -> Result<Vec<TensorValueStorageRequirement>, TensorError> {
        self.value_liveness
            .iter()
            .map(|life| {
                let shape = &self.graph.nodes[life.value.index].shape;
                let output_bytes = checked_node_output_bytes(life.value, shape, life.scalar_type)?;
                let (_, alignment_bytes) = dense_scalar_layout(life.scalar_type).ok_or(
                    TensorError::UnsupportedScalarType {
                        value: life.value,
                        scalar_type: life.scalar_type,
                    },
                )?;
                Ok(TensorValueStorageRequirement {
                    value: life.value,
                    scalar_type: life.scalar_type,
                    output_bytes: u64::try_from(output_bytes)
                        .map_err(|_| TensorError::ShapeOverflow)?,
                    alignment_bytes: u64::try_from(alignment_bytes)
                        .map_err(|_| TensorError::ShapeOverflow)?,
                    access: if self.is_writable_value(life.value) {
                        crate::PcuMemoryAccess::ReadWrite
                    } else {
                        crate::PcuMemoryAccess::ReadOnly
                    },
                })
            })
            .collect()
    }

    #[must_use]
    pub const fn peak_live_bytes(&self) -> Option<usize> {
        self.peak_live_bytes
    }

    /// Executes this graph with the deterministic CPU reference evaluator.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, extra, or wrongly shaped inputs.
    pub fn execute_reference(
        &self,
        inputs: &[(ValueId, TensorValue)],
    ) -> Result<Execution, TensorError> {
        self.graph.evaluate(inputs)
    }
}

fn for_each_descriptor_operand(op: OpDescriptor<'_>, mut visit: impl FnMut(ValueId)) {
    match op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {}
        OpDescriptor::Add { left, right }
        | OpDescriptor::Sub { left, right }
        | OpDescriptor::Mul { left, right }
        | OpDescriptor::Div { left, right }
        | OpDescriptor::MatMul { left, right, .. } => {
            visit(left);
            visit(right);
        }
        OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } => {
            visit(weights);
            visit(gradient);
        }
        OpDescriptor::Relu { input } => visit(input),
        OpDescriptor::ReluBackward { input, upstream } => {
            visit(input);
            visit(upstream);
        }
        OpDescriptor::MeanSquaredError { prediction, target } => {
            visit(prediction);
            visit(target);
        }
    }
}

fn identify_pointwise_groups(
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
) -> Vec<TensorPointwiseFusionGroup> {
    let mut groups = Vec::new();
    for relu in nodes {
        let OpDescriptor::Relu { input: add_output } = relu.op else {
            continue;
        };
        let Some(add_index) = index_by_value
            .get(add_output.index)
            .copied()
            .filter(|index| *index != usize::MAX)
        else {
            continue;
        };
        // A second graph consumer or output pin makes the intermediate observable.
        if use_counts[add_index] != 1 {
            continue;
        }
        let add = nodes[add_index];
        if !supports_unchecked_fusion(add) {
            continue;
        }
        let OpDescriptor::Add { left, right } = add.op else {
            continue;
        };
        let same_shape = |value: ValueId| {
            index_by_value
                .get(value.index)
                .copied()
                .filter(|index| *index != usize::MAX)
                .is_some_and(|index| nodes[index].shape == add.shape)
        };
        if relu.shape != add.shape || !same_shape(left) || !same_shape(right) {
            continue;
        }
        groups.push(TensorPointwiseFusionGroup {
            add_output,
            relu_output: relu.value,
            left,
            right,
            shape: add.shape.to_vec(),
        });
    }
    groups
}

fn select_pointwise_groups(
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
    output_values: &[ValueId],
    policy: TensorPointwiseGroupingPolicy,
) -> (
    Vec<TensorPointwiseFusionGroup>,
    Vec<TensorBoundedPointwiseFusionGroup>,
    Vec<TensorBoundedMulFusionGroup>,
) {
    match policy {
        TensorPointwiseGroupingPolicy::Disabled => (Vec::new(), Vec::new(), Vec::new()),
        TensorPointwiseGroupingPolicy::SingleUseAddRelu => (
            identify_pointwise_groups(nodes, index_by_value, use_counts),
            Vec::new(),
            Vec::new(),
        ),
        TensorPointwiseGroupingPolicy::BoundedAddSubRelu => (
            Vec::new(),
            identify_bounded_pointwise_groups(
                nodes,
                index_by_value,
                use_counts,
                output_values,
                TensorPointwiseEpilogue::Relu,
            ),
            Vec::new(),
        ),
        TensorPointwiseGroupingPolicy::BoundedAddSubIdentity => (
            Vec::new(),
            identify_bounded_pointwise_groups(
                nodes,
                index_by_value,
                use_counts,
                output_values,
                TensorPointwiseEpilogue::Identity,
            ),
            Vec::new(),
        ),
        TensorPointwiseGroupingPolicy::BoundedMulIdentity => (
            Vec::new(),
            Vec::new(),
            identify_bounded_mul_groups(nodes, index_by_value, use_counts, output_values),
        ),
    }
}

const MAX_BOUNDED_POINTWISE_STEPS: usize = 8;
const MAX_BOUNDED_POINTWISE_LEAVES: usize = 4;

fn identify_bounded_pointwise_groups(
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
    output_values: &[ValueId],
    epilogue: TensorPointwiseEpilogue,
) -> Vec<TensorBoundedPointwiseFusionGroup> {
    match epilogue {
        TensorPointwiseEpilogue::Relu => nodes
            .iter()
            .filter_map(|relu| {
                identify_bounded_pointwise_relu_group(relu, nodes, index_by_value, use_counts)
            })
            .collect(),
        TensorPointwiseEpilogue::Identity => nodes
            .iter()
            .filter(|node| {
                matches!(node.op, OpDescriptor::Add { .. } | OpDescriptor::Sub { .. })
                    && is_identity_group_boundary(node, nodes, output_values)
            })
            .filter_map(|terminal| {
                identify_bounded_pointwise_identity_group(
                    terminal,
                    nodes,
                    index_by_value,
                    use_counts,
                )
            })
            .collect(),
    }
}

fn identify_bounded_mul_groups(
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
    output_values: &[ValueId],
) -> Vec<TensorBoundedMulFusionGroup> {
    nodes
        .iter()
        .filter(|node| {
            supports_unchecked_fusion(**node) && matches!(node.op, OpDescriptor::Mul { .. })
        })
        .filter(|terminal| {
            output_values.contains(&terminal.value)
                || nodes.iter().any(|consumer| {
                    consumer.value != terminal.value
                        && !matches!(consumer.op, OpDescriptor::Mul { .. })
                        && descriptor_reads(consumer.op, terminal.value)
                })
                || use_counts[index_by_value[terminal.value.index]] > 1
        })
        .filter_map(|terminal| {
            let mut included = Vec::new();
            if !collect_bounded_mul(
                terminal.value,
                terminal.shape,
                nodes,
                index_by_value,
                use_counts,
                &mut included,
                false,
            ) {
                return None;
            }
            included.sort_unstable();
            included.dedup();
            if !(2..=MAX_BOUNDED_POINTWISE_STEPS).contains(&included.len()) {
                return None;
            }
            make_bounded_mul_group(terminal, &included, nodes, index_by_value)
        })
        .collect()
}

fn descriptor_reads(op: OpDescriptor<'_>, value: ValueId) -> bool {
    let mut reads = false;
    for_each_descriptor_operand(op, |operand| reads |= operand == value);
    reads
}

fn collect_bounded_mul(
    value: ValueId,
    shape: &[usize],
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
    included: &mut Vec<usize>,
    require_single_use: bool,
) -> bool {
    let Some(index) = index_by_value
        .get(value.index)
        .copied()
        .filter(|index| *index != usize::MAX)
    else {
        return false;
    };
    let node = nodes[index];
    if node.shape != shape
        || !supports_unchecked_fusion(node)
        || (require_single_use && use_counts[index] != 1)
        || !matches!(node.op, OpDescriptor::Mul { .. })
    {
        return false;
    }
    if included.contains(&index) || included.len() >= MAX_BOUNDED_POINTWISE_STEPS {
        return false;
    }
    included.push(index);
    let OpDescriptor::Mul { left, right } = node.op else {
        return false;
    };
    let is_same_shape_mul = |child: ValueId| {
        index_by_value
            .get(child.index)
            .copied()
            .filter(|child_index| *child_index != usize::MAX)
            .is_some_and(|child_index| {
                nodes[child_index].shape == shape
                    && matches!(nodes[child_index].op, OpDescriptor::Mul { .. })
            })
    };
    let left_mul = is_same_shape_mul(left);
    let right_mul = is_same_shape_mul(right);
    let left_eligible = left_mul && use_counts[index_by_value[left.index]] == 1;
    let right_eligible = right_mul && use_counts[index_by_value[right.index]] == 1;
    if (left_mul && !left_eligible)
        || (right_mul && !right_eligible)
        || (left_eligible && right_eligible)
    {
        return false;
    }
    (!left_eligible
        || collect_bounded_mul(
            left,
            shape,
            nodes,
            index_by_value,
            use_counts,
            included,
            true,
        ))
        && (!right_eligible
            || collect_bounded_mul(
                right,
                shape,
                nodes,
                index_by_value,
                use_counts,
                included,
                true,
            ))
}

fn make_bounded_mul_group(
    terminal: &NodeDescriptor<'_>,
    included: &[usize],
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
) -> Option<TensorBoundedMulFusionGroup> {
    let mut leaves = Vec::new();
    let mut steps = Vec::with_capacity(included.len());
    let mut step_by_node = vec![usize::MAX; nodes.len()];
    for &node_index in included {
        let node = nodes[node_index];
        let OpDescriptor::Mul { left, right } = node.op else {
            return None;
        };
        let mut resolve = |value: ValueId| {
            let producer = index_by_value
                .get(value.index)
                .copied()
                .unwrap_or(usize::MAX);
            if producer != usize::MAX && step_by_node[producer] != usize::MAX {
                TensorPointwiseOperand::Step(step_by_node[producer])
            } else {
                let leaf_index = leaves
                    .iter()
                    .position(|leaf| *leaf == value)
                    .unwrap_or_else(|| {
                        leaves.push(value);
                        leaves.len() - 1
                    });
                TensorPointwiseOperand::Leaf(leaf_index)
            }
        };
        let left = resolve(left);
        let right = resolve(right);
        if leaves.len() > MAX_BOUNDED_POINTWISE_LEAVES {
            return None;
        }
        step_by_node[node_index] = steps.len();
        steps.push(TensorPointwiseMulStep {
            output: node.value,
            left,
            right,
        });
    }
    Some(TensorBoundedMulFusionGroup {
        output: terminal.value,
        leaves,
        steps,
        shape: terminal.shape.to_vec(),
    })
}

fn is_identity_group_boundary(
    terminal: &NodeDescriptor<'_>,
    nodes: &[NodeDescriptor<'_>],
    output_values: &[ValueId],
) -> bool {
    if output_values.contains(&terminal.value) {
        return true;
    }
    let mut consumer_count = 0;
    for consumer in nodes {
        let mut reads_terminal = false;
        for_each_descriptor_operand(consumer.op, |operand| {
            reads_terminal |= operand == terminal.value;
        });
        if !reads_terminal {
            continue;
        }
        consumer_count += 1;
        // A lone terminal ReLU uses the established ReLU epilogue route. Any consumer outside
        // the Add/Sub chain makes this arithmetic value a real boundary to preserve.
        if !matches!(
            consumer.op,
            OpDescriptor::Add { .. } | OpDescriptor::Sub { .. } | OpDescriptor::Relu { .. }
        ) {
            return true;
        }
    }
    // Multiple readers form a cut: preserve the shared result and group only the single-use
    // chain that computes it.
    consumer_count > 1
}

fn identify_bounded_pointwise_relu_group(
    relu: &NodeDescriptor<'_>,
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
) -> Option<TensorBoundedPointwiseFusionGroup> {
    let OpDescriptor::Relu { input } = relu.op else {
        return None;
    };
    if relu.shape
        != nodes
            .get(
                index_by_value
                    .get(input.index)
                    .copied()
                    .unwrap_or(usize::MAX),
            )
            .map_or(&[][..], |node| node.shape)
    {
        return None;
    }
    let root_index = index_by_value
        .get(input.index)
        .copied()
        .filter(|index| *index != usize::MAX)?;
    let root = nodes[root_index];
    if !matches!(root.op, OpDescriptor::Add { .. } | OpDescriptor::Sub { .. })
        || !supports_unchecked_fusion(root)
        || use_counts[root_index] != 1
    {
        return None;
    }

    let shape = relu.shape;
    let mut included = Vec::new();
    if !collect_bounded_arithmetic(
        input,
        shape,
        nodes,
        index_by_value,
        use_counts,
        &mut included,
        false,
    ) {
        return None;
    }
    included.sort_unstable();
    included.dedup();
    // The existing two-node Add -> ReLU selection remains its own explicit policy. The
    // bounded policy is for actual arithmetic chains, not a second spelling of that group.
    if !(2..=MAX_BOUNDED_POINTWISE_STEPS).contains(&included.len()) {
        return None;
    }

    let mut leaves = Vec::new();
    let mut steps = Vec::with_capacity(included.len());
    let mut step_by_node = vec![usize::MAX; nodes.len()];
    for &node_index in &included {
        let node = nodes[node_index];
        let (op, left, right) = match node.op {
            OpDescriptor::Add { left, right } => (TensorPointwiseArithmeticOp::Add, left, right),
            OpDescriptor::Sub { left, right } => (TensorPointwiseArithmeticOp::Sub, left, right),
            _ => continue,
        };
        let resolve = |value: ValueId, leaves: &mut Vec<ValueId>, step_by_node: &[usize]| {
            let producer = index_by_value
                .get(value.index)
                .copied()
                .unwrap_or(usize::MAX);
            if producer != usize::MAX && step_by_node[producer] != usize::MAX {
                TensorPointwiseOperand::Step(step_by_node[producer])
            } else {
                let leaf_index = leaves
                    .iter()
                    .position(|leaf| *leaf == value)
                    .unwrap_or_else(|| {
                        leaves.push(value);
                        leaves.len() - 1
                    });
                TensorPointwiseOperand::Leaf(leaf_index)
            }
        };
        let left = resolve(left, &mut leaves, &step_by_node);
        let right = resolve(right, &mut leaves, &step_by_node);
        if leaves.len() > MAX_BOUNDED_POINTWISE_LEAVES {
            steps.clear();
            break;
        }
        step_by_node[node_index] = steps.len();
        steps.push(TensorPointwiseStep {
            output: node.value,
            op,
            left,
            right,
        });
    }
    if steps.len() != included.len() || leaves.len() > MAX_BOUNDED_POINTWISE_LEAVES {
        return None;
    }
    Some(TensorBoundedPointwiseFusionGroup {
        output: relu.value,
        epilogue: TensorPointwiseEpilogue::Relu,
        leaves,
        steps,
        shape: shape.to_vec(),
    })
}

fn identify_bounded_pointwise_identity_group(
    terminal: &NodeDescriptor<'_>,
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
) -> Option<TensorBoundedPointwiseFusionGroup> {
    if !supports_unchecked_fusion(*terminal) {
        return None;
    }
    let mut included = Vec::new();
    if !collect_bounded_arithmetic(
        terminal.value,
        terminal.shape,
        nodes,
        index_by_value,
        use_counts,
        &mut included,
        false,
    ) {
        return None;
    }
    included.sort_unstable();
    included.dedup();
    if !(2..=MAX_BOUNDED_POINTWISE_STEPS).contains(&included.len()) {
        return None;
    }
    make_bounded_pointwise_group(
        terminal.value,
        terminal.shape,
        TensorPointwiseEpilogue::Identity,
        &included,
        nodes,
        index_by_value,
    )
}

fn make_bounded_pointwise_group(
    output: ValueId,
    shape: &[usize],
    epilogue: TensorPointwiseEpilogue,
    included: &[usize],
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
) -> Option<TensorBoundedPointwiseFusionGroup> {
    let mut leaves = Vec::new();
    let mut steps = Vec::with_capacity(included.len());
    let mut step_by_node = vec![usize::MAX; nodes.len()];
    for &node_index in included {
        let node = nodes[node_index];
        let (op, left, right) = match node.op {
            OpDescriptor::Add { left, right } => (TensorPointwiseArithmeticOp::Add, left, right),
            OpDescriptor::Sub { left, right } => (TensorPointwiseArithmeticOp::Sub, left, right),
            _ => return None,
        };
        let resolve = |value: ValueId, leaves: &mut Vec<ValueId>, step_by_node: &[usize]| {
            let producer = index_by_value
                .get(value.index)
                .copied()
                .unwrap_or(usize::MAX);
            if producer != usize::MAX && step_by_node[producer] != usize::MAX {
                TensorPointwiseOperand::Step(step_by_node[producer])
            } else {
                let leaf_index = leaves
                    .iter()
                    .position(|leaf| *leaf == value)
                    .unwrap_or_else(|| {
                        leaves.push(value);
                        leaves.len() - 1
                    });
                TensorPointwiseOperand::Leaf(leaf_index)
            }
        };
        let left = resolve(left, &mut leaves, &step_by_node);
        let right = resolve(right, &mut leaves, &step_by_node);
        if leaves.len() > MAX_BOUNDED_POINTWISE_LEAVES {
            return None;
        }
        step_by_node[node_index] = steps.len();
        steps.push(TensorPointwiseStep {
            output: node.value,
            op,
            left,
            right,
        });
    }
    Some(TensorBoundedPointwiseFusionGroup {
        output,
        epilogue,
        leaves,
        steps,
        shape: shape.to_vec(),
    })
}

fn collect_bounded_arithmetic(
    value: ValueId,
    shape: &[usize],
    nodes: &[NodeDescriptor<'_>],
    index_by_value: &[usize],
    use_counts: &[usize],
    included: &mut Vec<usize>,
    require_single_use: bool,
) -> bool {
    let Some(index) = index_by_value
        .get(value.index)
        .copied()
        .filter(|index| *index != usize::MAX)
    else {
        return false;
    };
    let node = nodes[index];
    if node.shape != shape
        || !supports_unchecked_fusion(node)
        || (require_single_use && use_counts[index] != 1)
        || !matches!(node.op, OpDescriptor::Add { .. } | OpDescriptor::Sub { .. })
    {
        return false;
    }
    if included.contains(&index) {
        return false;
    }
    if included.len() >= MAX_BOUNDED_POINTWISE_STEPS {
        return false;
    }
    included.push(index);
    if let OpDescriptor::Add { left, right } | OpDescriptor::Sub { left, right } = node.op {
        let child_is_arithmetic = |value: ValueId| {
            index_by_value
                .get(value.index)
                .copied()
                .filter(|child| *child != usize::MAX)
                .is_some_and(|child| {
                    nodes[child].shape == shape
                        && matches!(
                            nodes[child].op,
                            OpDescriptor::Add { .. } | OpDescriptor::Sub { .. }
                        )
                })
        };
        let left_arithmetic = child_is_arithmetic(left);
        let right_arithmetic = child_is_arithmetic(right);
        let left_eligible = left_arithmetic && use_counts[index_by_value[left.index]] == 1;
        let right_eligible = right_arithmetic && use_counts[index_by_value[right.index]] == 1;
        // An arithmetic intermediate with another reader or output pin is a cut in this
        // candidate. Reject the group instead of smuggling it through as an external leaf.
        if (left_arithmetic && !left_eligible) || (right_arithmetic && !right_eligible) {
            return false;
        }
        // This first slice is a chain. Branched expressions stay separate until they have a
        // distinct numeric contract and backend implementation.
        if left_eligible && right_eligible {
            return false;
        }
        if left_eligible
            && !collect_bounded_arithmetic(
                left,
                shape,
                nodes,
                index_by_value,
                use_counts,
                included,
                true,
            )
        {
            return false;
        }
        if right_eligible
            && !collect_bounded_arithmetic(
                right,
                shape,
                nodes,
                index_by_value,
                use_counts,
                included,
                true,
            )
        {
            return false;
        }
    }
    true
}

fn build_selected_operations<'a>(
    nodes: &[NodeDescriptor<'a>],
    groups: &[TensorPointwiseFusionGroup],
    bounded_groups: &[TensorBoundedPointwiseFusionGroup],
    bounded_mul_groups: &[TensorBoundedMulFusionGroup],
    graph_node_count: usize,
    output_values: &[ValueId],
) -> (Vec<TensorSelectedOperation<'a>>, Vec<usize>, Vec<usize>) {
    let mut operations = Vec::with_capacity(nodes.len());
    let mut index_by_value = vec![usize::MAX; graph_node_count];
    for node in nodes {
        if groups.iter().any(|group| group.add_output == node.value)
            || bounded_mul_groups.iter().any(|group| {
                group.steps.iter().any(|step| step.output == node.value)
                    && group.output != node.value
            })
            || bounded_groups.iter().any(|group| {
                group.steps.iter().any(|step| step.output == node.value)
                    && !(group.epilogue == TensorPointwiseEpilogue::Identity
                        && group.output == node.value)
            })
        {
            continue;
        }
        let operation = bounded_groups
            .iter()
            .find(|group| group.output == node.value)
            .cloned()
            .map_or_else(
                || {
                    groups
                        .iter()
                        .find(|group| group.relu_output == node.value)
                        .map_or(TensorSelectedOperation::Node(*node), |group| {
                            TensorSelectedOperation::FusedAddRelu {
                                add_output: group.add_output,
                                relu_output: group.relu_output,
                                left: group.left,
                                right: group.right,
                                shape: node.shape,
                            }
                        })
                },
                |group| TensorSelectedOperation::FusedAddSub { group },
            );
        let operation = bounded_mul_groups
            .iter()
            .find(|group| group.output == node.value)
            .cloned()
            .map_or(operation, |group| TensorSelectedOperation::FusedMul {
                group,
            });
        index_by_value[node.value.index] = operations.len();
        operations.push(operation);
    }
    let mut use_counts = vec![0usize; operations.len()];
    for operation in &operations {
        for_each_selected_operand(operation, |operand| {
            if let Some(index) = index_by_value.get(operand.index).copied()
                && index != usize::MAX
            {
                use_counts[index] += 1;
            }
        });
    }
    for output in output_values {
        if let Some(index) = index_by_value.get(output.index).copied()
            && index != usize::MAX
        {
            use_counts[index] += 1;
        }
    }
    (operations, index_by_value, use_counts)
}

fn for_each_selected_operand(
    operation: &TensorSelectedOperation<'_>,
    mut visit: impl FnMut(ValueId),
) {
    match operation {
        TensorSelectedOperation::Node(node) => for_each_descriptor_operand(node.op, visit),
        TensorSelectedOperation::FusedAddRelu { left, right, .. } => {
            visit(*left);
            visit(*right);
        }
        TensorSelectedOperation::FusedAddSub { group } => {
            for step in &group.steps {
                for operand in [step.left, step.right] {
                    if let TensorPointwiseOperand::Leaf(index) = operand
                        && let Some(leaf) = group.leaves.get(index)
                    {
                        visit(*leaf);
                    }
                }
            }
        }
        TensorSelectedOperation::FusedMul { group } => {
            for step in &group.steps {
                for operand in [step.left, step.right] {
                    if let TensorPointwiseOperand::Leaf(index) = operand
                        && let Some(leaf) = group.leaves.get(index)
                    {
                        visit(*leaf);
                    }
                }
            }
        }
    }
}

const fn selected_operation_output(operation: &TensorSelectedOperation<'_>) -> ValueId {
    match operation {
        TensorSelectedOperation::Node(node) => node.value,
        TensorSelectedOperation::FusedAddRelu { relu_output, .. } => *relu_output,
        TensorSelectedOperation::FusedAddSub { group } => group.output,
        TensorSelectedOperation::FusedMul { group } => group.output,
    }
}

fn selected_operation_shape<'op>(operation: &'op TensorSelectedOperation<'_>) -> &'op [usize] {
    match operation {
        TensorSelectedOperation::Node(node) => node.shape,
        TensorSelectedOperation::FusedAddRelu { shape, .. } => shape,
        TensorSelectedOperation::FusedAddSub { group } => &group.shape,
        TensorSelectedOperation::FusedMul { group } => &group.shape,
    }
}

fn selected_operation_bytes(
    operation: &TensorSelectedOperation<'_>,
    nodes: &[NodeDescriptor<'_>],
) -> Result<u64, TensorError> {
    let output = selected_operation_output(operation);
    let scalar_type = nodes
        .iter()
        .find(|node| node.value == output)
        .map(|node| node.scalar_type)
        .ok_or(TensorError::UnknownValue(output))?;
    let bytes =
        checked_node_output_bytes(output, selected_operation_shape(operation), scalar_type)?;
    u64::try_from(bytes).map_err(|_| TensorError::ShapeOverflow)
}

fn checked_node_output_bytes(
    value: ValueId,
    shape: &[usize],
    scalar_type: PcuScalarType,
) -> Result<usize, TensorError> {
    if dense_scalar_layout(scalar_type).is_none() {
        return Err(TensorError::UnsupportedScalarType { value, scalar_type });
    }
    node_output_bytes(shape, scalar_type).ok_or(TensorError::ShapeOverflow)
}

const fn selected_operation_is_read_only(operation: &TensorSelectedOperation<'_>) -> bool {
    match operation {
        TensorSelectedOperation::Node(node) => is_read_only_descriptor(node.op),
        TensorSelectedOperation::FusedAddRelu { .. }
        | TensorSelectedOperation::FusedAddSub { .. }
        | TensorSelectedOperation::FusedMul { .. } => false,
    }
}

const fn is_read_only_descriptor(op: OpDescriptor<'_>) -> bool {
    matches!(
        op,
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
    )
}

#[derive(Clone, Debug)]
enum Op {
    Input,
    Constant(TensorValue),
    Uniform(TensorScalarValue),
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Div(ValueId, ValueId),
    SgdUpdate(ValueId, ValueId, f32),
    MatMul(ValueId, ValueId, bool, bool),
    Relu(ValueId),
    ReluBackward(ValueId, ValueId),
    MeanSquaredError(ValueId, ValueId),
}

fn binary_donor_operation(
    op: &Op,
    donor: ValueId,
    other: ValueId,
) -> Option<(TensorBinaryOperation, TensorBinaryOperand)> {
    match *op {
        Op::Add(left, right) if left == donor && right == other => {
            Some((TensorBinaryOperation::Add, TensorBinaryOperand::Left))
        }
        Op::Add(left, right) if left == other && right == donor => {
            Some((TensorBinaryOperation::Add, TensorBinaryOperand::Right))
        }
        Op::Sub(left, right) if left == donor && right == other => {
            Some((TensorBinaryOperation::Sub, TensorBinaryOperand::Left))
        }
        Op::Sub(left, right) if left == other && right == donor => {
            Some((TensorBinaryOperation::Sub, TensorBinaryOperand::Right))
        }
        Op::Mul(left, right) if left == donor && right == other => {
            Some((TensorBinaryOperation::Mul, TensorBinaryOperand::Left))
        }
        Op::Mul(left, right) if left == other && right == donor => {
            Some((TensorBinaryOperation::Mul, TensorBinaryOperand::Right))
        }
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn operands(op: &Op) -> impl Iterator<Item = ValueId> + '_ {
    let (first, second) = match op {
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::Div(a, b)
        | Op::SgdUpdate(a, b, _)
        | Op::MeanSquaredError(a, b)
        | Op::MatMul(a, b, _, _) => (Some(*a), Some(*b)),
        Op::Relu(x) => (Some(*x), None),
        Op::ReluBackward(input, upstream) => (Some(*input), Some(*upstream)),
        Op::Input | Op::Constant(_) | Op::Uniform(_) => (None, None),
    };
    first.into_iter().chain(second)
}

const fn checked_binary(
    op: &Op,
    scalar_type: PcuScalarType,
    float_underflow_policy: Option<PcuFloatUnderflowPolicy>,
) -> bool {
    matches!(
        op,
        Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Div(..) | Op::Relu(..) | Op::ReluBackward(..)
    ) && match scalar_type {
        PcuScalarType::U8
        | PcuScalarType::U16
        | PcuScalarType::U32
        | PcuScalarType::U64
        | PcuScalarType::I8
        | PcuScalarType::I16
        | PcuScalarType::I32
        | PcuScalarType::I64
        | PcuScalarType::U128
        | PcuScalarType::I128
        | PcuScalarType::U256
        | PcuScalarType::I256
        | PcuScalarType::U512
        | PcuScalarType::I512 => true,
        PcuScalarType::F16
        | PcuScalarType::BF16
        | PcuScalarType::F32
        | PcuScalarType::F64
        | PcuScalarType::F8E4M3FN
        | PcuScalarType::F8E5M2 => float_underflow_policy.is_some(),
        _ => false,
    }
}

const fn supports_unchecked_fusion(node: NodeDescriptor<'_>) -> bool {
    matches!(node.scalar_type, PcuScalarType::F64) && node.float_underflow_policy.is_none()
}

#[derive(Clone, Debug)]
struct Node {
    op: Op,
    shape: Vec<usize>,
    scalar_type: PcuScalarType,
    float_underflow_policy: Option<PcuFloatUnderflowPolicy>,
    numerical_mode: Option<PcuNumericalMode>,
    numerical_options: PcuNumericalOptions,
}

/// An immutable-in-practice, append-only graph builder. Value identifiers are graph-local.
#[derive(Debug)]
pub struct Graph {
    id: u64,
    nodes: Vec<Node>,
    numerical_mode: PcuNumericalMode,
    numerical_options: PcuNumericalOptions,
}

impl Default for Graph {
    /// Convenience construction with the standard infallible Default signature.
    /// # Panics
    /// Panics only if graph identities are exhausted; fallible execution frontends
    /// use `Graph::try_new` and lift `GraphIdentityExhausted` to their consumers.
    fn default() -> Self {
        Self::try_new().expect("PCU graph identity domain exhausted")
    }
}

impl Graph {
    /// Constructs an empty graph with a unique, nonwrapping process-local identity.
    ///
    /// On targets without 64-bit atomics, a checked 32-bit identity counter is
    /// widened losslessly. This permits alloc-backed tensor graphs on suitable
    /// 32-bit `no_std` targets without emulated atomic locks or unsafe global state.
    /// # Errors
    /// Returns `GraphIdentityExhausted` instead of wrapping and reusing stale IDs.
    pub fn try_new() -> Result<Self, TensorError> {
        Ok(Self {
            id: next_graph_id(&NEXT_GRAPH_ID)?,
            nodes: Vec::new(),
            numerical_mode: PcuNumericalMode::default(),
            numerical_options: PcuNumericalOptions::default(),
        })
    }

    /// Sets independent requirements captured by subsequent values; existing nodes are unchanged.
    pub const fn set_numerical_options(&mut self, options: PcuNumericalOptions) {
        self.numerical_options = options;
    }

    /// Independent defaults for subsequent graph values.
    #[must_use]
    pub const fn numerical_options(&self) -> PcuNumericalOptions {
        self.numerical_options
    }

    /// Freezes one value's numerical requirements without changing its checking mode.
    ///
    /// # Errors
    /// Returns `UnknownValue` when the value does not belong to this graph.
    pub fn set_value_numerical_options(
        &mut self,
        value: ValueId,
        options: PcuNumericalOptions,
    ) -> Result<(), TensorError> {
        self.node(value)?;
        self.nodes[value.index].numerical_options = options;
        Ok(())
    }

    /// Sets the mode captured by subsequently appended compound operations.
    pub const fn set_numerical_mode(&mut self, mode: PcuNumericalMode) {
        self.numerical_mode = mode;
    }

    /// Default mode for subsequently appended compound operations.
    #[must_use]
    pub const fn numerical_mode(&self) -> PcuNumericalMode {
        self.numerical_mode
    }

    /// Sets a compound node's exception detection mode without changing its underflow policy.
    ///
    /// # Errors
    /// Returns `UnknownValue` for invalid IDs or `UnsupportedNumericalMode` for scalar nodes.
    pub fn set_value_numerical_mode(
        &mut self,
        value: ValueId,
        mode: PcuNumericalMode,
    ) -> Result<(), TensorError> {
        self.node(value)?;
        let node = &mut self.nodes[value.index];
        if node.numerical_mode.is_none() {
            return Err(TensorError::UnsupportedNumericalMode { value, mode });
        }
        node.numerical_mode = Some(mode);
        Ok(())
    }

    /// Sets the independent underflow policy for a checked floating arithmetic node.
    ///
    /// # Errors
    /// Returns `UnknownValue` for invalid IDs or `UnsupportedScalarType` for other nodes.
    pub fn set_value_float_underflow_policy(
        &mut self,
        value: ValueId,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<(), TensorError> {
        self.node(value)?;
        let node = &mut self.nodes[value.index];
        if node.float_underflow_policy.is_none() {
            return Err(TensorError::UnsupportedScalarType {
                value,
                scalar_type: node.scalar_type,
            });
        }
        node.float_underflow_policy = Some(policy);
        Ok(())
    }

    /// Builds a stable topological CPU reference plan with checked liveness facts.
    ///
    /// # Panics
    ///
    /// Panics only if the graph's internally derived terminal outputs fail validation.
    #[must_use]
    pub fn execution_plan(&self) -> TensorExecutionPlan<'_> {
        if self.nodes.is_empty() {
            return TensorExecutionPlan {
                graph: self,
                node_order: Vec::new(),
                input_values: Vec::new(),
                output_values: Vec::new(),
                value_liveness: Vec::new(),
                index_by_value: Vec::new(),
                use_counts: Vec::new(),
                peak_live_bytes: Some(0),
            };
        }
        let outputs: Vec<_> = self
            .nodes()
            .map(|node| node.value)
            .filter(|value| {
                !self
                    .nodes
                    .iter()
                    .any(|node| operands(&node.op).any(|operand| operand == *value))
            })
            .collect();
        self.execution_plan_for_outputs(&outputs)
            .expect("terminal graph outputs always form a valid plan")
    }

    /// Builds a backend-neutral plan for the dependency closure of the requested outputs.
    ///
    /// Node order is stable append order, output order matches the request, and requested outputs
    /// remain live through the end of the plan. Checked integer arithmetic and checked F16/BF16/F32/F64
    /// elementwise nodes are retained as fault effects with their dependencies, even when
    /// they do not contribute to a requested output. Other unrelated graph nodes are omitted.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::EmptyOutputs`], [`TensorError::UnknownValue`], or
    /// [`TensorError::DuplicateOutput`] when the output selection is invalid.
    #[allow(clippy::too_many_lines)]
    pub fn execution_plan_for_outputs(
        &self,
        outputs: &[ValueId],
    ) -> Result<TensorExecutionPlan<'_>, TensorError> {
        if outputs.is_empty() {
            return Err(TensorError::EmptyOutputs);
        }
        let mut selected = vec![false; self.nodes.len()];
        let mut requested = Vec::with_capacity(outputs.len());
        for &output in outputs {
            let index = if output.graph_id == self.id {
                output.index
            } else {
                return Err(TensorError::UnknownValue(output));
            };
            if index >= self.nodes.len() {
                return Err(TensorError::UnknownValue(output));
            }
            if requested.contains(&output) {
                return Err(TensorError::DuplicateOutput(output));
            }
            requested.push(output);
            let mut stack = vec![index];
            while let Some(at) = stack.pop() {
                if selected[at] {
                    continue;
                }
                selected[at] = true;
                stack.extend(operands(&self.nodes[at].op).map(|operand| operand.index));
            }
        }
        for (index, node) in self.nodes.iter().enumerate() {
            if checked_binary(&node.op, node.scalar_type, node.float_underflow_policy)
                || node.numerical_mode.is_some()
            {
                let mut stack = vec![index];
                while let Some(at) = stack.pop() {
                    if selected[at] {
                        continue;
                    }
                    selected[at] = true;
                    stack.extend(operands(&self.nodes[at].op).map(|operand| operand.index));
                }
            }
        }
        let node_order: Vec<_> = selected
            .iter()
            .enumerate()
            .filter_map(|(index, is_selected)| {
                is_selected.then_some(ValueId {
                    graph_id: self.id,
                    index,
                })
            })
            .collect();
        let mut plan_index = vec![usize::MAX; self.nodes.len()];
        for (position, value) in node_order.iter().enumerate() {
            plan_index[value.index] = position;
        }
        let mut last_use: Vec<usize> = (0..self.nodes.len())
            .map(|index| plan_index[index])
            .collect();
        // Each edge and pin occupies one graph node/output entry, so this count is bounded by
        // allocations already represented by the graph and requested output slice.
        let mut use_counts = vec![0usize; node_order.len()];
        for (position, value) in node_order.iter().enumerate() {
            for operand in operands(&self.nodes[value.index].op) {
                last_use[operand.index] = position;
                use_counts[plan_index[operand.index]] += 1;
            }
        }
        let final_position = node_order.len() - 1;
        for output in &requested {
            last_use[output.index] = final_position;
            use_counts[plan_index[output.index]] += 1;
        }
        let input_values = node_order
            .iter()
            .filter_map(|value| matches!(self.nodes[value.index].op, Op::Input).then_some(*value))
            .collect();
        let value_liveness: Vec<_> = node_order
            .iter()
            .enumerate()
            .map(|(position, value)| TensorValueLiveness {
                value: *value,
                scalar_type: self.nodes[value.index].scalar_type,
                output_bytes: node_output_bytes(
                    &self.nodes[value.index].shape,
                    self.nodes[value.index].scalar_type,
                ),
                alignment_bytes: dense_scalar_layout(self.nodes[value.index].scalar_type)
                    .map(|(_, alignment)| alignment),
                first_live_node: position,
                last_live_node: last_use[value.index],
            })
            .collect();
        let mut ending_bytes = vec![0usize; node_order.len()];
        let mut extents_known = true;
        for life in &value_liveness {
            let Some(bytes) = life.output_bytes else {
                extents_known = false;
                break;
            };
            let Some(ending) = ending_bytes[life.last_live_node].checked_add(bytes) else {
                extents_known = false;
                break;
            };
            ending_bytes[life.last_live_node] = ending;
        }
        let peak_live_bytes = extents_known
            .then(|| {
                value_liveness.iter().enumerate().try_fold(
                    (0usize, 0usize),
                    |(live, peak), (at, life)| {
                        let next_live = live.checked_add(life.output_bytes?)?;
                        Some((next_live - ending_bytes[at], peak.max(next_live)))
                    },
                )
            })
            .flatten()
            .map(|(_, peak)| peak);
        Ok(TensorExecutionPlan {
            graph: self,
            node_order,
            input_values,
            output_values: requested,
            value_liveness,
            index_by_value: plan_index,
            use_counts,
            peak_live_bytes,
        })
    }

    /// Consumes this graph and captures lifetime-free selected execution metadata for reuse.
    ///
    /// The graph itself is retained exactly once. Selected source nodes are stored as `ValueId`s,
    /// constants remain in the graph, and fused group vectors are moved from the cold lowering
    /// plan. No graph-sized constant clone or plan reconstruction is needed to inspect this
    /// program on later calls.
    ///
    /// # Errors
    ///
    /// Returns an error when outputs are empty, duplicated, foreign, or when selected storage
    /// facts cannot be represented.
    pub fn into_selected_program(
        self,
        outputs: &[ValueId],
        rewrite_policy: TensorArithmeticRewritePolicy,
        target_arithmetic: TensorArithmeticCapability,
        grouping_policy: TensorPointwiseGroupingPolicy,
    ) -> Result<TensorOwnedSelectedProgram, TensorError> {
        let plan = self.execution_plan_for_outputs(outputs)?;
        let source_storage_constraints = plan.storage_constraints()?;
        let lowering =
            plan.select_lowering_with_grouping(rewrite_policy, target_arithmetic, grouping_policy);
        let operation_storage_constraints = lowering.operation_storage_constraints()?;
        let operation_liveness = lowering.operation_liveness()?;

        let TensorExecutionPlan {
            graph: _,
            node_order,
            input_values,
            output_values,
            value_liveness,
            index_by_value: _,
            use_counts: _,
            peak_live_bytes,
        } = plan;
        let selected_nodes = lowering.nodes.iter().map(|node| node.value).collect();
        let operation_read_only = lowering
            .operations
            .iter()
            .map(|operation| match operation {
                TensorSelectedOperation::Node(node) => is_read_only_descriptor(node.op),
                TensorSelectedOperation::FusedAddRelu { .. }
                | TensorSelectedOperation::FusedAddSub { .. }
                | TensorSelectedOperation::FusedMul { .. } => false,
            })
            .collect();
        let TensorSelectedLoweringPlan {
            graph_id: _,
            nodes: _,
            input_values: _,
            output_values: _,
            index_by_value: _,
            use_counts: selected_use_counts,
            rewritten: rewrites,
            suppressed_values,
            pointwise_groups: _,
            bounded_pointwise_groups: _,
            bounded_mul_groups: _,
            operations,
            operation_index_by_value,
            operation_use_counts,
        } = lowering;
        let operations = operations
            .into_iter()
            .map(|operation| match operation {
                TensorSelectedOperation::Node(node) => {
                    TensorOwnedSelectedOperation::Node { value: node.value }
                }
                TensorSelectedOperation::FusedAddRelu {
                    add_output,
                    relu_output,
                    left,
                    right,
                    shape: _,
                } => TensorOwnedSelectedOperation::FusedAddRelu {
                    add_output,
                    relu_output,
                    left,
                    right,
                },
                TensorSelectedOperation::FusedAddSub { group } => {
                    TensorOwnedSelectedOperation::FusedAddSub { group }
                }
                TensorSelectedOperation::FusedMul { group } => {
                    TensorOwnedSelectedOperation::FusedMul { group }
                }
            })
            .collect();

        Ok(TensorOwnedSelectedProgram {
            graph: self,
            node_order,
            input_values,
            output_values,
            value_liveness,
            peak_live_bytes,
            source_storage_constraints,
            selected_nodes,
            selected_use_counts,
            rewrites,
            suppressed_values,
            operations,
            operation_index_by_value,
            operation_use_counts,
            operation_liveness,
            operation_read_only,
            operation_storage_constraints,
        })
    }

    /// Reports backend-neutral per-value output storage facts.
    ///
    /// The total is the sum of storage for every graph value. It is not a peak/live-memory
    /// estimate; intermediate values may be released or reused by an execution plan.
    #[must_use]
    pub fn requirements(&self) -> TensorGraphRequirements<'_> {
        let values: Vec<_> = self
            .nodes()
            .map(|node| TensorValueRequirement {
                value: node.value,
                shape: node.shape,
                scalar_type: node.scalar_type,
                output_bytes: node_output_bytes(node.shape, node.scalar_type),
                alignment_bytes: dense_scalar_layout(node.scalar_type)
                    .map(|(_, alignment)| alignment),
            })
            .collect();
        let total_value_bytes = values.iter().try_fold(0usize, |total, value| {
            total.checked_add(value.output_bytes?)
        });
        TensorGraphRequirements {
            values,
            total_value_bytes,
        }
    }

    /// Assesses every node with the explicitly selected adapter in stable append order.
    #[must_use]
    pub fn assess_with<A: TensorOperationAssessor>(
        &self,
        assessor: &A,
    ) -> TensorGraphAssessment<'_> {
        TensorGraphAssessment {
            nodes: self
                .nodes()
                .map(|node| TensorNodeAssessment {
                    output_bytes: node_output_bytes(node.shape, node.scalar_type),
                    support: assessor.assess_node(self, node),
                    node,
                })
                .collect(),
        }
    }

    /// Iterates over graph nodes in their stable append order without allocating.
    #[must_use]
    pub fn nodes(&self) -> impl ExactSizeIterator<Item = NodeDescriptor<'_>> + '_ {
        self.nodes
            .iter()
            .enumerate()
            .map(|(index, _)| self.node_descriptor(index))
    }

    fn node_descriptor(&self, index: usize) -> NodeDescriptor<'_> {
        let node = &self.nodes[index];
        let value = ValueId {
            graph_id: self.id,
            index,
        };
        let op = match &node.op {
            Op::Input => OpDescriptor::Input,
            Op::Constant(tensor) => OpDescriptor::Constant(tensor),
            Op::Uniform(value) => OpDescriptor::Uniform { value },
            Op::Add(left, right) => OpDescriptor::Add {
                left: *left,
                right: *right,
            },
            Op::Sub(left, right) => OpDescriptor::Sub {
                left: *left,
                right: *right,
            },
            Op::Mul(left, right) => OpDescriptor::Mul {
                left: *left,
                right: *right,
            },
            Op::Div(left, right) => OpDescriptor::Div {
                left: *left,
                right: *right,
            },
            Op::SgdUpdate(weights, gradient, learning_rate) => OpDescriptor::SgdUpdate {
                weights: *weights,
                gradient: *gradient,
                learning_rate: *learning_rate,
            },
            Op::MatMul(left, right, transpose_left, transpose_right) => OpDescriptor::MatMul {
                left: *left,
                right: *right,
                transpose_left: *transpose_left,
                transpose_right: *transpose_right,
            },
            Op::Relu(input) => OpDescriptor::Relu { input: *input },
            Op::ReluBackward(input, upstream) => OpDescriptor::ReluBackward {
                input: *input,
                upstream: *upstream,
            },
            Op::MeanSquaredError(prediction, target) => OpDescriptor::MeanSquaredError {
                prediction: *prediction,
                target: *target,
            },
        };
        NodeDescriptor {
            value,
            op,
            shape: &node.shape,
            scalar_type: node.scalar_type,
            float_underflow_policy: node.float_underflow_policy,
            numerical_mode: node.numerical_mode,
            numerical_options: node.numerical_options,
        }
    }

    fn push(&mut self, op: Op, shape: Vec<usize>, scalar_type: PcuScalarType) -> ValueId {
        let id = ValueId {
            graph_id: self.id,
            index: self.nodes.len(),
        };
        let numerical_mode = matches!(
            op,
            Op::MatMul(..) | Op::MeanSquaredError(..) | Op::SgdUpdate(..) | Op::ReluBackward(..)
        )
        .then_some(self.numerical_mode);
        let float_underflow_policy = (matches!(
            scalar_type,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F32
                | PcuScalarType::F64
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
        ) && (matches!(
            op,
            Op::Add(..) | Op::Sub(..) | Op::Mul(..) | Op::Div(..) | Op::Relu(..)
        ) || numerical_mode.is_some()))
        .then(PcuFloatUnderflowPolicy::default);
        self.nodes.push(Node {
            op,
            shape,
            scalar_type,
            float_underflow_policy,
            numerical_mode,
            numerical_options: self.numerical_options,
        });
        id
    }

    /// Adds an input value with the supplied shape.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeOverflow`] if the shape's element count overflows.
    pub fn input(
        &mut self,
        shape: impl Into<Vec<usize>>,
        scalar_type: PcuScalarType,
    ) -> Result<ValueId, TensorError> {
        let shape = shape.into();
        element_count(&shape)?;
        Ok(self.push(Op::Input, shape, scalar_type))
    }

    /// Adds a typed input and retains its scalar type in the graph.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` when its logical extent cannot be represented.
    pub fn input_typed<T: PcuScalar>(
        &mut self,
        shape: impl Into<Vec<usize>>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.input(shape, T::TYPE).map(TensorValueId::new)
    }

    /// Checks a dynamic value's graph identity and scalar type before creating a typed handle.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for a foreign value or `ScalarTypeMismatch` when its type differs.
    pub fn typed_view<T: PcuScalar>(
        &self,
        value: ValueId,
    ) -> Result<TensorValueId<T>, TensorError> {
        let descriptor = self.node(value)?;
        if descriptor.scalar_type != T::TYPE {
            return Err(TensorError::ScalarTypeMismatch {
                value,
                expected: T::TYPE,
                actual: descriptor.scalar_type,
            });
        }
        Ok(TensorValueId::new(value))
    }

    /// Adds an owned heterogeneous tensor constant.
    pub fn constant_value(&mut self, value: TensorValue) -> ValueId {
        let scalar_type = value.scalar_type();
        let shape = value.shape().to_vec();
        self.push(Op::Constant(value), shape, scalar_type)
    }

    /// Adds a typed tensor constant.
    pub fn constant_typed<T: TensorElement>(&mut self, value: Tensor<T>) -> TensorValueId<T> {
        TensorValueId::new(self.constant_value(TensorValue::from_tensor(value)))
    }

    /// Adds a same-type typed elementwise sum.
    ///
    /// # Errors
    ///
    /// Returns graph identity or shape/type validation errors from [`Self::add`].
    pub fn add_typed<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.add(a.value, b.value).map(TensorValueId::new)
    }

    /// Adds same-type checked elementwise arithmetic using an explicit F32/F64 underflow policy.
    ///
    /// # Errors
    ///
    /// Returns graph identity, shape/type, or non-float policy errors from
    /// [`Self::add_with_underflow_policy`].
    pub fn add_typed_with_underflow_policy<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.add_with_underflow_policy(a.value, b.value, policy)
            .map(TensorValueId::new)
    }

    /// Adds a same-type typed elementwise difference.
    ///
    /// # Errors
    ///
    /// Returns graph identity or shape/type validation errors from [`Self::sub`].
    pub fn sub_typed<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.sub(a.value, b.value).map(TensorValueId::new)
    }

    /// Subtracts with an explicit checked floating-point underflow policy.
    ///
    /// # Errors
    ///
    /// Returns graph identity, shape/type, or non-float policy errors from
    /// [`Self::sub_with_underflow_policy`].
    pub fn sub_typed_with_underflow_policy<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.sub_with_underflow_policy(a.value, b.value, policy)
            .map(TensorValueId::new)
    }

    /// Adds a same-type typed elementwise product.
    ///
    /// # Errors
    ///
    /// Returns graph identity or shape/type validation errors from [`Self::mul`].
    pub fn mul_typed<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.mul(a.value, b.value).map(TensorValueId::new)
    }

    /// Multiplies with an explicit checked floating-point underflow policy.
    ///
    /// # Errors
    ///
    /// Returns graph identity, shape/type, or non-float policy errors from
    /// [`Self::mul_with_underflow_policy`].
    pub fn mul_typed_with_underflow_policy<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.mul_with_underflow_policy(a.value, b.value, policy)
            .map(TensorValueId::new)
    }

    /// Divides same-type F32/F64 tensors elementwise using the default checked-float policy.
    ///
    /// # Errors
    ///
    /// Returns graph identity, shape, type, or unsupported-scalar errors.
    pub fn div_typed<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.div(a.value, b.value).map(TensorValueId::new)
    }

    /// Divides same-type F32/F64 tensors with an explicit checked-float underflow policy.
    ///
    /// # Errors
    ///
    /// Returns graph identity, shape, type, or unsupported-scalar errors.
    pub fn div_typed_with_underflow_policy<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.div_with_underflow_policy(a.value, b.value, policy)
            .map(TensorValueId::new)
    }

    /// Adds a typed `ReLU` operation.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` if the handle is not from this graph.
    pub fn relu_typed<T: PcuScalar>(
        &mut self,
        value: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.relu(value.value).map(TensorValueId::new)
    }

    /// Returns the same typed graph handle without adding an operation.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for a foreign handle or `ScalarTypeMismatch` if its metadata does
    /// not match `T`.
    pub fn identity_typed<T: PcuScalar>(
        &self,
        value: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.typed_view(value.erase())
    }

    /// Adds a same-type typed matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns graph identity or matrix-shape errors from [`Self::matmul`].
    pub fn matmul_typed<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.matmul(a.value, b.value).map(TensorValueId::new)
    }

    /// Adds typed mean-squared error, retaining the declared scalar result identity.
    ///
    /// The current primitive supports F32 and F64. Other scalar representations reject explicitly;
    /// a typed carrier alone does not establish a backend arithmetic implementation.
    ///
    /// # Errors
    /// Returns graph identity, shape, or unsupported-scalar errors from [`Self::mean_squared_error`].
    pub fn mean_squared_error_typed<T: PcuScalar>(
        &mut self,
        prediction: TensorValueId<T>,
        target: TensorValueId<T>,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.mean_squared_error(prediction.value, target.value)
            .map(TensorValueId::new)
    }

    /// Adds typed SGD with a finite F32 rate stored in the immutable graph descriptor.
    ///
    /// F32 and F64 are supported; the finite F32 rate widens exactly for F64 arithmetic.
    /// A dynamic learning rate needs a resource binding;
    /// callers must not snapshot a changing rate into a reusable prepared graph.
    ///
    /// # Errors
    /// Returns graph identity, shape, scalar or rate errors from [`Self::sgd_update`].
    pub fn sgd_update_typed<T: PcuScalar>(
        &mut self,
        weights: TensorValueId<T>,
        gradient: TensorValueId<T>,
        learning_rate: f32,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.sgd_update(weights.value, gradient.value, learning_rate)
            .map(TensorValueId::new)
    }

    /// Adds a same-type typed matrix multiplication with logical transposes.
    ///
    /// # Errors
    ///
    /// Returns graph identity or matrix-shape errors from [`Self::matmul_transposed`].
    pub fn matmul_transposed_typed<T: PcuScalar>(
        &mut self,
        a: TensorValueId<T>,
        b: TensorValueId<T>,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.matmul_transposed(a.value, b.value, transpose_left, transpose_right)
            .map(TensorValueId::new)
    }

    /// Adds a compact graph-level constant with the supplied logical shape.
    ///
    /// Unlike [`Tensor::splat`], this stores only the scalar in the graph. The CPU reference
    /// evaluator materializes the dense logical tensor when executing it. Graph requirement,
    /// liveness, and storage metadata still report the full dense output size; a backend may use
    /// scalar physical storage only when its own representation explicitly supports that choice.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeOverflow`] if the logical element count overflows.
    pub fn uniform_value(
        &mut self,
        shape: impl Into<Vec<usize>>,
        value: TensorScalarValue,
    ) -> Result<ValueId, TensorError> {
        let shape = shape.into();
        element_count(&shape)?;
        Ok(self.push(Op::Uniform(value), shape, value.scalar_type()))
    }

    /// Adds a uniform value with compile-time scalar identity.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` when the logical extent cannot be represented.
    pub fn uniform_typed<T: TensorElement>(
        &mut self,
        shape: impl Into<Vec<usize>>,
        value: T,
    ) -> Result<TensorValueId<T>, TensorError> {
        self.uniform_value(shape, T::into_scalar(value))
            .map(TensorValueId::new)
    }

    /// Adds two values with identical shapes.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for an invalid value ID, `ShapeMismatch` for different shapes,
    /// `ScalarTypeMismatch` for different scalar identities, or `UnsupportedScalarType` when
    /// the scalar has no elementwise arithmetic semantics.
    pub fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Add)
    }

    /// Adds supported floating tensors using an explicit checked floating-point underflow policy.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue`, `ShapeMismatch`, `ScalarTypeMismatch`, or
    /// `UnsupportedScalarType` if the operands have no checked binary floating contract.
    pub fn add_with_underflow_policy(
        &mut self,
        a: ValueId,
        b: ValueId,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<ValueId, TensorError> {
        self.same_shape_binary_with_policy(a, b, BinaryOp::Add, Some(policy))
    }

    /// Subtracts two values with identical shapes, without broadcasting.
    ///
    /// # Errors
    ///
    /// Returns the same graph identity, shape, and scalar-semantics errors as [`Self::add`].
    pub fn sub(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Sub)
    }

    /// Subtracts supported floating tensors using an explicit checked floating-point underflow policy.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue`, `ShapeMismatch`, `ScalarTypeMismatch`, or
    /// `UnsupportedScalarType` if the operands have no checked binary floating contract.
    pub fn sub_with_underflow_policy(
        &mut self,
        a: ValueId,
        b: ValueId,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<ValueId, TensorError> {
        self.same_shape_binary_with_policy(a, b, BinaryOp::Sub, Some(policy))
    }

    /// Multiplies two values elementwise with identical shapes, without broadcasting.
    ///
    /// # Errors
    ///
    /// Returns the same graph identity, shape, and scalar-semantics errors as [`Self::add`].
    pub fn mul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Mul)
    }

    /// Multiplies supported floating tensors using an explicit checked floating-point underflow policy.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue`, `ShapeMismatch`, `ScalarTypeMismatch`, or
    /// `UnsupportedScalarType` if the operands have no checked binary floating contract.
    pub fn mul_with_underflow_policy(
        &mut self,
        a: ValueId,
        b: ValueId,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<ValueId, TensorError> {
        self.same_shape_binary_with_policy(a, b, BinaryOp::Mul, Some(policy))
    }

    /// Divides supported floating tensors elementwise using checked rounding and the default underflow policy.
    ///
    /// # Errors
    ///
    /// Returns shape/type errors, or `UnsupportedScalarType` unless both operands have a checked binary floating contract.
    pub fn div(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Div)
    }

    /// Divides supported floating tensors with an explicit checked floating-point underflow policy.
    ///
    /// # Errors
    ///
    /// Returns shape/type errors, or `UnsupportedScalarType` unless both operands have a checked binary floating contract.
    pub fn div_with_underflow_policy(
        &mut self,
        a: ValueId,
        b: ValueId,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<ValueId, TensorError> {
        self.same_shape_binary_with_policy(a, b, BinaryOp::Div, Some(policy))
    }

    /// Adds an elementwise SGD update, `weights - learning_rate * gradient`.
    ///
    /// F32/F64 strict execution checks and rounds the product, then the subtraction. It does
    /// not contract them into an FMA. Fault order is logical element, then multiply/subtract.
    /// IEEE 754-derived rounding and underflow classification come from the scalar contract;
    /// this exception granularity and first-fault order are PCU rules. The immutable F32 rate
    /// widens exactly for F64. Boundary execution still requires separate backend admission.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for invalid operands,
    /// [`TensorError::ShapeMismatch`] for differing shapes, or
    /// [`TensorError::InvalidLearningRate`] when the rate is not finite.
    pub fn sgd_update(
        &mut self,
        weights: ValueId,
        gradient: ValueId,
        learning_rate: f32,
    ) -> Result<ValueId, TensorError> {
        let weights_shape = self.shape(weights)?.to_vec();
        let gradient_shape = self.shape(gradient)?.to_vec();
        if weights_shape != gradient_shape {
            return Err(TensorError::ShapeMismatch {
                left: weights_shape,
                right: gradient_shape,
            });
        }
        let weights_type = self.nodes[weights.index].scalar_type;
        let gradient_type = self.nodes[gradient.index].scalar_type;
        if weights_type != gradient_type {
            return Err(TensorError::ScalarTypeMismatch {
                value: gradient,
                expected: weights_type,
                actual: gradient_type,
            });
        }
        if !learning_rate.is_finite() {
            return Err(TensorError::InvalidLearningRate);
        }
        let scalar_type = self.nodes[weights.index].scalar_type;
        self.require_float(weights)?;
        self.require_float(gradient)?;
        Ok(self.push(
            Op::SgdUpdate(weights, gradient, learning_rate),
            weights_shape,
            scalar_type,
        ))
    }

    fn same_shape_binary(
        &mut self,
        a: ValueId,
        b: ValueId,
        operation: BinaryOp,
    ) -> Result<ValueId, TensorError> {
        self.same_shape_binary_with_policy(a, b, operation, None)
    }

    fn same_shape_binary_with_policy(
        &mut self,
        a: ValueId,
        b: ValueId,
        operation: BinaryOp,
        explicit_policy: Option<PcuFloatUnderflowPolicy>,
    ) -> Result<ValueId, TensorError> {
        let sa = self.shape(a)?.to_vec();
        let sb = self.shape(b)?.to_vec();
        if sa != sb {
            return Err(TensorError::ShapeMismatch {
                left: sa,
                right: sb,
            });
        }
        let left_type = self.nodes[a.index].scalar_type;
        let right_type = self.nodes[b.index].scalar_type;
        if left_type != right_type {
            return Err(TensorError::ScalarTypeMismatch {
                value: b,
                expected: left_type,
                actual: right_type,
            });
        }
        self.require_elementwise_numeric(a)?;
        let is_float_binary = matches!(
            left_type,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F32
                | PcuScalarType::F64
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
        ) && matches!(
            operation,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
        );
        if matches!(operation, BinaryOp::Div) && !is_float_binary {
            return Err(TensorError::UnsupportedScalarType {
                value: a,
                scalar_type: left_type,
            });
        }
        if explicit_policy.is_some() && !is_float_binary {
            return Err(TensorError::UnsupportedScalarType {
                value: a,
                scalar_type: left_type,
            });
        }
        let op = match operation {
            BinaryOp::Add => Op::Add(a, b),
            BinaryOp::Sub => Op::Sub(a, b),
            BinaryOp::Mul => Op::Mul(a, b),
            BinaryOp::Div => Op::Div(a, b),
        };
        let result = self.push(op, self.nodes[a.index].shape.clone(), left_type);
        self.nodes[result.index].float_underflow_policy = if is_float_binary {
            Some(explicit_policy.unwrap_or_default())
        } else {
            None
        };
        Ok(result)
    }

    fn require_float(&self, value: ValueId) -> Result<PcuScalarType, TensorError> {
        let scalar_type = self.node(value)?.scalar_type;
        if !matches!(scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
            return Err(TensorError::UnsupportedScalarType { value, scalar_type });
        }
        Ok(scalar_type)
    }

    fn require_elementwise_numeric(&self, value: ValueId) -> Result<PcuScalarType, TensorError> {
        let scalar_type = self.node(value)?.scalar_type;
        if matches!(
            scalar_type,
            PcuScalarType::Bool
                | PcuScalarType::I4
                | PcuScalarType::U4
                | PcuScalarType::F128
                | PcuScalarType::F256
        ) {
            return Err(TensorError::UnsupportedScalarType { value, scalar_type });
        }
        Ok(scalar_type)
    }

    /// Adds a rank-two matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for invalid IDs, `MatMulShape` for incompatible dimensions,
    /// `ScalarTypeMismatch` for different scalar identities, `UnsupportedScalarType` for types
    /// without matrix arithmetic semantics, or `ShapeOverflow` for an unrepresentable extent.
    pub fn matmul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.matmul_transposed(a, b, false, false)
    }

    /// Adds a rank-two matrix multiplication with optional logical transposes on its operands.
    /// Inputs remain in their original row-major shapes; no transpose tensor is materialized.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for invalid IDs, `MatMulShape` for incompatible/non-matrix shapes,
    /// `ScalarTypeMismatch` for different scalar identities, `UnsupportedScalarType` for types
    /// without matrix arithmetic semantics, or `ShapeOverflow` for an unrepresentable extent.
    pub fn matmul_transposed(
        &mut self,
        a: ValueId,
        b: ValueId,
        transpose_left: bool,
        transpose_right: bool,
    ) -> Result<ValueId, TensorError> {
        let sa = self.shape(a)?.to_vec();
        let sb = self.shape(b)?.to_vec();
        if sa.len() != 2 || sb.len() != 2 {
            return Err(TensorError::MatMulShape {
                left: sa,
                right: sb,
            });
        }
        let (left_rows, left_inner) = if transpose_left {
            (sa[1], sa[0])
        } else {
            (sa[0], sa[1])
        };
        let (right_inner, right_columns) = if transpose_right {
            (sb[1], sb[0])
        } else {
            (sb[0], sb[1])
        };
        if left_inner != right_inner {
            return Err(TensorError::MatMulShape {
                left: sa,
                right: sb,
            });
        }
        let shape = vec![left_rows, right_columns];
        element_count(&shape)?;
        let left_type = self.nodes[a.index].scalar_type;
        let right_type = self.nodes[b.index].scalar_type;
        if left_type != right_type {
            return Err(TensorError::ScalarTypeMismatch {
                value: b,
                expected: left_type,
                actual: right_type,
            });
        }
        self.require_float(a)?;
        Ok(self.push(
            Op::MatMul(a, b, transpose_left, transpose_right),
            shape,
            left_type,
        ))
    }

    /// Adds an elementwise `ReLU` operation.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for a foreign ID or `UnsupportedScalarType` when the scalar has no
    /// elementwise maximum semantics.
    pub fn relu(&mut self, x: ValueId) -> Result<ValueId, TensorError> {
        let shape = self.shape(x)?.to_vec();
        let scalar_type = self.require_elementwise_numeric(x)?;
        Ok(self.push(Op::Relu(x), shape, scalar_type))
    }

    /// Adds the `ReLU` derivative `input > 0 ? upstream : 0` elementwise.
    ///
    /// # Errors
    ///
    /// Returns graph identity, shape, type, or unsupported-type errors when inputs are invalid.
    ///
    /// The strict comparison matches reverse-mode differentiation in `Execution::gradients`:
    /// either signed zero produces positive zero; nonfinite operands are faults.
    pub fn relu_backward(
        &mut self,
        input: ValueId,
        upstream: ValueId,
    ) -> Result<ValueId, TensorError> {
        let input_shape = self.shape(input)?.to_vec();
        let upstream_shape = self.shape(upstream)?.to_vec();
        if input_shape != upstream_shape {
            return Err(TensorError::ShapeMismatch {
                left: input_shape,
                right: upstream_shape,
            });
        }
        let input_type = self.nodes[input.index].scalar_type;
        let upstream_type = self.nodes[upstream.index].scalar_type;
        if input_type != upstream_type {
            return Err(TensorError::ScalarTypeMismatch {
                value: upstream,
                expected: input_type,
                actual: upstream_type,
            });
        }
        let scalar_type = self.node(input)?.scalar_type;
        if !matches!(
            scalar_type,
            PcuScalarType::F16
                | PcuScalarType::BF16
                | PcuScalarType::F32
                | PcuScalarType::F64
                | PcuScalarType::F8E4M3FN
                | PcuScalarType::F8E5M2
        ) {
            return Err(TensorError::UnsupportedScalarType {
                value: input,
                scalar_type,
            });
        }
        Ok(self.push(
            Op::ReluBackward(input, upstream),
            input_shape,
            self.nodes[input.index].scalar_type,
        ))
    }

    /// Adds a scalar mean-squared-error operation over equally shaped values.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID,
    /// [`TensorError::ShapeMismatch`] for differing or empty shapes.
    pub fn mean_squared_error(
        &mut self,
        prediction: ValueId,
        target: ValueId,
    ) -> Result<ValueId, TensorError> {
        let p = self.shape(prediction)?.to_vec();
        let t = self.shape(target)?.to_vec();
        if p != t {
            return Err(TensorError::ShapeMismatch { left: p, right: t });
        }
        let prediction_type = self.nodes[prediction.index].scalar_type;
        let target_type = self.nodes[target.index].scalar_type;
        if prediction_type != target_type {
            return Err(TensorError::ScalarTypeMismatch {
                value: target,
                expected: prediction_type,
                actual: target_type,
            });
        }
        self.require_float(prediction)?;
        self.require_float(target)?;
        if element_count(&p)? == 0 {
            return Err(TensorError::ShapeMismatch { left: p, right: t });
        }
        Ok(self.push(
            Op::MeanSquaredError(prediction, target),
            vec![],
            prediction_type,
        ))
    }

    /// Append a bounded reverse-mode graph for a scalar mean-squared-error root.
    ///
    /// The returned vector has one entry per node that existed before this call; each entry is
    /// the graph value holding that node's gradient, if the node contributes to `loss`. The
    /// resulting graph is ordinary tensor IR and can be assessed by a selected backend. This
    /// profile supports Add, Sub, Mul, `ReLU`, and transpose-aware `MatMul` in the loss dependency
    /// closure. Derived operations inherit their source operation's numerical options,
    /// compound mode and underflow policy, independent of later graph defaults. Defaults are
    /// restored even if constructing a derived node fails. Nested MSE, division and second
    /// derivatives are rejected before any node is appended.
    ///
    /// # Errors
    ///
    /// Returns an invalid-root or unsupported-gradient error without changing the graph, or a
    /// shape/extent error while constructing a gradient node.
    pub fn backward_mse(&mut self, loss: ValueId) -> Result<Vec<Option<ValueId>>, TensorError> {
        gradients::backward_mse(self, loss)
    }

    /// Append only the reverse graph needed for the derivative with respect to `target`.
    ///
    /// Uses the same bounded operation and numerical contract as [`Self::backward_mse`].
    /// Forward checked effects remain observable. Unrequested derivative branches are
    /// never appended, so their arithmetic cannot introduce faults or device work.
    /// This cold transform does not evaluate input values or imply host execution.
    ///
    /// # Errors
    /// Rejects a foreign/disconnected target, invalid root or unsupported source
    /// operation before appending nodes; shape/extent failures can arise during construction.
    pub fn backward_mse_for(
        &mut self,
        loss: ValueId,
        target: ValueId,
    ) -> Result<ValueId, TensorError> {
        gradients::backward_mse_for(self, loss, target)
    }

    /// Append one shared reverse pass for the requested MSE derivatives.
    ///
    /// Results follow `targets` order. Shared ancestors are differentiated once;
    /// branches that cannot reach any requested target are never appended.
    /// Forward checked effects and each source operation's numerical contract
    /// remain unchanged. Repeated target IDs refer to the same derivative value.
    /// This is a cold graph transform, not host execution or repeated training.
    ///
    /// # Errors
    /// Rejects empty targets, foreign/disconnected targets, invalid roots and
    /// unsupported source operations before appending nodes. Shape/extent errors
    /// can arise during derivative construction.
    pub fn backward_mse_for_targets(
        &mut self,
        loss: ValueId,
        targets: &[ValueId],
    ) -> Result<Vec<ValueId>, TensorError> {
        gradients::backward_mse_for_targets(self, loss, targets)
    }

    /// Returns the shape of a graph value.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] if `id` does not belong to this graph.
    pub fn shape(&self, id: ValueId) -> Result<&[usize], TensorError> {
        self.nodes
            .get(id.index)
            .filter(|_| id.graph_id == self.id)
            .map(|n| n.shape.as_slice())
            .ok_or(TensorError::UnknownValue(id))
    }

    /// Borrows the graph descriptor for one value.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] when `id` does not belong to this graph.
    pub fn node(&self, id: ValueId) -> Result<NodeDescriptor<'_>, TensorError> {
        if id.graph_id != self.id {
            return Err(TensorError::UnknownValue(id));
        }
        self.nodes
            .get(id.index)
            .map(|_| self.node_descriptor(id.index))
            .ok_or(TensorError::UnknownValue(id))
    }

    /// Executes only reference operations with implemented checked numerical contracts.
    ///
    /// Strict F32/F64 `MatMul` checks separate multiply/add steps in increasing reduction order;
    /// strict SGD checks a multiply then subtraction for each logical element.
    /// Boundary compounds and other raw reference operations have no checked certificate.
    /// Permitting backend-defined arithmetic or optimized precision does not require weakening
    /// this route's ordered checks or precision. Portable reproducibility remains unsupported.
    ///
    /// # Errors
    /// Returns `UnsupportedNumericalMode` for unsupported compound contracts, or the ordinary
    /// input validation and checked arithmetic errors from [`Self::evaluate`].
    pub fn evaluate_checked(
        &self,
        inputs: &[(ValueId, TensorValue)],
    ) -> Result<Execution, TensorError> {
        for node in self.nodes() {
            if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
                return Err(TensorError::UnsupportedNumericalOptions {
                    value: node.value,
                    options: node.numerical_options,
                });
            }
            if let Some(mode) = node.numerical_mode
                && !matches!(node.op, OpDescriptor::ReluBackward { .. })
                && (mode != PcuNumericalMode::Strict
                    || !matches!(
                        node.op,
                        OpDescriptor::MatMul { .. }
                            | OpDescriptor::SgdUpdate { .. }
                            | OpDescriptor::MeanSquaredError { .. }
                            | OpDescriptor::ReluBackward { .. }
                    ))
            {
                return Err(TensorError::UnsupportedNumericalMode {
                    value: node.value,
                    mode,
                });
            }
        }
        self.evaluate(inputs)
    }

    /// Evaluates the graph using the supplied input tensors.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, extra, wrongly typed/shaped inputs, or an
    /// unsupported arithmetic operation in the graph.
    ///
    /// Boundary compounds use historical raw arithmetic and are not certified checked.
    /// Use [`Self::evaluate_checked`] when a checked reference contract is required.
    #[allow(clippy::too_many_lines)] // Keeps reference operation dispatch and validation together.
    pub fn evaluate(&self, inputs: &[(ValueId, TensorValue)]) -> Result<Execution, TensorError> {
        for node in self.nodes() {
            if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
                return Err(TensorError::UnsupportedNumericalOptions {
                    value: node.value,
                    options: node.numerical_options,
                });
            }
            if node.numerical_mode == Some(PcuNumericalMode::Strict)
                && !matches!(
                    node.op,
                    OpDescriptor::MatMul { .. }
                        | OpDescriptor::SgdUpdate { .. }
                        | OpDescriptor::MeanSquaredError { .. }
                        | OpDescriptor::ReluBackward { .. }
                )
            {
                return Err(TensorError::UnsupportedNumericalMode {
                    value: node.value,
                    mode: PcuNumericalMode::Strict,
                });
            }
        }
        let mut supplied: Vec<Option<&TensorValue>> = vec![None; self.nodes.len()];
        for (id, tensor) in inputs {
            let node = self
                .nodes
                .get(id.index)
                .filter(|_| id.graph_id == self.id)
                .ok_or(TensorError::ExtraInput(*id))?;
            if !matches!(node.op, Op::Input) {
                return Err(TensorError::ExtraInput(*id));
            }
            if supplied[id.index].replace(tensor).is_some() {
                return Err(TensorError::DuplicateInput(*id));
            }
            if tensor.scalar_type() != node.scalar_type {
                return Err(TensorError::ScalarTypeMismatch {
                    value: *id,
                    expected: node.scalar_type,
                    actual: tensor.scalar_type(),
                });
            }
            if tensor.shape() != node.shape {
                return Err(TensorError::ShapeMismatch {
                    left: tensor.shape().to_vec(),
                    right: node.shape.clone(),
                });
            }
        }
        let mut values: Vec<TensorValue> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let id = ValueId {
                graph_id: self.id,
                index,
            };
            let value = match &node.op {
                Op::Input => {
                    let value = supplied[id.index].ok_or(TensorError::MissingInput(id))?;
                    value.clone()
                }
                Op::Constant(tensor) => tensor.clone(),
                Op::Uniform(scalar) => scalar.splat(node.shape.clone())?,
                Op::Add(a, b) => binary_value(
                    &values[a.index],
                    &values[b.index],
                    id,
                    BinaryOp::Add,
                    node.float_underflow_policy,
                )?,
                Op::Sub(a, b) => binary_value(
                    &values[a.index],
                    &values[b.index],
                    id,
                    BinaryOp::Sub,
                    node.float_underflow_policy,
                )?,
                Op::Mul(a, b) => binary_value(
                    &values[a.index],
                    &values[b.index],
                    id,
                    BinaryOp::Mul,
                    node.float_underflow_policy,
                )?,
                Op::Div(a, b) => binary_value(
                    &values[a.index],
                    &values[b.index],
                    id,
                    BinaryOp::Div,
                    node.float_underflow_policy,
                )?,
                Op::SgdUpdate(weights, gradient, learning_rate) => sgd_value(
                    &values[weights.index],
                    &values[gradient.index],
                    *learning_rate,
                    id,
                    node.numerical_mode.unwrap_or_default(),
                    node.float_underflow_policy.unwrap_or_default(),
                )?,
                Op::MatMul(a, b, transpose_left, transpose_right) => matmul_value(
                    &values[a.index],
                    &values[b.index],
                    *transpose_left,
                    *transpose_right,
                    id,
                    node.numerical_mode.unwrap_or_default(),
                    node.float_underflow_policy.unwrap_or_default(),
                )?,
                Op::Relu(input) => relu_value(
                    &values[input.index],
                    id,
                    node.float_underflow_policy.unwrap_or_default(),
                )?,
                Op::ReluBackward(input, upstream) => relu_backward_value(
                    &values[input.index],
                    &values[upstream.index],
                    id,
                    node.float_underflow_policy.unwrap_or_default(),
                )?,
                Op::MeanSquaredError(prediction, target) => mean_squared_error_value(
                    &values[prediction.index],
                    &values[target.index],
                    id,
                    node.numerical_mode.unwrap_or_default(),
                    node.float_underflow_policy.unwrap_or_default(),
                )?,
            };
            values.push(value);
        }
        Ok(Execution {
            graph_id: self.id,
            node_count: self.nodes.len(),
            values,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Execution {
    graph_id: u64,
    node_count: usize,
    values: Vec<TensorValue>,
}
impl Execution {
    /// Returns the evaluated tensor for a graph value.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] if `id` does not belong to this execution.
    pub fn value(&self, id: ValueId) -> Result<&TensorValue, TensorError> {
        self.values
            .get(id.index)
            .filter(|_| id.graph_id == self.graph_id)
            .ok_or(TensorError::UnknownValue(id))
    }

    /// Returns one result with its checked typed host representation.
    ///
    /// # Errors
    ///
    /// Returns `UnknownValue` for an invalid ID or `ScalarTypeMismatch` when the stored type
    /// differs from `T`.
    pub fn value_typed<T: TensorElement>(&self, id: ValueId) -> Result<&Tensor<T>, TensorError> {
        let value = self.value(id)?;
        value
            .as_typed::<T>()
            .map_err(|_| TensorError::ScalarTypeMismatch {
                value: id,
                expected: T::TYPE,
                actual: value.scalar_type(),
            })
    }

    /// Reverse-mode derivative of the selected scalar output with respect to every graph value.
    ///
    /// # Errors
    ///
    /// Returns an error if the graph differs from the evaluated graph, the output ID is
    /// invalid, or the selected output is not scalar.
    #[allow(clippy::too_many_lines)] // Keeps the operation-by-operation reference derivatives together.
    pub fn gradients(
        &self,
        graph: &Graph,
        output: ValueId,
    ) -> Result<Vec<Option<Tensor>>, TensorError> {
        if self.graph_id != graph.id {
            return Err(TensorError::WrongGraph);
        }
        if self.node_count != graph.nodes.len() || self.values.len() != graph.nodes.len() {
            return Err(TensorError::GraphChanged);
        }
        graph.shape(output)?;
        let out = self.value_typed::<f32>(output)?;
        if !out.shape().is_empty() {
            return Err(TensorError::LossMustBeScalar(out.shape().to_vec()));
        }
        let values: Vec<Option<&Tensor>> = self
            .values
            .iter()
            .map(|value| value.as_typed::<f32>().ok())
            .collect();
        let mut grads: Vec<Option<Tensor>> = vec![None; graph.nodes.len()];
        grads[output.index] = Some(Tensor::scalar(1.0));
        for i in (0..=output.index).rev() {
            let Some(grad) = grads[i].clone() else {
                continue;
            };
            let node = &graph.nodes[i];
            match node.op {
                Op::Input | Op::Constant(_) | Op::Uniform(_) => {}
                Op::Add(a, b) => {
                    accumulate(&mut grads, a, grad.clone());
                    accumulate(&mut grads, b, grad);
                }
                Op::Sub(a, b) => {
                    accumulate(&mut grads, a, grad.clone());
                    accumulate(
                        &mut grads,
                        b,
                        Tensor::new(grad.shape.clone(), grad.data.iter().map(|v| -*v).collect())?,
                    );
                }
                Op::Mul(a, b) => {
                    let left = execution_f32_value(&values, graph, a)?;
                    let right = execution_f32_value(&values, graph, b)?;
                    let (left_gradient, right_gradient) =
                        elementwise_mul_gradients(left, right, &grad)?;
                    accumulate(&mut grads, a, left_gradient);
                    accumulate(&mut grads, b, right_gradient);
                }
                Op::Div(..) | Op::ReluBackward(..) => {
                    return Err(TensorError::UnsupportedGradient(ValueId {
                        graph_id: graph.id,
                        index: i,
                    }));
                }
                Op::SgdUpdate(weights, update_gradient, learning_rate) => {
                    let input_gradient = grad.clone();
                    let scaled_gradient = Tensor::new(
                        grad.shape.clone(),
                        grad.data
                            .iter()
                            .map(|value| -learning_rate * value)
                            .collect(),
                    )?;
                    accumulate(&mut grads, weights, input_gradient);
                    accumulate(&mut grads, update_gradient, scaled_gradient);
                }
                Op::Relu(x) => {
                    let g = Tensor::new(
                        node_shape(graph, x)?,
                        execution_f32_value(&values, graph, x)?
                            .data
                            .iter()
                            .zip(grad.data)
                            .map(|(v, d)| if *v > 0.0 { d } else { 0.0 })
                            .collect(),
                    )?;
                    accumulate(&mut grads, x, g);
                }
                Op::MeanSquaredError(a, b) => {
                    let prediction = execution_f32_value(&values, graph, a)?;
                    let target = execution_f32_value(&values, graph, b)?;
                    let divisor = constants::count_f32(prediction.data.len());
                    let gp = Tensor::new(
                        prediction.shape.clone(),
                        prediction
                            .data
                            .iter()
                            .zip(&target.data)
                            .map(|(predicted, expected)| {
                                2.0 * (predicted - expected) / divisor * grad.data[0]
                            })
                            .collect(),
                    )?;
                    let gt = Tensor::new(
                        target.shape.clone(),
                        prediction
                            .data
                            .iter()
                            .zip(&target.data)
                            .map(|(predicted, expected)| {
                                2.0 * (expected - predicted) / divisor * grad.data[0]
                            })
                            .collect(),
                    )?;
                    accumulate(&mut grads, a, gp);
                    accumulate(&mut grads, b, gt);
                }
                Op::MatMul(a, b, transpose_left, transpose_right) => {
                    let left = execution_f32_value(&values, graph, a)?;
                    let right = execution_f32_value(&values, graph, b)?;
                    let (left_gradient, right_gradient) =
                        matmul_gradients(left, right, &grad, transpose_left, transpose_right)?;
                    accumulate(&mut grads, a, left_gradient);
                    accumulate(&mut grads, b, right_gradient);
                }
            }
        }
        Ok(grads)
    }
}

fn elementwise_mul_gradients(
    left: &Tensor,
    right: &Tensor,
    gradient: &Tensor,
) -> Result<(Tensor, Tensor), TensorError> {
    let left_gradient = Tensor::new(
        gradient.shape.clone(),
        gradient
            .data
            .iter()
            .zip(&right.data)
            .map(|(g, r)| g * r)
            .collect(),
    )?;
    let right_gradient = Tensor::new(
        gradient.shape.clone(),
        gradient
            .data
            .iter()
            .zip(&left.data)
            .map(|(g, l)| g * l)
            .collect(),
    )?;
    Ok((left_gradient, right_gradient))
}

// Match the reference forward path's non-fused accumulation semantics.
#[allow(clippy::suboptimal_flops)]
fn matmul_gradients(
    left: &Tensor,
    right: &Tensor,
    output_gradient: &Tensor,
    transpose_left: bool,
    transpose_right: bool,
) -> Result<(Tensor, Tensor), TensorError> {
    let (rows, inner) = if transpose_left {
        (left.shape[1], left.shape[0])
    } else {
        (left.shape[0], left.shape[1])
    };
    let columns = if transpose_right {
        right.shape[0]
    } else {
        right.shape[1]
    };
    let mut left_gradient = vec![0.0; left.data.len()];
    let mut right_gradient = vec![0.0; right.data.len()];
    for row in 0..rows {
        for column in 0..columns {
            let gradient = output_gradient.data[row * columns + column];
            for inner_index in 0..inner {
                let li = if transpose_left {
                    inner_index * left.shape[1] + row
                } else {
                    row * left.shape[1] + inner_index
                };
                let ri = if transpose_right {
                    column * right.shape[1] + inner_index
                } else {
                    inner_index * right.shape[1] + column
                };
                left_gradient[li] += gradient * right.data[ri];
                right_gradient[ri] += left.data[li] * gradient;
            }
        }
    }
    Ok((
        Tensor::new(left.shape.clone(), left_gradient)?,
        Tensor::new(right.shape.clone(), right_gradient)?,
    ))
}

fn node_shape(graph: &Graph, id: ValueId) -> Result<Vec<usize>, TensorError> {
    Ok(graph.shape(id)?.to_vec())
}
fn accumulate(grads: &mut [Option<Tensor>], id: ValueId, add: Tensor) {
    if let Some(current) = &mut grads[id.index] {
        for (x, y) in current.data.iter_mut().zip(add.data) {
            *x += y;
        }
    } else {
        grads[id.index] = Some(add);
    }
}

#[cfg(test)]
#[path = "tensor/tests.rs"]
mod tests;
