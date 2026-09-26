//! Small standalone tensor dialect with a deterministic CPU reference evaluator.
//!
//! This crate is intentionally outside `fusion-pcu`: tensor semantics belong to a dialect, not
//! to the generic coprocessor IR. The current graph supports f32 inputs/constants, elementwise
//! add/subtract/multiply, SGD updates, matrix multiplication, `ReLU`, and mean-squared error. It can append
//! a backward graph for the supported MSE-rooted subset. It does not implement broadcasting,
//! batching, convolution, views, mixed precision, optimizers, or serialization; selected backend
//! adapters provide device execution separately.

use std::fmt;
use std::sync::atomic::{
    AtomicU64,
    Ordering,
};

#[path = "tensor/storage.rs"]
mod storage;
pub use storage::{
    TensorGraphRequirements,
    TensorStorageConstraint,
    TensorStorageValidationError,
    TensorValueLiveness,
    TensorValueRequirement,
};
use storage::node_output_bytes;

#[path = "tensor/feedback.rs"]
mod feedback;
pub use feedback::{
    TensorFeedbackBinding,
    TensorFeedbackInput,
    TensorFeedbackPlan,
};

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

fn next_graph_id() -> u64 {
    NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
    known_uniform_value: Option<f32>,
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl Tensor {
    /// Creates a tensor after checking that its data length matches its shape.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeOverflow`] if the element count overflows, or
    /// [`TensorError::DataLength`] if `data` has the wrong number of elements.
    pub fn new(shape: impl Into<Vec<usize>>, data: Vec<f32>) -> Result<Self, TensorError> {
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
    pub fn splat(shape: impl Into<Vec<usize>>, value: f32) -> Result<Self, TensorError> {
        let shape = shape.into();
        let len = element_count(&shape)?;
        Ok(Self {
            shape,
            data: vec![value; len],
            known_uniform_value: (len != 0).then_some(value),
        })
    }

    #[must_use]
    pub fn scalar(value: f32) -> Self {
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
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Returns a uniform element value when the tensor was constructed with a uniformity-aware
    /// constructor. General tensors created with [`Self::new`] remain unknown until an
    /// optimization elects to inspect their data.
    #[must_use]
    pub const fn known_uniform_value(&self) -> Option<f32> {
        self.known_uniform_value
    }

    /// Consumes the tensor and returns its contiguous data without copying it.
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorError {
    ShapeOverflow,
    DataLength { expected: usize, actual: usize },
    UnknownValue(ValueId),
    ShapeMismatch { left: Vec<usize>, right: Vec<usize> },
    MatMulShape { left: Vec<usize>, right: Vec<usize> },
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
impl std::error::Error for TensorError {}

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

/// A read-only description of one operation in a graph.
///
/// Constants are borrowed from the graph, so inspecting a plan does not clone tensor data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpDescriptor<'a> {
    Input,
    Constant(&'a Tensor),
    /// A compact graph-level constant whose logical output is a dense tensor filled with `value`.
    /// Storage and liveness metadata continue to describe the full logical tensor extent.
    Uniform {
        value: f32,
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

/// Read-only metadata for one graph value, yielded in stable append order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeDescriptor<'a> {
    pub value: ValueId,
    pub op: OpDescriptor<'a>,
    pub shape: &'a [usize],
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
/// Applications must select this route deliberately; assessing another backend never causes an
/// automatic retry through the reference evaluator. Its temporary workspace is not yet bounded.
pub struct TensorReferenceAssessor;

impl TensorOperationAssessor for TensorReferenceAssessor {
    fn assess_node(&self, _graph: &Graph, _node: NodeDescriptor<'_>) -> TensorOperationSupport {
        TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Reference,
            workspace_bytes: None,
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
                if last_use[left_index] < right_index
                    || (selected_operation_is_read_only(left)
                        && selected_operation_is_read_only(right))
                {
                    continue;
                }
                let left_bytes = selected_operation_bytes(left)?;
                let right_bytes = selected_operation_bytes(right)?;
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
                if last_use[left_index] < right_index
                    || (is_read_only_descriptor(left.op) && is_read_only_descriptor(right.op))
                {
                    continue;
                }
                let left_bytes =
                    u64::try_from(node_output_bytes(left.shape).ok_or(TensorError::ShapeOverflow)?)
                        .map_err(|_| TensorError::ShapeOverflow)?;
                let right_bytes = u64::try_from(
                    node_output_bytes(right.shape).ok_or(TensorError::ShapeOverflow)?,
                )
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
                Op::Uniform(value) => (*value, rate_node.shape.as_slice(), true),
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

    /// Returns all pairwise storage disjointness requirements implied by live writable values.
    ///
    /// Lifetimes are inclusive, so values whose intervals touch at a node are both considered
    /// live there. Read-only input/constant pairs are omitted; all computed values require unique
    /// storage until an operation explicitly grants an in-place alias permission.
    ///
    /// # Errors
    ///
    /// Returns `ShapeOverflow` if a selected value's byte extent cannot be represented.
    pub fn storage_constraints(&self) -> Result<Vec<TensorStorageConstraint>, TensorError> {
        let mut constraints = Vec::new();
        for (i, left) in self.value_liveness.iter().enumerate() {
            for right in &self.value_liveness[i + 1..] {
                if left.last_live_node < right.first_live_node
                    || right.last_live_node < left.first_live_node
                    || (!self.is_writable_value(left.value) && !self.is_writable_value(right.value))
                {
                    continue;
                }
                let left_bytes =
                    u64::try_from(left.output_bytes.ok_or(TensorError::ShapeOverflow)?)
                        .map_err(|_| TensorError::ShapeOverflow)?;
                let right_bytes =
                    u64::try_from(right.output_bytes.ok_or(TensorError::ShapeOverflow)?)
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
        inputs: &[(ValueId, Tensor)],
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
        .filter(|node| matches!(node.op, OpDescriptor::Mul { .. }))
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

fn selected_operation_bytes(operation: &TensorSelectedOperation<'_>) -> Result<u64, TensorError> {
    let bytes =
        node_output_bytes(selected_operation_shape(operation)).ok_or(TensorError::ShapeOverflow)?;
    u64::try_from(bytes).map_err(|_| TensorError::ShapeOverflow)
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
    Constant(Tensor),
    Uniform(f32),
    Add(ValueId, ValueId),
    Sub(ValueId, ValueId),
    Mul(ValueId, ValueId),
    SgdUpdate(ValueId, ValueId, f32),
    MatMul(ValueId, ValueId, bool, bool),
    Relu(ValueId),
    ReluBackward(ValueId, ValueId),
    MeanSquaredError(ValueId, ValueId),
}

#[derive(Clone, Copy)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
}

fn operands(op: &Op) -> impl Iterator<Item = ValueId> + '_ {
    let (first, second) = match op {
        Op::Add(a, b)
        | Op::Sub(a, b)
        | Op::Mul(a, b)
        | Op::SgdUpdate(a, b, _)
        | Op::MeanSquaredError(a, b)
        | Op::MatMul(a, b, _, _) => (Some(*a), Some(*b)),
        Op::Relu(x) => (Some(*x), None),
        Op::ReluBackward(input, upstream) => (Some(*input), Some(*upstream)),
        Op::Input | Op::Constant(_) | Op::Uniform(_) => (None, None),
    };
    first.into_iter().chain(second)
}

#[derive(Clone, Debug)]
struct Node {
    op: Op,
    shape: Vec<usize>,
}

/// An immutable-in-practice, append-only graph builder. Value identifiers are graph-local.
#[derive(Debug)]
pub struct Graph {
    id: u64,
    nodes: Vec<Node>,
}

impl Default for Graph {
    fn default() -> Self {
        Self {
            id: next_graph_id(),
            nodes: Vec::new(),
        }
    }
}

impl Graph {
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
    /// remain live through the end of the plan. Unrelated graph nodes are omitted.
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
                output_bytes: node_output_bytes(&self.nodes[value.index].shape),
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
                output_bytes: node_output_bytes(node.shape),
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
                    output_bytes: node_output_bytes(node.shape),
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
            Op::Uniform(value) => OpDescriptor::Uniform { value: *value },
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
        }
    }

    fn push(&mut self, op: Op, shape: Vec<usize>) -> ValueId {
        let id = ValueId {
            graph_id: self.id,
            index: self.nodes.len(),
        };
        self.nodes.push(Node { op, shape });
        id
    }

    /// Adds an input value with the supplied shape.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::ShapeOverflow`] if the shape's element count overflows.
    pub fn input(&mut self, shape: impl Into<Vec<usize>>) -> Result<ValueId, TensorError> {
        let shape = shape.into();
        element_count(&shape)?;
        Ok(self.push(Op::Input, shape))
    }

    pub fn constant(&mut self, value: Tensor) -> ValueId {
        self.push(Op::Constant(value.clone()), value.shape)
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
    pub fn uniform(
        &mut self,
        shape: impl Into<Vec<usize>>,
        value: f32,
    ) -> Result<ValueId, TensorError> {
        let shape = shape.into();
        element_count(&shape)?;
        Ok(self.push(Op::Uniform(value), shape))
    }

    /// Adds two values with identical shapes.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID or
    /// [`TensorError::ShapeMismatch`] when the shapes differ.
    pub fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Add)
    }

    /// Subtracts two values with identical shapes, without broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID or
    /// [`TensorError::ShapeMismatch`] when the shapes differ.
    pub fn sub(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Sub)
    }

    /// Multiplies two values elementwise with identical shapes, without broadcasting.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID or
    /// [`TensorError::ShapeMismatch`] when the shapes differ.
    pub fn mul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.same_shape_binary(a, b, BinaryOp::Mul)
    }

    /// Adds an elementwise SGD update, `weights - learning_rate * gradient`.
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
        if !learning_rate.is_finite() {
            return Err(TensorError::InvalidLearningRate);
        }
        Ok(self.push(
            Op::SgdUpdate(weights, gradient, learning_rate),
            weights_shape,
        ))
    }

    fn same_shape_binary(
        &mut self,
        a: ValueId,
        b: ValueId,
        operation: BinaryOp,
    ) -> Result<ValueId, TensorError> {
        let sa = self.shape(a)?.to_vec();
        let sb = self.shape(b)?.to_vec();
        if sa != sb {
            return Err(TensorError::ShapeMismatch {
                left: sa,
                right: sb,
            });
        }
        let op = match operation {
            BinaryOp::Add => Op::Add(a, b),
            BinaryOp::Sub => Op::Sub(a, b),
            BinaryOp::Mul => Op::Mul(a, b),
        };
        Ok(self.push(op, self.nodes[a.index].shape.clone()))
    }

    /// Adds a rank-two matrix multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID,
    /// [`TensorError::MatMulShape`] for incompatible shapes, or
    /// [`TensorError::ShapeOverflow`] if the output element count overflows.
    pub fn matmul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        self.matmul_transposed(a, b, false, false)
    }

    /// Adds a rank-two matrix multiplication with optional logical transposes on its operands.
    /// Inputs remain in their original row-major shapes; no transpose tensor is materialized.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID,
    /// [`TensorError::MatMulShape`] for incompatible or non-matrix shapes, or
    /// [`TensorError::ShapeOverflow`] if the output element count overflows.
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
        Ok(self.push(Op::MatMul(a, b, transpose_left, transpose_right), shape))
    }

    /// Adds an elementwise `ReLU` operation.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] for an invalid value ID.
    pub fn relu(&mut self, x: ValueId) -> Result<ValueId, TensorError> {
        let shape = self.shape(x)?.to_vec();
        Ok(self.push(Op::Relu(x), shape))
    }

    /// Adds the `ReLU` derivative `input > 0 ? upstream : 0` elementwise.
    ///
    /// # Errors
    ///
    /// Returns a graph identity or shape error if either operand is invalid or their shapes differ.
    ///
    /// The strict comparison matches reverse-mode differentiation in `Execution::gradients`:
    /// zero and NaN inputs both produce zero gradients.
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
        Ok(self.push(Op::ReluBackward(input, upstream), input_shape))
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
        if element_count(&p)? == 0 {
            return Err(TensorError::ShapeMismatch { left: p, right: t });
        }
        Ok(self.push(Op::MeanSquaredError(prediction, target), vec![]))
    }

    /// Append a bounded reverse-mode graph for a scalar mean-squared-error root.
    ///
    /// The returned vector has one entry per node that existed before this call; each entry is
    /// the graph value holding that node's gradient, if the node contributes to `loss`. The
    /// resulting graph is ordinary tensor IR and can be assessed by a selected backend. This
    /// profile supports Add, Sub, Mul, `ReLU`, and transpose-aware `MatMul` in the loss dependency
    /// closure. Nested MSE is rejected before any node is appended.
    ///
    /// # Errors
    ///
    /// Returns an invalid-root or unsupported-gradient error without changing the graph, or a
    /// shape/extent error while constructing a gradient node.
    pub fn backward_mse(&mut self, loss: ValueId) -> Result<Vec<Option<ValueId>>, TensorError> {
        let original_len = self.nodes.len();
        let root = self
            .nodes
            .get(loss.index)
            .filter(|_| loss.graph_id == self.id)
            .ok_or(TensorError::UnknownValue(loss))?;
        if !matches!(root.op, Op::MeanSquaredError(..)) {
            return Err(TensorError::GradientRootNotMse(loss));
        }
        let mut visited = vec![false; original_len];
        let mut pending = vec![loss.index];
        while let Some(index) = pending.pop() {
            if visited[index] {
                continue;
            }
            visited[index] = true;
            let op = &self.nodes[index].op;
            if index != loss.index && matches!(op, Op::MeanSquaredError(..)) {
                return Err(TensorError::UnsupportedGradient(ValueId {
                    graph_id: self.id,
                    index,
                }));
            }
            pending.extend(operands(op).map(|id| id.index));
        }
        let mut gradients = vec![None; original_len];
        gradients[loss.index] = Some(self.constant(Tensor::scalar(1.0)));
        for index in (0..=loss.index).rev() {
            let Some(gradient) = gradients[index] else {
                continue;
            };
            let op = match &self.nodes[index].op {
                Op::Input | Op::Constant(_) | Op::Uniform(_) => continue,
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
                Op::MeanSquaredError(prediction, target) => OpDescriptor::MeanSquaredError {
                    prediction: *prediction,
                    target: *target,
                },
                Op::Relu(input) => OpDescriptor::Relu { input: *input },
                Op::ReluBackward(..) => {
                    unreachable!("generated backward nodes are not in the source graph")
                }
            };
            append_node_gradients(self, &mut gradients, op, gradient)?;
        }
        Ok(gradients)
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

    /// Evaluates the graph using the supplied input tensors.
    ///
    /// # Errors
    ///
    /// Returns an error for missing, duplicate, extra, or wrongly shaped inputs.
    pub fn evaluate(&self, inputs: &[(ValueId, Tensor)]) -> Result<Execution, TensorError> {
        let mut supplied: Vec<Option<&Tensor>> = vec![None; self.nodes.len()];
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
        }
        let mut values: Vec<Tensor> = Vec::with_capacity(self.nodes.len());
        for (index, node) in self.nodes.iter().enumerate() {
            let value = match &node.op {
                Op::Input => {
                    let id = ValueId {
                        graph_id: self.id,
                        index,
                    };
                    let value = supplied[id.index].ok_or(TensorError::MissingInput(id))?;
                    if value.shape != node.shape {
                        return Err(TensorError::ShapeMismatch {
                            left: value.shape.clone(),
                            right: node.shape.clone(),
                        });
                    }
                    value.clone()
                }
                Op::Constant(t) => t.clone(),
                Op::Uniform(value) => Tensor::splat(node.shape.clone(), *value)?,
                Op::Add(a, b) => binary(&values[a.index], &values[b.index], |x, y| x + y),
                Op::Sub(a, b) => binary(&values[a.index], &values[b.index], |x, y| x - y),
                Op::Mul(a, b) => binary(&values[a.index], &values[b.index], |x, y| x * y),
                Op::SgdUpdate(weights, gradient, learning_rate) => binary(
                    &values[weights.index],
                    &values[gradient.index],
                    |weight, grad| weight - learning_rate * grad,
                ),
                Op::MatMul(a, b, transpose_left, transpose_right) => matmul(
                    &values[a.index],
                    &values[b.index],
                    *transpose_left,
                    *transpose_right,
                ),
                Op::Relu(x) => Tensor::new(
                    node.shape.clone(),
                    values[x.index].data.iter().map(|v| v.max(0.0)).collect(),
                )?,
                Op::ReluBackward(input, upstream) => Tensor::new(
                    node.shape.clone(),
                    values[input.index]
                        .data
                        .iter()
                        .zip(&values[upstream.index].data)
                        .map(|(x, grad)| if *x > 0.0 { *grad } else { 0.0 })
                        .collect(),
                )?,
                Op::MeanSquaredError(a, b) => {
                    let n = mean_denominator(values[a.index].data.len());
                    Tensor::scalar(
                        values[a.index]
                            .data
                            .iter()
                            .zip(&values[b.index].data)
                            .map(|(x, y)| (x - y) * (x - y))
                            .sum::<f32>()
                            / n,
                    )
                }
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

fn append_node_gradients(
    graph: &mut Graph,
    gradients: &mut [Option<ValueId>],
    op: OpDescriptor<'_>,
    gradient: ValueId,
) -> Result<(), TensorError> {
    match op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {}
        OpDescriptor::Add { left, right } => {
            accumulate_gradient(graph, gradients, left, gradient)?;
            accumulate_gradient(graph, gradients, right, gradient)?;
        }
        OpDescriptor::Sub { left, right } => {
            accumulate_gradient(graph, gradients, left, gradient)?;
            let negative = graph.constant(filled_like(graph, right, -1.0)?);
            let right_gradient = graph.mul(gradient, negative)?;
            accumulate_gradient(graph, gradients, right, right_gradient)?;
        }
        OpDescriptor::Mul { left, right } => {
            let left_gradient = graph.mul(gradient, right)?;
            let right_gradient = graph.mul(gradient, left)?;
            accumulate_gradient(graph, gradients, left, left_gradient)?;
            accumulate_gradient(graph, gradients, right, right_gradient)?;
        }
        OpDescriptor::SgdUpdate {
            weights,
            gradient: update_gradient,
            learning_rate,
        } => {
            let scaled = graph.constant(Tensor::new(
                graph.shape(update_gradient)?.to_vec(),
                vec![-learning_rate; element_count(graph.shape(update_gradient)?)?],
            )?);
            let input_gradient = graph.mul(gradient, scaled)?;
            accumulate_gradient(graph, gradients, weights, gradient)?;
            accumulate_gradient(graph, gradients, update_gradient, input_gradient)?;
        }
        OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } => {
            append_matmul_gradients(
                graph,
                gradients,
                left,
                right,
                gradient,
                transpose_left,
                transpose_right,
            )?;
        }
        OpDescriptor::MeanSquaredError { prediction, target } => {
            append_mse_gradients(graph, gradients, prediction, target)?;
        }
        OpDescriptor::Relu { input } => {
            let input_gradient = graph.relu_backward(input, gradient)?;
            accumulate_gradient(graph, gradients, input, input_gradient)?;
        }
        OpDescriptor::ReluBackward { .. } => {
            unreachable!("generated backward nodes are not in the source graph")
        }
    }
    Ok(())
}

fn append_mse_gradients(
    graph: &mut Graph,
    gradients: &mut [Option<ValueId>],
    prediction: ValueId,
    target: ValueId,
) -> Result<(), TensorError> {
    let shape = graph.shape(prediction)?.to_vec();
    let count = element_count(&shape)?;
    let scale = graph.constant(Tensor::new(
        shape,
        vec![2.0 / mean_denominator(count); count],
    )?);
    let prediction_difference = graph.sub(prediction, target)?;
    let target_difference = graph.sub(target, prediction)?;
    let prediction_gradient = graph.mul(prediction_difference, scale)?;
    let target_gradient = graph.mul(target_difference, scale)?;
    accumulate_gradient(graph, gradients, prediction, prediction_gradient)?;
    accumulate_gradient(graph, gradients, target, target_gradient)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)] // MatMul differentiation needs both operands and flags.
fn append_matmul_gradients(
    graph: &mut Graph,
    gradients: &mut [Option<ValueId>],
    left: ValueId,
    right: ValueId,
    gradient: ValueId,
    transpose_left: bool,
    transpose_right: bool,
) -> Result<(), TensorError> {
    let left_gradient = if transpose_left {
        graph.matmul_transposed(right, gradient, transpose_right, true)?
    } else {
        graph.matmul_transposed(gradient, right, false, !transpose_right)?
    };
    let right_gradient = if transpose_right {
        graph.matmul_transposed(gradient, left, true, transpose_left)?
    } else {
        graph.matmul_transposed(left, gradient, !transpose_left, false)?
    };
    accumulate_gradient(graph, gradients, left, left_gradient)?;
    accumulate_gradient(graph, gradients, right, right_gradient)?;
    Ok(())
}

fn accumulate_gradient(
    graph: &mut Graph,
    gradients: &mut [Option<ValueId>],
    target: ValueId,
    addition: ValueId,
) -> Result<(), TensorError> {
    gradients[target.index] = Some(match gradients[target.index] {
        Some(existing) => graph.add(existing, addition)?,
        None => addition,
    });
    Ok(())
}

fn filled_like(graph: &Graph, value: ValueId, scalar: f32) -> Result<Tensor, TensorError> {
    let shape = graph.shape(value)?.to_vec();
    let count = element_count(&shape)?;
    Tensor::new(shape, vec![scalar; count])
}

fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
    Tensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(x, y)| f(*x, *y)).collect(),
        known_uniform_value: None,
    }
}

// Keep the CPU reference's multiply and accumulation steps explicit; fused rounding is a
// backend numerical choice and would change this oracle's results.
#[allow(clippy::suboptimal_flops)]
fn matmul(left: &Tensor, right: &Tensor, transpose_left: bool, transpose_right: bool) -> Tensor {
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
    let mut output = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            for inner_index in 0..inner {
                let left_index = if transpose_left {
                    inner_index * left.shape[1] + row
                } else {
                    row * left.shape[1] + inner_index
                };
                let right_index = if transpose_right {
                    column * right.shape[1] + inner_index
                } else {
                    inner_index * right.shape[1] + column
                };
                output[row * columns + column] += left.data[left_index] * right.data[right_index];
            }
        }
    }
    Tensor {
        shape: vec![rows, columns],
        data: output,
        known_uniform_value: None,
    }
}

// Tensor storage must already fit in memory; f32 precision is sufficient for a
// practically allocatable element count used as a mean divisor.
#[allow(clippy::cast_precision_loss)]
const fn mean_denominator(element_count: usize) -> f32 {
    element_count as f32
}

#[derive(Clone, Debug)]
pub struct Execution {
    graph_id: u64,
    node_count: usize,
    values: Vec<Tensor>,
}
impl Execution {
    /// Returns the evaluated tensor for a graph value.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::UnknownValue`] if `id` does not belong to this execution.
    pub fn value(&self, id: ValueId) -> Result<&Tensor, TensorError> {
        self.values
            .get(id.index)
            .filter(|_| id.graph_id == self.graph_id)
            .ok_or(TensorError::UnknownValue(id))
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
        let out = self.value(output)?;
        if !out.shape.is_empty() {
            return Err(TensorError::LossMustBeScalar(out.shape.clone()));
        }
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
                    let left = &self.values[a.index];
                    let right = &self.values[b.index];
                    let (left_gradient, right_gradient) =
                        elementwise_mul_gradients(left, right, &grad)?;
                    accumulate(&mut grads, a, left_gradient);
                    accumulate(&mut grads, b, right_gradient);
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
                        self.values[x.index]
                            .data
                            .iter()
                            .zip(grad.data)
                            .map(|(v, d)| if *v > 0.0 { d } else { 0.0 })
                            .collect(),
                    )?;
                    accumulate(&mut grads, x, g);
                }
                Op::ReluBackward(..) => unreachable!(
                    "backward operation has no second derivative in reference gradients"
                ),
                Op::MeanSquaredError(a, b) => {
                    let divisor = mean_denominator(self.values[a.index].data.len());
                    let prediction = &self.values[a.index];
                    let target = &self.values[b.index];
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
                    let left = &self.values[a.index];
                    let right = &self.values[b.index];
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
