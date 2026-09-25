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

static NEXT_GRAPH_ID: AtomicU64 = AtomicU64::new(1);

fn next_graph_id() -> u64 {
    NEXT_GRAPH_ID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
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
        Ok(Self { shape, data })
    }

    #[must_use]
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
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

/// Backend-neutral storage facts for graph values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorGraphRequirements<'a> {
    /// Dense row-major f32 values are the only layout and element type currently represented.
    pub values: Vec<TensorValueRequirement<'a>>,
    /// Sum of all graph-value output storage, not a peak/live memory estimate.
    pub total_value_bytes: Option<usize>,
}

/// Inclusive lifetime and output-storage facts for one value in an execution plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueLiveness {
    pub value: ValueId,
    pub output_bytes: Option<usize>,
    /// Index in `TensorExecutionPlan::node_order` where this value is produced.
    pub first_live_node: usize,
    /// Last node that reads this value, or its production node when it has no consumers.
    pub last_live_node: usize,
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
    peak_live_bytes: Option<usize>,
}

impl TensorExecutionPlan<'_> {
    #[must_use]
    pub fn node_order(&self) -> &[ValueId] {
        &self.node_order
    }

    #[must_use]
    pub fn input_values(&self) -> &[ValueId] {
        &self.input_values
    }

    #[must_use]
    pub fn output_values(&self) -> &[ValueId] {
        &self.output_values
    }

    #[must_use]
    pub fn value_liveness(&self) -> &[TensorValueLiveness] {
        &self.value_liveness
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValueRequirement<'a> {
    pub value: ValueId,
    pub shape: &'a [usize],
    pub output_bytes: Option<usize>,
}

fn node_output_bytes(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(std::mem::size_of::<f32>(), |bytes, &dimension| {
            bytes.checked_mul(dimension)
        })
}

#[derive(Clone, Debug)]
enum Op {
    Input,
    Constant(Tensor),
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
        Op::Input | Op::Constant(_) => (None, None),
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
    #[must_use]
    pub fn execution_plan(&self) -> TensorExecutionPlan<'_> {
        let node_order: Vec<_> = self.nodes().map(|node| node.value).collect();
        let mut last_use: Vec<usize> = (0..self.nodes.len()).collect();
        let mut has_consumer = vec![false; self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            for operand in operands(&node.op) {
                last_use[operand.index] = index;
                has_consumer[operand.index] = true;
            }
        }
        let input_values = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, node)| {
                matches!(node.op, Op::Input).then_some(ValueId {
                    graph_id: self.id,
                    index,
                })
            })
            .collect();
        let output_values = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, _)| {
                (!has_consumer[index]).then_some(ValueId {
                    graph_id: self.id,
                    index,
                })
            })
            .collect();
        let value_liveness: Vec<_> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| TensorValueLiveness {
                value: ValueId {
                    graph_id: self.id,
                    index,
                },
                output_bytes: node_output_bytes(&node.shape),
                first_live_node: index,
                last_live_node: last_use[index],
            })
            .collect();
        let mut ending_bytes = vec![0usize; self.nodes.len()];
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
        TensorExecutionPlan {
            graph: self,
            node_order,
            input_values,
            output_values,
            value_liveness,
            peak_live_bytes,
        }
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
        self.nodes.iter().enumerate().map(|(index, node)| {
            let value = ValueId {
                graph_id: self.id,
                index,
            };
            let op = match &node.op {
                Op::Input => OpDescriptor::Input,
                Op::Constant(tensor) => OpDescriptor::Constant(tensor),
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
        })
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
                Op::Input | Op::Constant(_) => continue,
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
        OpDescriptor::Input | OpDescriptor::Constant(_) => {}
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
                Op::Input | Op::Constant(_) => {}
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
mod tests {
    use super::*;

    fn assert_gradient_graph_matches_reference(
        graph: &mut Graph,
        inputs: &[(ValueId, Tensor)],
        loss: ValueId,
        values: &[ValueId],
    ) {
        let gradient_values = graph.backward_mse(loss).unwrap();
        let execution = graph.evaluate(inputs).unwrap();
        let reference = execution.gradients(graph, loss).unwrap();
        for value in values {
            let graph_gradient = execution
                .value(gradient_values[value.index].unwrap())
                .unwrap();
            let reference_gradient = reference[value.index].as_ref().unwrap();
            assert_eq!(graph_gradient.shape(), reference_gradient.shape());
            for (actual, expected) in graph_gradient.data().iter().zip(reference_gradient.data()) {
                assert!(
                    (actual - expected).abs() <= 1.0e-5,
                    "{actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn backward_mse_graph_matches_reference_with_fanout() {
        let mut graph = Graph::default();
        let samples = graph.input([2, 3]).unwrap();
        let weights = graph.input([3, 2]).unwrap();
        let prediction = graph.matmul(samples, weights).unwrap();
        let squared = graph.mul(prediction, prediction).unwrap();
        let target = graph.input([2, 2]).unwrap();
        let loss = graph.mean_squared_error(squared, target).unwrap();
        let inputs = [
            (
                samples,
                Tensor::new([2, 3], vec![1.0, 2.0, -1.0, 0.5, -2.0, 3.0]).unwrap(),
            ),
            (
                weights,
                Tensor::new([3, 2], vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.2]).unwrap(),
            ),
            (
                target,
                Tensor::new([2, 2], vec![0.0, 1.0, -1.0, 0.5]).unwrap(),
            ),
        ];
        assert_gradient_graph_matches_reference(
            &mut graph,
            &inputs,
            loss,
            &[samples, weights, prediction],
        );
    }

    #[test]
    fn backward_mse_graph_matches_reference_for_transposed_matmul() {
        let mut graph = Graph::default();
        let left = graph.input([3, 2]).unwrap();
        let right = graph.input([4, 3]).unwrap();
        let prediction = graph.matmul_transposed(left, right, true, true).unwrap();
        let target = graph.input([2, 4]).unwrap();
        let loss = graph.mean_squared_error(prediction, target).unwrap();
        let inputs = [
            (
                left,
                Tensor::new([3, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5]).unwrap(),
            ),
            (
                right,
                Tensor::new(
                    [4, 3],
                    vec![
                        0.2, -0.1, 0.3, 0.4, 0.5, -0.2, 0.1, -0.3, 0.6, 0.7, 0.2, 0.8,
                    ],
                )
                .unwrap(),
            ),
            (target, Tensor::new([2, 4], vec![0.0; 8]).unwrap()),
        ];
        assert_gradient_graph_matches_reference(&mut graph, &inputs, loss, &[left, right]);
    }

    #[test]
    fn relu_backward_checks_shapes_and_matches_strict_derivative() {
        let mut graph = Graph::default();
        let input = graph.input([2]).unwrap();
        let upstream = graph.input([2]).unwrap();
        let wrong_shape = graph.constant(Tensor::new([1, 2], vec![0.0, 0.0]).unwrap());
        assert_eq!(
            graph.relu_backward(input, wrong_shape),
            Err(TensorError::ShapeMismatch {
                left: vec![2],
                right: vec![1, 2]
            })
        );
        let derivative = graph.relu_backward(input, upstream).unwrap();
        let inputs = [
            (input, Tensor::new([2], vec![f32::NAN, 0.0]).unwrap()),
            (upstream, Tensor::new([2], vec![3.0, 4.0]).unwrap()),
        ];
        assert_eq!(
            graph
                .evaluate(&inputs)
                .unwrap()
                .value(derivative)
                .unwrap()
                .data(),
            &[0.0, 0.0]
        );
    }

    #[test]
    fn backward_mse_graph_matches_reference_for_two_hidden_relu_layers() {
        let mut graph = Graph::default();
        let samples = graph.input([2, 3]).unwrap();
        let weights1 = graph.input([3, 4]).unwrap();
        let hidden1 = graph.matmul(samples, weights1).unwrap();
        let activated1 = graph.relu(hidden1).unwrap();
        let weights2 = graph.input([4, 4]).unwrap();
        let hidden2 = graph.matmul(activated1, weights2).unwrap();
        let activated2 = graph.relu(hidden2).unwrap();
        let weights3 = graph.input([4, 2]).unwrap();
        let prediction = graph.matmul(activated2, weights3).unwrap();
        let target = graph.input([2, 2]).unwrap();
        let loss = graph.mean_squared_error(prediction, target).unwrap();
        let inputs = [
            (
                samples,
                Tensor::new([2, 3], vec![1.0, -2.0, 0.5, -1.0, 0.25, 2.0]).unwrap(),
            ),
            (
                weights1,
                Tensor::new(
                    [3, 4],
                    vec![
                        0.2, -0.3, 0.5, 0.1, -0.4, 0.6, 0.2, -0.1, 0.3, 0.2, -0.5, 0.4,
                    ],
                )
                .unwrap(),
            ),
            (
                weights2,
                Tensor::new(
                    [4, 4],
                    vec![
                        0.1, 0.2, -0.3, 0.4, -0.2, 0.5, 0.1, -0.1, 0.3, -0.4, 0.2, 0.6, 0.5, 0.1,
                        -0.2, 0.3,
                    ],
                )
                .unwrap(),
            ),
            (
                weights3,
                Tensor::new([4, 2], vec![0.2, -0.1, 0.4, 0.3, -0.5, 0.2, 0.1, 0.6]).unwrap(),
            ),
            (
                target,
                Tensor::new([2, 2], vec![0.0, 1.0, -0.5, 0.25]).unwrap(),
            ),
        ];
        assert_gradient_graph_matches_reference(
            &mut graph,
            &inputs,
            loss,
            &[
                samples, weights1, hidden1, activated1, weights2, hidden2, activated2, weights3,
                prediction,
            ],
        );
    }

    struct ReferenceAssessor;

    #[test]
    fn elementwise_algebra_checks_shapes_and_evaluates_without_broadcasting() {
        let mut graph = Graph::default();
        let left = graph.input(vec![2, 2]).unwrap();
        let right = graph.input(vec![2, 2]).unwrap();
        let sub = graph.sub(left, right).unwrap();
        let mul = graph.mul(sub, right).unwrap();
        let shape_mismatch = graph.input(vec![2]).unwrap();
        assert!(matches!(
            graph.mul(left, shape_mismatch),
            Err(TensorError::ShapeMismatch { .. })
        ));
        let execution = graph
            .evaluate(&[
                (
                    left,
                    Tensor::new(vec![2, 2], vec![3.0, 5.0, 7.0, 9.0]).unwrap(),
                ),
                (
                    right,
                    Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ),
                (
                    shape_mismatch,
                    Tensor::new(vec![2], vec![0.0, 0.0]).unwrap(),
                ),
            ])
            .unwrap();
        assert_eq!(execution.value(sub).unwrap().data(), &[2.0, 3.0, 4.0, 5.0]);
        assert_eq!(
            execution.value(mul).unwrap().data(),
            &[2.0, 6.0, 12.0, 20.0]
        );
        assert!(
            matches!(graph.nodes().nth(2).unwrap().op, OpDescriptor::Sub { left: a, right: b } if a == left && b == right)
        );
        assert!(
            matches!(graph.nodes().nth(3).unwrap().op, OpDescriptor::Mul { left: a, right: b } if a == sub && b == right)
        );
        let plan = graph.execution_plan();
        assert_eq!(plan.node_order().len(), graph.nodes().len());
        assert_eq!(plan.output_values(), &[mul, shape_mismatch]);
    }

    #[test]
    fn sgd_update_checks_inputs_and_evaluates_explicit_operation() {
        let mut graph = Graph::default();
        let weights = graph.input([3]).unwrap();
        let gradient = graph.input([3]).unwrap();
        let updated = graph.sgd_update(weights, gradient, 0.25).unwrap();
        let wrong_shape = graph.input([2]).unwrap();
        assert!(matches!(
            graph.sgd_update(weights, wrong_shape, 0.25),
            Err(TensorError::ShapeMismatch { .. })
        ));
        assert!(matches!(
            graph.sgd_update(weights, gradient, f32::NAN),
            Err(TensorError::InvalidLearningRate)
        ));
        assert!(matches!(
            graph.sgd_update(weights, gradient, f32::INFINITY),
            Err(TensorError::InvalidLearningRate)
        ));

        let execution = graph
            .evaluate(&[
                (weights, Tensor::new([3], vec![1.0, 2.0, -3.0]).unwrap()),
                (gradient, Tensor::new([3], vec![0.4, -2.0, 8.0]).unwrap()),
                (wrong_shape, Tensor::new([2], vec![0.0, 0.0]).unwrap()),
            ])
            .unwrap();
        for (actual, expected) in execution
            .value(updated)
            .unwrap()
            .data()
            .iter()
            .zip([0.9, 2.5, -5.0])
        {
            assert!((actual - expected).abs() < 1.0e-6);
        }
        assert!(matches!(
            graph.nodes().nth(2).unwrap().op,
            OpDescriptor::SgdUpdate {
                weights: value_weights,
                gradient: value_gradient,
                learning_rate
            } if value_weights == weights
                && value_gradient == gradient
                && (learning_rate - 0.25).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn transposed_matmul_evaluates_and_differentiates_in_original_layout() {
        let mut graph = Graph::default();
        let left = graph.input(vec![3, 2]).unwrap();
        let right = graph.input(vec![3, 4]).unwrap();
        let product = graph.matmul_transposed(left, right, true, false).unwrap();
        assert_eq!(graph.shape(product).unwrap(), &[2, 4]);
        assert!(matches!(
            graph.matmul_transposed(left, right, false, false),
            Err(TensorError::MatMulShape { .. })
        ));
        let execution = graph
            .evaluate(&[
                (
                    left,
                    Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ),
                (
                    right,
                    Tensor::new(
                        vec![3, 4],
                        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    )
                    .unwrap(),
                ),
            ])
            .unwrap();
        assert_eq!(
            execution.value(product).unwrap().data(),
            &[1.0, 3.0, 5.0, 5.0, 2.0, 4.0, 6.0, 6.0]
        );
        let mut scalar_graph = Graph::default();
        let x = scalar_graph.input(vec![3, 2]).unwrap();
        let y = scalar_graph.input(vec![3, 1]).unwrap();
        let transposed = scalar_graph.matmul_transposed(x, y, true, false).unwrap();
        let target = scalar_graph.constant(Tensor::new(vec![2, 1], vec![0.0, 0.0]).unwrap());
        let loss = scalar_graph.mean_squared_error(transposed, target).unwrap();
        let execution = scalar_graph
            .evaluate(&[
                (
                    x,
                    Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ),
                (y, Tensor::new(vec![3, 1], vec![1.0, 1.0, 1.0]).unwrap()),
            ])
            .unwrap();
        let gradients = execution.gradients(&scalar_graph, loss).unwrap();
        assert_eq!(
            gradients[x.index].as_ref().unwrap().data(),
            &[9.0, 12.0, 9.0, 12.0, 9.0, 12.0]
        );
        assert_eq!(
            gradients[y.index].as_ref().unwrap().data(),
            &[33.0, 75.0, 117.0]
        );
    }

    impl TensorOperationAssessor for ReferenceAssessor {
        fn assess_node(&self, _graph: &Graph, _node: NodeDescriptor<'_>) -> TensorOperationSupport {
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Reference,
                workspace_bytes: None,
            }
        }
    }

    #[test]
    fn selected_assessment_and_storage_requirements_preserve_node_order() {
        let mut graph = Graph::default();
        let input = graph.input(vec![2, 2]).unwrap();
        let constant = graph.constant(Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap());
        let sum = graph.add(input, constant).unwrap();
        let output = graph.relu(sum).unwrap();

        let assessment = graph.assess_with(&ReferenceAssessor);
        assert_eq!(assessment.nodes.len(), 4);
        assert_eq!(assessment.nodes[0].node.value, input);
        assert_eq!(assessment.nodes[1].node.value, constant);
        assert_eq!(assessment.nodes[2].node.value, sum);
        assert_eq!(assessment.nodes[3].node.value, output);
        assert!(assessment.nodes.iter().all(|node| {
            matches!(
                node.support,
                TensorOperationSupport::Supported {
                    route: TensorExecutionRoute::Reference,
                    workspace_bytes: None,
                }
            ) && node.output_bytes == Some(16)
        }));
        assert!(assessment.is_supported());

        let requirements = graph.requirements();
        assert_eq!(requirements.values.len(), 4);
        assert_eq!(requirements.total_value_bytes, Some(64));
        assert!(
            requirements
                .values
                .iter()
                .all(|value| value.output_bytes == Some(16))
        );
    }

    #[test]
    fn execution_plan_reports_stable_order_outputs_and_fanout_liveness() {
        let mut graph = Graph::default();
        let input = graph.input(vec![2]).unwrap();
        let constant = graph.constant(Tensor::new(vec![2], vec![1.0, 1.0]).unwrap());
        let first = graph.add(input, constant).unwrap();
        let second = graph.relu(first).unwrap();
        let output = graph.add(first, second).unwrap();

        let plan = graph.execution_plan();
        assert_eq!(plan.node_order(), &[input, constant, first, second, output]);
        assert_eq!(plan.input_values(), &[input]);
        assert_eq!(plan.output_values(), &[output]);
        assert_eq!(plan.value_liveness()[2].first_live_node, 2);
        assert_eq!(plan.value_liveness()[2].last_live_node, 4);
        assert_eq!(plan.peak_live_bytes(), Some(24));

        let inputs = [(input, Tensor::new(vec![2], vec![-1.0, 2.0]).unwrap())];
        let planned = plan.execute_reference(&inputs).unwrap();
        let ordinary = graph.evaluate(&inputs).unwrap();
        assert_eq!(planned.value(output).unwrap().data(), &[0.0, 6.0]);
        assert_eq!(
            planned.value(output).unwrap(),
            ordinary.value(output).unwrap()
        );
    }

    #[test]
    fn unsupported_assessment_has_no_implicit_fallback() {
        struct RejectAll;
        impl TensorOperationAssessor for RejectAll {
            fn assess_node(
                &self,
                _graph: &Graph,
                _node: NodeDescriptor<'_>,
            ) -> TensorOperationSupport {
                TensorOperationSupport::Unsupported {
                    reason: TensorUnsupportedReason::Operation,
                }
            }
        }

        let mut graph = Graph::default();
        graph.input(vec![1]).unwrap();
        let assessment = graph.assess_with(&RejectAll);
        assert!(!assessment.is_supported());
        assert_eq!(assessment.unsupported_nodes().count(), 1);
        assert!(matches!(
            assessment.nodes[0].support,
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation
            }
        ));
    }

    fn build() -> (Graph, ValueId, ValueId, ValueId, ValueId) {
        let mut g = Graph::default();
        let x = g.input(vec![2, 2]).unwrap();
        let w = g.input(vec![2, 1]).unwrap();
        let b = g.input(vec![2, 1]).unwrap();
        let affine = g.matmul(x, w).unwrap();
        let prediction = g.add(affine, b).unwrap();
        let target = g.constant(Tensor::new(vec![2, 1], vec![1.0, -1.0]).unwrap());
        let loss = g.mean_squared_error(prediction, target).unwrap();
        (g, x, w, b, loss)
    }

    #[test]
    fn linear_regression_value_and_finite_difference_gradients() {
        let (g, x, w, b, loss) = build();
        let xv = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let wv = Tensor::new(vec![2, 1], vec![0.25, -0.5]).unwrap();
        let bv = Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap();
        let inputs = [(x, xv.clone()), (w, wv.clone()), (b, bv.clone())];
        let execution = g.evaluate(&inputs).unwrap();
        let analytic = execution.gradients(&g, loss).unwrap()[w.index]
            .as_ref()
            .unwrap()
            .data
            .clone();
        let eps = 1e-3;
        for (index, analytic_gradient) in analytic.iter().enumerate() {
            let mut plus = wv.clone();
            plus.data[index] += eps;
            let mut minus = wv.clone();
            minus.data[index] -= eps;
            let eval_loss = |candidate_w: Tensor| {
                let eval = g
                    .evaluate(&[(x, xv.clone()), (w, candidate_w), (b, bv.clone())])
                    .unwrap();
                eval.value(loss).unwrap().data[0]
            };
            let numeric = (eval_loss(plus) - eval_loss(minus)) / (2.0 * eps);
            assert!(
                (analytic_gradient - numeric).abs() < 1e-3,
                "{index}: {analytic_gradient} != {numeric}"
            );
        }
        assert_eq!(execution.value(loss).unwrap().shape(), &[]);

        let gradients = execution.gradients(&g, loss).unwrap();
        let learning_rate = 0.05;
        let updated_w = Tensor::new(
            wv.shape.clone(),
            wv.data
                .iter()
                .zip(&gradients[w.index].as_ref().unwrap().data)
                .map(|(value, grad)| value - learning_rate * grad)
                .collect(),
        )
        .unwrap();
        let updated_b = Tensor::new(
            bv.shape.clone(),
            bv.data
                .iter()
                .zip(&gradients[b.index].as_ref().unwrap().data)
                .map(|(value, grad)| value - learning_rate * grad)
                .collect(),
        )
        .unwrap();
        let updated = g
            .evaluate(&[(x, xv), (w, updated_w), (b, updated_b)])
            .unwrap();
        assert!(updated.value(loss).unwrap().data[0] < execution.value(loss).unwrap().data[0]);
    }

    #[test]
    fn checked_shape_and_operator_shapes() {
        assert_eq!(
            Tensor::new(vec![usize::MAX, 2], vec![]),
            Err(TensorError::ShapeOverflow)
        );
        let mut g = Graph::default();
        let a = g.input(vec![2, 3]).unwrap();
        let b = g.input(vec![4, 2]).unwrap();
        assert!(matches!(
            g.matmul(a, b),
            Err(TensorError::MatMulShape { .. })
        ));
    }

    #[test]
    fn graph_plan_exposes_borrowed_operations_shapes_and_append_order() {
        let mut graph = Graph::default();
        let input = graph.input(vec![2, 2]).unwrap();
        let weights = graph.input(vec![2, 1]).unwrap();
        let product = graph.matmul(input, weights).unwrap();
        let activated = graph.relu(product).unwrap();
        let bias = graph.constant(Tensor::new(vec![2, 1], vec![0.5, -0.5]).unwrap());
        let prediction = graph.add(activated, bias).unwrap();
        let target = graph.constant(Tensor::new(vec![2, 1], vec![1.0, 0.0]).unwrap());
        let loss = graph.mean_squared_error(prediction, target).unwrap();

        let plan: Vec<_> = graph.nodes().collect();
        assert_eq!(plan.len(), 8);
        assert_eq!(
            plan.iter().map(|node| node.value).collect::<Vec<_>>(),
            [
                input, weights, product, activated, bias, prediction, target, loss
            ]
        );
        assert_eq!(plan[0].op, OpDescriptor::Input);
        assert_eq!(plan[0].shape, [2, 2]);
        assert_eq!(plan[1].shape, [2, 1]);
        assert_eq!(
            plan[2].op,
            OpDescriptor::MatMul {
                left: input,
                right: weights,
                transpose_left: false,
                transpose_right: false,
            }
        );
        assert_eq!(plan[2].shape, [2, 1]);
        assert_eq!(plan[3].op, OpDescriptor::Relu { input: product });
        assert_eq!(
            plan[4].op,
            OpDescriptor::Constant(&Tensor::new(vec![2, 1], vec![0.5, -0.5]).unwrap())
        );
        assert_eq!(
            plan[5].op,
            OpDescriptor::Add {
                left: activated,
                right: bias
            }
        );
        assert_eq!(plan[6].shape, [2, 1]);
        assert_eq!(
            plan[7].op,
            OpDescriptor::MeanSquaredError { prediction, target }
        );
        assert_eq!(plan[7].shape, []);
    }

    #[test]
    fn graph_plan_value_ids_keep_graph_identity() {
        let mut first = Graph::default();
        let first_id = first.input(vec![1]).unwrap();
        let mut second = Graph::default();
        let second_id = second.input(vec![1]).unwrap();

        assert_ne!(first_id, second_id);
        assert_eq!(first.nodes().next().unwrap().value, first_id);
        assert_eq!(second.nodes().next().unwrap().value, second_id);
        assert_eq!(
            second.shape(first_id),
            Err(TensorError::UnknownValue(first_id))
        );
    }

    #[test]
    fn reference_route_requires_explicit_assessor() {
        let (graph, ..) = build();
        let assessment = graph.assess_with(&TensorReferenceAssessor);
        assert!(assessment.is_supported());
        assert_eq!(assessment.nodes.len(), graph.nodes().len());
        assert!(assessment.nodes.iter().all(|node| matches!(
            node.support,
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Reference,
                workspace_bytes: None
            }
        )));
    }

    #[test]
    fn composed_loss_propagates_incoming_gradient() {
        let (mut g, x, w, b, loss) = build();
        let doubled_loss = g.add(loss, loss).unwrap();
        let inputs = [
            (
                x,
                Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ),
            (w, Tensor::new(vec![2, 1], vec![0.25, -0.5]).unwrap()),
            (b, Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap()),
        ];
        let execution = g.evaluate(&inputs).unwrap();
        let gradients = execution.gradients(&g, doubled_loss).unwrap();
        let base_gradient = execution.gradients(&g, loss).unwrap();
        for (actual, base) in gradients[w.index]
            .as_ref()
            .unwrap()
            .data
            .iter()
            .zip(&base_gradient[w.index].as_ref().unwrap().data)
        {
            assert!((actual - 2.0 * base).abs() < 1e-6);
        }
    }

    #[test]
    fn execution_and_input_bindings_are_graph_checked() {
        let (mut graph, x, w, b, loss) = build();
        let inputs = [
            (
                x,
                Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ),
            (w, Tensor::new(vec![2, 1], vec![0.25, -0.5]).unwrap()),
            (b, Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap()),
        ];

        let duplicated = [
            inputs[0].clone(),
            inputs[0].clone(),
            inputs[1].clone(),
            inputs[2].clone(),
        ];
        assert_eq!(
            graph.evaluate(&duplicated).unwrap_err(),
            TensorError::DuplicateInput(x)
        );
        let extra = [(loss, Tensor::scalar(0.0))];
        assert_eq!(
            graph
                .evaluate(&[
                    inputs[0].clone(),
                    inputs[1].clone(),
                    inputs[2].clone(),
                    extra[0].clone()
                ])
                .unwrap_err(),
            TensorError::ExtraInput(loss)
        );

        let execution = graph.evaluate(&inputs).unwrap();
        let mut other_graph = Graph::default();
        let foreign_id = other_graph.input(vec![2, 2]).unwrap();
        assert_eq!(
            other_graph.shape(x).unwrap_err(),
            TensorError::UnknownValue(x)
        );
        assert_eq!(
            other_graph.add(foreign_id, x).unwrap_err(),
            TensorError::UnknownValue(x)
        );
        let other_graph = Graph::default();
        assert_eq!(
            execution.gradients(&other_graph, loss).unwrap_err(),
            TensorError::WrongGraph
        );
        let new_scalar = graph.constant(Tensor::scalar(1.0));
        let _ = graph.add(loss, new_scalar).unwrap();
        assert_eq!(
            execution.gradients(&graph, loss).unwrap_err(),
            TensorError::GraphChanged
        );
    }
}
