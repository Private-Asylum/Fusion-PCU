//! Small standalone tensor dialect with a deterministic CPU reference evaluator.
//!
//! This crate is intentionally outside `fusion-pcu`: tensor semantics belong to a dialect, not
//! to the generic coprocessor IR. The current graph supports f32 inputs/constants, add, matrix
//! multiplication, ReLU, and mean-squared error. It does not implement broadcasting, batching,
//! convolution, views, mixed precision, optimizers, serialization, or device execution.

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

    pub fn scalar(value: f32) -> Self {
        Self {
            shape: vec![],
            data: vec![value],
        }
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn data(&self) -> &[f32] {
        &self.data
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

#[derive(Clone, Debug)]
enum Op {
    Input,
    Constant(Tensor),
    Add(ValueId, ValueId),
    MatMul(ValueId, ValueId),
    Relu(ValueId),
    MeanSquaredError(ValueId, ValueId),
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
    fn push(&mut self, op: Op, shape: Vec<usize>) -> ValueId {
        let id = ValueId {
            graph_id: self.id,
            index: self.nodes.len(),
        };
        self.nodes.push(Node { op, shape });
        id
    }

    pub fn input(&mut self, shape: impl Into<Vec<usize>>) -> Result<ValueId, TensorError> {
        let shape = shape.into();
        element_count(&shape)?;
        Ok(self.push(Op::Input, shape))
    }

    pub fn constant(&mut self, value: Tensor) -> ValueId {
        self.push(Op::Constant(value.clone()), value.shape)
    }

    pub fn add(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        let sa = self.shape(a)?.to_vec();
        let sb = self.shape(b)?.to_vec();
        if sa != sb {
            return Err(TensorError::ShapeMismatch {
                left: sa,
                right: sb,
            });
        }
        Ok(self.push(Op::Add(a, b), self.nodes[a.index].shape.clone()))
    }

    pub fn matmul(&mut self, a: ValueId, b: ValueId) -> Result<ValueId, TensorError> {
        let sa = self.shape(a)?.to_vec();
        let sb = self.shape(b)?.to_vec();
        if sa.len() != 2 || sb.len() != 2 || sa[1] != sb[0] {
            return Err(TensorError::MatMulShape {
                left: sa,
                right: sb,
            });
        }
        let shape = vec![sa[0], sb[1]];
        element_count(&shape)?;
        Ok(self.push(Op::MatMul(a, b), shape))
    }

    pub fn relu(&mut self, x: ValueId) -> Result<ValueId, TensorError> {
        let shape = self.shape(x)?.to_vec();
        Ok(self.push(Op::Relu(x), shape))
    }

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

    pub fn shape(&self, id: ValueId) -> Result<&[usize], TensorError> {
        self.nodes
            .get(id.index)
            .filter(|_| id.graph_id == self.id)
            .map(|n| n.shape.as_slice())
            .ok_or(TensorError::UnknownValue(id))
    }

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
                Op::MatMul(a, b) => matmul(&values[a.index], &values[b.index]),
                Op::Relu(x) => Tensor::new(
                    node.shape.clone(),
                    values[x.index].data.iter().map(|v| v.max(0.0)).collect(),
                )?,
                Op::MeanSquaredError(a, b) => {
                    let n = values[a.index].data.len() as f32;
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

fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
    Tensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(x, y)| f(*x, *y)).collect(),
    }
}

fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                out[i * n + j] += a.data[i * k + p] * b.data[p * n + j];
            }
        }
    }
    Tensor {
        shape: vec![m, n],
        data: out,
    }
}

#[derive(Clone, Debug)]
pub struct Execution {
    graph_id: u64,
    node_count: usize,
    values: Vec<Tensor>,
}
impl Execution {
    pub fn value(&self, id: ValueId) -> Result<&Tensor, TensorError> {
        self.values
            .get(id.index)
            .filter(|_| id.graph_id == self.graph_id)
            .ok_or(TensorError::UnknownValue(id))
    }

    /// Reverse-mode derivative of the selected scalar output with respect to every graph value.
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
                Op::MeanSquaredError(a, b) => {
                    let n = self.values[a.index].data.len() as f32;
                    let p = &self.values[a.index];
                    let t = &self.values[b.index];
                    let gp = Tensor::new(
                        p.shape.clone(),
                        p.data
                            .iter()
                            .zip(&t.data)
                            .map(|(x, y)| 2.0 * (x - y) / n * grad.data[0])
                            .collect(),
                    )?;
                    let gt = Tensor::new(
                        t.shape.clone(),
                        p.data
                            .iter()
                            .zip(&t.data)
                            .map(|(x, y)| 2.0 * (y - x) / n * grad.data[0])
                            .collect(),
                    )?;
                    accumulate(&mut grads, a, gp);
                    accumulate(&mut grads, b, gt);
                }
                Op::MatMul(a, b) => {
                    let left = &self.values[a.index];
                    let right = &self.values[b.index];
                    let (m, k, n) = (left.shape[0], left.shape[1], right.shape[1]);
                    let mut gl = vec![0.0; m * k];
                    let mut gr = vec![0.0; k * n];
                    for i in 0..m {
                        for j in 0..n {
                            let d = grad.data[i * n + j];
                            for p in 0..k {
                                gl[i * k + p] += d * right.data[p * n + j];
                                gr[p * n + j] += left.data[i * k + p] * d;
                            }
                        }
                    }
                    accumulate(&mut grads, a, Tensor::new(left.shape.clone(), gl)?);
                    accumulate(&mut grads, b, Tensor::new(right.shape.clone(), gr)?);
                }
            }
        }
        Ok(grads)
    }
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
        for index in 0..wv.data.len() {
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
                (analytic[index] - numeric).abs() < 1e-3,
                "{index}: {} != {numeric}",
                analytic[index]
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
