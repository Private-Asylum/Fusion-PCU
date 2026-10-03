//! Cold reverse-mode graph construction with type-preserving compact constants.
//! No host evaluation or warm scheduling lives in this module.
#[rustfmt::skip]
use alloc::{
    vec,
    vec::Vec,
};
#[rustfmt::skip]
use super::{
    Graph,
    Op,
    operands,
    PcuNumericalMode,
    OpDescriptor,
    PcuScalarType,
    TensorError,
    TensorValueId,
    ValueId,
    element_count,
};

#[rustfmt::skip]
use crate::{
    PcuCheckedFloat,
    PcuCheckedFloatConversion,
    PcuCheckedFloatWidening,
    PcuFloatUnderflowPolicy,
    dialect::tensor::constants::{count_f32, count_f64},
};

pub(super) fn append_node_gradients(
    graph: &mut Graph,
    gradients: &mut [Option<ValueId>],
    op: OpDescriptor<'_>,
    gradient: ValueId,
    needed: &[bool],
) -> Result<(), TensorError> {
    match op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {}
        OpDescriptor::Add { left, right } => {
            if needed[left.index] {
                accumulate_gradient(graph, gradients, left, gradient)?;
            }
            if needed[right.index] {
                accumulate_gradient(graph, gradients, right, gradient)?;
            }
        }
        OpDescriptor::Sub { left, right } => {
            if needed[left.index] {
                accumulate_gradient(graph, gradients, left, gradient)?;
            }
            if needed[right.index] {
                let negative = filled_like(graph, right, -1.0)?;
                let right_gradient = graph.mul(gradient, negative)?;
                accumulate_gradient(graph, gradients, right, right_gradient)?;
            }
        }
        OpDescriptor::Mul { left, right } => {
            let left_gradient = if needed[left.index] {
                Some(graph.mul(gradient, right)?)
            } else {
                None
            };
            let right_gradient = if needed[right.index] {
                Some(graph.mul(gradient, left)?)
            } else {
                None
            };
            if let Some(addition) = left_gradient {
                accumulate_gradient(graph, gradients, left, addition)?;
            }
            if let Some(addition) = right_gradient {
                accumulate_gradient(graph, gradients, right, addition)?;
            }
        }
        OpDescriptor::Div { .. } => {
            unreachable!("division is rejected by the gradient preflight")
        }
        OpDescriptor::SgdUpdate {
            weights,
            gradient: update_gradient,
            learning_rate,
        } => {
            let input_gradient = if needed[update_gradient.index] {
                // Exact sign inversion and widening must preserve a subnormal rate even
                // when an unrelated caller enabled host input flushing.
                let negative_rate = f32::from_bits(learning_rate.to_bits() ^ 0x8000_0000)
                    .pcu_checked_to_f64()
                    .map_err(|kind| TensorError::ArithmeticFault {
                        value: update_gradient,
                        element_index: 0,
                        kind,
                    })?;
                let scaled = filled_like(graph, update_gradient, negative_rate)?;
                Some(graph.mul(gradient, scaled)?)
            } else {
                None
            };
            if needed[weights.index] {
                accumulate_gradient(graph, gradients, weights, gradient)?;
            }
            if let Some(addition) = input_gradient {
                accumulate_gradient(graph, gradients, update_gradient, addition)?;
            }
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
                needed,
            )?;
        }
        OpDescriptor::MeanSquaredError { prediction, target } => {
            append_mse_gradients(graph, gradients, prediction, target, needed)?;
        }
        OpDescriptor::Relu { input } => {
            if needed[input.index] {
                let input_gradient = graph.relu_backward(input, gradient)?;
                accumulate_gradient(graph, gradients, input, input_gradient)?;
            }
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
    needed: &[bool],
) -> Result<(), TensorError> {
    if !needed[prediction.index] && !needed[target.index] {
        return Ok(());
    }
    let shape = graph.shape(prediction)?.to_vec();
    let count = element_count(&shape)?;
    let constant_fault = |kind| TensorError::ArithmeticFault {
        value: prediction,
        element_index: 0,
        kind,
    };
    let scale = match graph.nodes[prediction.index].scalar_type {
        PcuScalarType::F32 => graph
            .uniform_typed(
                shape,
                2.0_f32
                    .pcu_checked_div(count_f32(count))
                    .map_err(constant_fault)?,
            )?
            .erase(),
        PcuScalarType::F64 => {
            let scale = 2.0_f64
                .pcu_checked_div(count_f64(count))
                .map_err(constant_fault)?;
            graph.uniform_typed(shape, scale)?.erase()
        }
        scalar_type => {
            return Err(TensorError::UnsupportedScalarType {
                value: prediction,
                scalar_type,
            });
        }
    };
    let prediction_difference = if needed[prediction.index] {
        Some(graph.sub(prediction, target)?)
    } else {
        None
    };
    let target_difference = if needed[target.index] {
        Some(graph.sub(target, prediction)?)
    } else {
        None
    };
    let prediction_gradient = prediction_difference
        .map(|difference| graph.mul(difference, scale))
        .transpose()?;
    let target_gradient = target_difference
        .map(|difference| graph.mul(difference, scale))
        .transpose()?;
    if let Some(addition) = prediction_gradient {
        accumulate_gradient(graph, gradients, prediction, addition)?;
    }
    if let Some(addition) = target_gradient {
        accumulate_gradient(graph, gradients, target, addition)?;
    }
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
    needed: &[bool],
) -> Result<(), TensorError> {
    let left_gradient = if needed[left.index] {
        Some(if transpose_left {
            graph.matmul_transposed(right, gradient, transpose_right, true)?
        } else {
            graph.matmul_transposed(gradient, right, false, !transpose_right)?
        })
    } else {
        None
    };
    let right_gradient = if needed[right.index] {
        Some(if transpose_right {
            graph.matmul_transposed(gradient, left, true, transpose_left)?
        } else {
            graph.matmul_transposed(left, gradient, !transpose_left, false)?
        })
    } else {
        None
    };
    if let Some(addition) = left_gradient {
        accumulate_gradient(graph, gradients, left, addition)?;
    }
    if let Some(addition) = right_gradient {
        accumulate_gradient(graph, gradients, right, addition)?;
    }
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

fn filled_like(graph: &mut Graph, value: ValueId, scalar: f64) -> Result<ValueId, TensorError> {
    let shape = graph.shape(value)?.to_vec();
    match graph.nodes[value.index].scalar_type {
        PcuScalarType::F32 => {
            // Exact stored constants do not themselves execute graph arithmetic.
            // Keep gradual representations; the consuming node applies its policy.
            let scalar = scalar
                .pcu_checked_to_f32_with_policy(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
                .map_err(|kind| TensorError::ArithmeticFault {
                    value,
                    element_index: 0,
                    kind,
                })?;
            graph.uniform_typed(shape, scalar).map(TensorValueId::erase)
        }
        PcuScalarType::F64 => graph.uniform_typed(shape, scalar).map(TensorValueId::erase),
        scalar_type => Err(TensorError::UnsupportedScalarType { value, scalar_type }),
    }
}

pub(super) fn backward_mse(
    graph: &mut Graph,
    loss: ValueId,
) -> Result<Vec<Option<ValueId>>, TensorError> {
    build_backward_mse(graph, loss, None)
}

pub(super) fn backward_mse_for(
    graph: &mut Graph,
    loss: ValueId,
    target: ValueId,
) -> Result<ValueId, TensorError> {
    graph.shape(target)?;
    let gradients = build_backward_mse(graph, loss, Some(target))?;
    gradients[target.index].ok_or(TensorError::UnsupportedGradient(target))
}

// Validate the entire forward closure before mutation, then select derivative paths.
fn derivative_dependencies(
    graph: &Graph,
    loss: ValueId,
    target: Option<ValueId>,
) -> Result<Vec<bool>, TensorError> {
    let original_len = graph.nodes.len();
    let root = graph
        .nodes
        .get(loss.index)
        .filter(|_| loss.graph_id == graph.id)
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
        let op = &graph.nodes[index].op;
        if index != loss.index && matches!(op, Op::MeanSquaredError(..)) {
            return Err(TensorError::UnsupportedGradient(ValueId {
                graph_id: graph.id,
                index,
            }));
        }
        if matches!(op, Op::Div(..) | Op::ReluBackward(..)) {
            return Err(TensorError::UnsupportedGradient(ValueId {
                graph_id: graph.id,
                index,
            }));
        }
        pending.extend(operands(op).map(|id| id.index));
    }
    // Only propagate derivatives through values depending on the requested target.
    // Forward checked effects stay in the graph. Derivatives never requested by
    // the consumer are not source effects and must not be manufactured or fault.
    if let Some(target) = target {
        if !visited[target.index] {
            return Err(TensorError::UnsupportedGradient(target));
        }
        let mut needed = vec![false; original_len];
        needed[target.index] = true;
        for index in (target.index + 1)..=loss.index {
            needed[index] = operands(&graph.nodes[index].op).any(|id| needed[id.index]);
        }
        Ok(needed)
    } else {
        Ok(vec![true; original_len])
    }
}

fn build_backward_mse(
    graph: &mut Graph,
    loss: ValueId,
    target: Option<ValueId>,
) -> Result<Vec<Option<ValueId>>, TensorError> {
    let original_len = graph.nodes.len();
    let needed = derivative_dependencies(graph, loss, target)?;
    let mut gradients = vec![None; original_len];
    if target.is_some_and(|target| target != loss) {
        // The supported root MSE has an implicit scalar cotangent of one. Its
        // derivative rule computes 2/n directly, so a separate Uniform(1)
        // would be dead storage, not a consumed arithmetic operand.
        let Op::MeanSquaredError(prediction, target) = graph.nodes[loss.index].op else {
            unreachable!("the root was validated during gradient preflight")
        };
        with_source_scope(graph, loss.index, |graph| {
            append_mse_gradients(graph, &mut gradients, prediction, target, &needed)
        })?;
    } else {
        // The all-target result and an explicitly requested dloss/dloss expose
        // the root cotangent as a logical value, so retain its exact one.
        gradients[loss.index] = Some(with_source_scope(graph, loss.index, |graph| {
            Ok(match graph.nodes[loss.index].scalar_type {
                PcuScalarType::F32 => graph.uniform_typed([], 1.0_f32)?.erase(),
                PcuScalarType::F64 => graph.uniform_typed([], 1.0_f64)?.erase(),
                scalar_type => {
                    return Err(TensorError::UnsupportedScalarType {
                        value: loss,
                        scalar_type,
                    });
                }
            })
        })?);
    }
    for index in (0..=loss.index).rev() {
        let Some(gradient) = gradients[index] else {
            continue;
        };
        let op = match &graph.nodes[index].op {
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
            Op::Div(..) => unreachable!("division was rejected during gradient preflight"),
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
        with_source_scope(graph, index, |graph| {
            append_node_gradients(graph, &mut gradients, op, gradient, &needed)
        })?;
    }
    Ok(gradients)
}

// Differentiation is a cold graph transform. Captured source-node contracts govern the
// introduced arithmetic; mutable defaults describe future user nodes, not existing ones.
fn with_source_scope<R>(
    graph: &mut Graph,
    source: usize,
    append: impl FnOnce(&mut Graph) -> Result<R, TensorError>,
) -> Result<R, TensorError> {
    let mode = graph.nodes[source]
        .numerical_mode
        .unwrap_or(PcuNumericalMode::Boundary);
    let options = graph.nodes[source].numerical_options;
    let underflow = graph.nodes[source].float_underflow_policy;
    let previous_mode = graph.numerical_mode;
    let previous_options = graph.numerical_options;
    graph.numerical_mode = mode;
    graph.numerical_options = options;
    let first_new = graph.nodes.len();
    let result = append(graph);
    if let Some(policy) = underflow {
        for node in &mut graph.nodes[first_new..] {
            if node.float_underflow_policy.is_some() {
                node.float_underflow_policy = Some(policy);
            }
        }
    }
    graph.numerical_mode = previous_mode;
    graph.numerical_options = previous_options;
    result
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
