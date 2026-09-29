//! CPU reference value dispatch and numeric helpers for tensor graphs.
//!
//! This module keeps evaluator behavior separate from graph construction and storage planning.

#[rustfmt::skip]
use alloc::{
    vec,
};
#[rustfmt::skip]
use super::{
    BinaryOp,
    Graph,
    Tensor,
    TensorError,
    TensorValue,
    ValueId,
};
use crate::core::PcuScalarType;

pub(super) const fn unsupported_value(
    graph_id: u64,
    index: usize,
    scalar_type: PcuScalarType,
) -> TensorError {
    TensorError::UnsupportedScalarType {
        value: ValueId { graph_id, index },
        scalar_type,
    }
}

pub(super) fn as_f32_value(
    value: &TensorValue,
    operation: ValueId,
) -> Result<&Tensor, TensorError> {
    value
        .as_typed::<f32>()
        .map_err(|_| unsupported_value(operation.graph_id, operation.index, value.scalar_type()))
}

pub(super) fn binary_value(
    left: &TensorValue,
    right: &TensorValue,
    output: ValueId,
    operation: BinaryOp,
) -> Result<TensorValue, TensorError> {
    match (left, right) {
        (TensorValue::F32(left), TensorValue::F32(right)) => {
            let result = binary(left, right, |a, b| match operation {
                BinaryOp::Add => a + b,
                BinaryOp::Sub => a - b,
                BinaryOp::Mul => a * b,
            });
            Ok(TensorValue::F32(result))
        }
        (TensorValue::F64(left), TensorValue::F64(right)) => {
            let data = left
                .data
                .iter()
                .zip(&right.data)
                .map(|(a, b)| match operation {
                    BinaryOp::Add => a + b,
                    BinaryOp::Sub => a - b,
                    BinaryOp::Mul => a * b,
                })
                .collect();
            Ok(TensorValue::F64(Tensor::new(left.shape.clone(), data)?))
        }
        (left, right) if left.scalar_type() != right.scalar_type() => {
            Err(TensorError::ScalarTypeMismatch {
                value: output,
                expected: left.scalar_type(),
                actual: right.scalar_type(),
            })
        }
        (left, _) => Err(unsupported_value(
            output.graph_id,
            output.index,
            left.scalar_type(),
        )),
    }
}

pub(super) fn relu_value(input: &TensorValue, output: ValueId) -> Result<TensorValue, TensorError> {
    match input {
        TensorValue::F32(tensor) => Ok(TensorValue::F32(Tensor::new(
            tensor.shape.clone(),
            tensor.data.iter().map(|value| value.max(0.0)).collect(),
        )?)),
        TensorValue::F64(tensor) => Ok(TensorValue::F64(Tensor::new(
            tensor.shape.clone(),
            tensor.data.iter().map(|value| value.max(0.0)).collect(),
        )?)),
        other => Err(unsupported_value(
            output.graph_id,
            output.index,
            other.scalar_type(),
        )),
    }
}

pub(super) fn relu_backward_value(
    input: &TensorValue,
    upstream: &TensorValue,
    output: ValueId,
) -> Result<TensorValue, TensorError> {
    match (input, upstream) {
        (TensorValue::F32(input), TensorValue::F32(upstream)) => Ok(TensorValue::F32(Tensor::new(
            input.shape.clone(),
            input
                .data
                .iter()
                .zip(&upstream.data)
                .map(|(value, gradient)| if *value > 0.0 { *gradient } else { 0.0 })
                .collect(),
        )?)),
        (input, upstream) if input.scalar_type() != upstream.scalar_type() => {
            Err(TensorError::ScalarTypeMismatch {
                value: output,
                expected: input.scalar_type(),
                actual: upstream.scalar_type(),
            })
        }
        (input, _) => Err(unsupported_value(
            output.graph_id,
            output.index,
            input.scalar_type(),
        )),
    }
}

pub(super) fn matmul_value(
    left: &TensorValue,
    right: &TensorValue,
    transpose_left: bool,
    transpose_right: bool,
    output: ValueId,
) -> Result<TensorValue, TensorError> {
    match (left, right) {
        (TensorValue::F32(left), TensorValue::F32(right)) => Ok(TensorValue::F32(matmul(
            left,
            right,
            transpose_left,
            transpose_right,
        ))),
        (TensorValue::F64(left), TensorValue::F64(right)) => Ok(TensorValue::F64(matmul_f64(
            left,
            right,
            transpose_left,
            transpose_right,
        ))),
        (left, right) if left.scalar_type() != right.scalar_type() => {
            Err(TensorError::ScalarTypeMismatch {
                value: output,
                expected: left.scalar_type(),
                actual: right.scalar_type(),
            })
        }
        (left, _) => Err(unsupported_value(
            output.graph_id,
            output.index,
            left.scalar_type(),
        )),
    }
}

pub(super) fn mean_squared_error_value(
    prediction: &TensorValue,
    target: &TensorValue,
    output: ValueId,
) -> Result<TensorValue, TensorError> {
    match (prediction, target) {
        (TensorValue::F32(prediction), TensorValue::F32(target)) => {
            let divisor = mean_denominator(prediction.data.len());
            Ok(TensorValue::F32(Tensor::scalar(
                prediction
                    .data
                    .iter()
                    .zip(&target.data)
                    .map(|(left, right)| (left - right) * (left - right))
                    .sum::<f32>()
                    / divisor,
            )))
        }
        (prediction, target) if prediction.scalar_type() != target.scalar_type() => {
            Err(TensorError::ScalarTypeMismatch {
                value: output,
                expected: prediction.scalar_type(),
                actual: target.scalar_type(),
            })
        }
        (prediction, _) => Err(unsupported_value(
            output.graph_id,
            output.index,
            prediction.scalar_type(),
        )),
    }
}

pub(super) fn execution_f32_value<'a>(
    values: &[Option<&'a Tensor>],
    graph: &Graph,
    value: ValueId,
) -> Result<&'a Tensor, TensorError> {
    if let Some(tensor) = values.get(value.index).copied().flatten() {
        return Ok(tensor);
    }
    Err(unsupported_value(
        value.graph_id,
        value.index,
        graph.node(value)?.scalar_type,
    ))
}

pub(super) fn binary(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
    Tensor {
        shape: a.shape.clone(),
        data: a.data.iter().zip(&b.data).map(|(x, y)| f(*x, *y)).collect(),
        known_uniform_value: None,
    }
}

// Keep the CPU reference's multiply and accumulation steps explicit; fused rounding is a
// backend numerical choice and would change this oracle's results.
#[allow(clippy::suboptimal_flops)]
pub(super) fn matmul(
    left: &Tensor,
    right: &Tensor,
    transpose_left: bool,
    transpose_right: bool,
) -> Tensor {
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

#[allow(clippy::suboptimal_flops)] // Keep reference accumulation order explicit, as for f32.
pub(super) fn matmul_f64(
    left: &Tensor<f64>,
    right: &Tensor<f64>,
    transpose_left: bool,
    transpose_right: bool,
) -> Tensor<f64> {
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
    let mut output = vec![0.0_f64; rows * columns];
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
pub(super) const fn mean_denominator(element_count: usize) -> f32 {
    element_count as f32
}
