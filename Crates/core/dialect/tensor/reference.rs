//! CPU reference value dispatch and numeric helpers for tensor graphs.
//!
//! This module keeps evaluator behavior separate from graph construction and storage planning.

#[rustfmt::skip]
use alloc::{
    vec,
    vec::Vec,
};
#[rustfmt::skip]
use super::{
    BinaryOp,
    Graph,
    Tensor,
    TensorElement,
    TensorError,
    TensorValue,
    ValueId,
};
use crate::core::PcuScalarType;
#[rustfmt::skip]
use crate::{
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
};
use crate::scalar_checked::PcuCheckedInteger;
use crate::scalar_checked_float::PcuCheckedFloat;

#[path = "reference/strict_matmul/strict_matmul.rs"]
mod strict_matmul;
use strict_matmul::checked_matmul;

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
    float_underflow_policy: Option<PcuFloatUnderflowPolicy>,
) -> Result<TensorValue, TensorError> {
    match (left, right) {
        (TensorValue::F32(left), TensorValue::F32(right)) => checked_float_value(
            left,
            right,
            output,
            operation,
            float_underflow_policy.unwrap_or_default(),
        )
        .map(TensorValue::F32),
        (TensorValue::F64(left), TensorValue::F64(right)) => checked_float_value(
            left,
            right,
            output,
            operation,
            float_underflow_policy.unwrap_or_default(),
        )
        .map(TensorValue::F64),
        (TensorValue::U8(left), TensorValue::U8(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::U8)
        }
        (TensorValue::U16(left), TensorValue::U16(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::U16)
        }
        (TensorValue::U32(left), TensorValue::U32(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::U32)
        }
        (TensorValue::U64(left), TensorValue::U64(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::U64)
        }
        (TensorValue::I8(left), TensorValue::I8(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::I8)
        }
        (TensorValue::I16(left), TensorValue::I16(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::I16)
        }
        (TensorValue::I32(left), TensorValue::I32(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::I32)
        }
        (TensorValue::I64(left), TensorValue::I64(right)) => {
            checked_integer_value(left, right, output, operation).map(TensorValue::I64)
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

fn checked_float_value<T: PcuCheckedFloat + TensorElement>(
    left: &Tensor<T>,
    right: &Tensor<T>,
    output: ValueId,
    operation: BinaryOp,
    underflow_policy: PcuFloatUnderflowPolicy,
) -> Result<Tensor<T>, TensorError> {
    let data = left
        .data
        .iter()
        .zip(&right.data)
        .enumerate()
        .map(|(element_index, (left, right))| {
            let result = match operation {
                BinaryOp::Add => left.pcu_checked_add_with_policy(*right, underflow_policy),
                BinaryOp::Sub => left.pcu_checked_sub_with_policy(*right, underflow_policy),
                BinaryOp::Mul => left.pcu_checked_mul_with_policy(*right, underflow_policy),
                BinaryOp::Div => left.pcu_checked_div_with_policy(*right, underflow_policy),
            };
            result.map_err(|kind| TensorError::ArithmeticFault {
                value: output,
                element_index,
                kind,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor {
        shape: left.shape.clone(),
        data,
        known_uniform_value: None,
    })
}

fn checked_integer_value<T: PcuCheckedInteger>(
    left: &Tensor<T>,
    right: &Tensor<T>,
    output: ValueId,
    operation: BinaryOp,
) -> Result<Tensor<T>, TensorError> {
    let data = left
        .data
        .iter()
        .zip(&right.data)
        .enumerate()
        .map(|(element_index, (left, right))| {
            let result = match operation {
                BinaryOp::Add => left.pcu_checked_add(*right),
                BinaryOp::Sub => left.pcu_checked_sub(*right),
                BinaryOp::Mul => left.pcu_checked_mul(*right),
                BinaryOp::Div => {
                    return Err(unsupported_value(output.graph_id, output.index, T::TYPE));
                }
            };
            result.map_err(|kind| TensorError::ArithmeticFault {
                value: output,
                element_index,
                kind,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor {
        shape: left.shape.clone(),
        data,
        known_uniform_value: None,
    })
}

pub(super) fn relu_value(input: &TensorValue, output: ValueId) -> Result<TensorValue, TensorError> {
    match input {
        TensorValue::F32(tensor) => {
            let data = tensor
                .data
                .iter()
                .enumerate()
                .map(|(element_index, value)| {
                    value
                        .pcu_checked_relu()
                        .map_err(|kind| TensorError::ArithmeticFault {
                            value: output,
                            element_index,
                            kind,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TensorValue::F32(Tensor::new(tensor.shape.clone(), data)?))
        }
        TensorValue::F64(tensor) => {
            let data = tensor
                .data
                .iter()
                .enumerate()
                .map(|(element_index, value)| {
                    value
                        .pcu_checked_relu()
                        .map_err(|kind| TensorError::ArithmeticFault {
                            value: output,
                            element_index,
                            kind,
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(TensorValue::F64(Tensor::new(tensor.shape.clone(), data)?))
        }
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
    numerical_mode: PcuNumericalMode,
    underflow_policy: PcuFloatUnderflowPolicy,
) -> Result<TensorValue, TensorError> {
    match (left, right) {
        (TensorValue::F32(left), TensorValue::F32(right)) => {
            Ok(TensorValue::F32(match numerical_mode {
                PcuNumericalMode::Strict => checked_matmul(
                    left,
                    right,
                    transpose_left,
                    transpose_right,
                    output,
                    underflow_policy,
                    0.0,
                )?,
                PcuNumericalMode::Boundary => matmul(left, right, transpose_left, transpose_right),
            }))
        }
        (TensorValue::F64(left), TensorValue::F64(right)) => {
            Ok(TensorValue::F64(match numerical_mode {
                PcuNumericalMode::Strict => checked_matmul(
                    left,
                    right,
                    transpose_left,
                    transpose_right,
                    output,
                    underflow_policy,
                    0.0,
                )?,
                PcuNumericalMode::Boundary => {
                    matmul_f64(left, right, transpose_left, transpose_right)
                }
            }))
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

#[cfg(test)]
#[path = "reference/integer_tests.rs"]
mod integer_tests;

#[cfg(test)]
#[path = "reference/float_tests.rs"]
mod float_tests;
