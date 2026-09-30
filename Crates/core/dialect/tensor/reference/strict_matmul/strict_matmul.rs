//! Checked, ordered host `MatMul` reference with independently classified arithmetic steps.
#[rustfmt::skip]
use alloc::{
    vec,
};
#[rustfmt::skip]
use super::{
    Tensor,
    TensorElement,
    TensorError,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    PcuFloatUnderflowPolicy,
    dialect::tensor::TensorArithmeticStep,
    scalar_checked_float::PcuCheckedFloat,
};

/// Strict ordered dot products: each multiply rounds and checks before each checked add.
/// No wider accumulation, contraction, or reassociation is used.
pub(super) fn checked_matmul<T: PcuCheckedFloat + TensorElement>(
    left: &Tensor<T>,
    right: &Tensor<T>,
    transpose_left: bool,
    transpose_right: bool,
    value: ValueId,
    underflow_policy: PcuFloatUnderflowPolicy,
    zero: T,
) -> Result<Tensor<T>, TensorError> {
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
    let mut data = vec![zero; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            let element_index = row * columns + column;
            for reduction_index in 0..inner {
                let left_index = if transpose_left {
                    reduction_index * left.shape[1] + row
                } else {
                    row * left.shape[1] + reduction_index
                };
                let right_index = if transpose_right {
                    column * right.shape[1] + reduction_index
                } else {
                    reduction_index * right.shape[1] + column
                };
                let fault = |step, kind| TensorError::CompoundArithmeticFault {
                    value,
                    element_index,
                    reduction_index,
                    step,
                    kind,
                };
                let product = left.data[left_index]
                    .pcu_checked_mul_with_policy(right.data[right_index], underflow_policy)
                    .map_err(|kind| fault(TensorArithmeticStep::Multiply, kind))?;
                data[element_index] = data[element_index]
                    .pcu_checked_add_with_policy(product, underflow_policy)
                    .map_err(|kind| fault(TensorArithmeticStep::Add, kind))?;
            }
        }
    }
    Tensor::new(vec![rows, columns], data)
}

#[cfg(test)]
mod tests;
