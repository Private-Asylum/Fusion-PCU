//! Indexed checked tensor steps; operation selection and shapes are frozen before execution.
// Indexed storage allows repeated operands and liveness reuse without unsafe split borrows.
#![allow(clippy::needless_range_loop)]
#[rustfmt::skip]
use alloc::vec::Vec;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    NodeDescriptor,
    OpDescriptor,
    TensorArithmeticStep,
    TensorElement,
    TensorError,
    ValueId,
};
#[rustfmt::skip]
use super::{
    PcuCpuTensorBinding,
    denominator,
    scalar,
    zero,
};

type Runner<T> = fn(&mut [Vec<T>], &Step<T>) -> Result<(), TensorError>;

pub(super) struct Step<T> {
    pub run: Runner<T>,
    value: ValueId,
    out: usize,
    left: usize,
    right: usize,
    count: usize,
    policy: PcuFloatUnderflowPolicy,
    factor: T,
    rows: usize,
    columns: usize,
    inner: usize,
    left_stride: usize,
    right_stride: usize,
}

pub(super) fn compile<T: TensorElement + PcuCheckedFloat>(
    node: NodeDescriptor<'_>,
    bindings: &[PcuCpuTensorBinding],
) -> Result<Option<Step<T>>, TensorError> {
    let find = |value| {
        bindings
            .iter()
            .find(|binding| binding.value == value)
            .ok_or(TensorError::UnknownValue(value))
    };
    let output = find(node.value)?;
    let mut step = Step {
        run: elementwise::<T, 0>,
        value: node.value,
        out: output.slot,
        left: 0,
        right: 0,
        count: output.element_count,
        policy: node.float_underflow_policy.unwrap_or_default(),
        factor: zero(),
        rows: 0,
        columns: 0,
        inner: 0,
        left_stride: 0,
        right_stride: 0,
    };
    let (left, right) = match node.op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => {
            return Ok(None);
        }
        OpDescriptor::Add { left, right } => (left, right),
        OpDescriptor::Sub { left, right } => {
            step.run = elementwise::<T, 1>;
            (left, right)
        }
        OpDescriptor::Mul { left, right } => {
            step.run = elementwise::<T, 2>;
            (left, right)
        }
        OpDescriptor::Div { left, right } => {
            step.run = elementwise::<T, 3>;
            (left, right)
        }
        OpDescriptor::Relu { input } => {
            step.run = relu;
            (input, input)
        }
        OpDescriptor::ReluBackward { input, upstream } => {
            step.run = relu_backward;
            (input, upstream)
        }
        OpDescriptor::SgdUpdate {
            weights,
            gradient,
            learning_rate,
        } => {
            strict(node)?;
            step.run = sgd;
            step.factor = scalar(learning_rate);
            (weights, gradient)
        }
        OpDescriptor::MeanSquaredError { prediction, target } => {
            strict(node)?;
            step.run = mse;
            step.inner = find(prediction)?.element_count;
            step.factor = denominator(step.inner);
            (prediction, target)
        }
        OpDescriptor::MatMul {
            left,
            right,
            transpose_left,
            transpose_right,
        } => {
            strict(node)?;
            step.run = match (transpose_left, transpose_right) {
                (false, false) => matmul::<T, false, false>,
                (false, true) => matmul::<T, false, true>,
                (true, false) => matmul::<T, true, false>,
                (true, true) => matmul::<T, true, true>,
            };
            let left_shape = &find(left)?.shape;
            let right_shape = &find(right)?.shape;
            if left_shape.len() != 2 || right_shape.len() != 2 {
                return Err(TensorError::MatMulShape {
                    left: left_shape.clone(),
                    right: right_shape.clone(),
                });
            }
            step.left_stride = left_shape[1];
            step.right_stride = right_shape[1];
            step.rows = left_shape[usize::from(transpose_left)];
            step.inner = left_shape[usize::from(!transpose_left)];
            step.columns = right_shape[usize::from(!transpose_right)];
            (left, right)
        }
    };
    step.left = find(left)?.slot;
    step.right = find(right)?.slot;
    // Core's inclusive scratch lifetimes exclude output/input overlap at every operation.
    // Keep this check cold so execution never relies on unsafe alias assumptions.
    if step.out == step.left || step.out == step.right {
        return Err(TensorError::UnknownValue(node.value));
    }
    Ok(Some(step))
}

fn strict(node: NodeDescriptor<'_>) -> Result<(), TensorError> {
    if node.numerical_mode != Some(PcuNumericalMode::Strict) {
        return Err(TensorError::UnsupportedNumericalMode {
            value: node.value,
            mode: node.numerical_mode.unwrap_or(PcuNumericalMode::Boundary),
        });
    }
    Ok(())
}

const fn fault<T>(
    step: &Step<T>,
    element_index: usize,
    kind: PcuExecutionFaultKind,
) -> TensorError {
    TensorError::ArithmeticFault {
        value: step.value,
        element_index,
        kind,
    }
}
const fn compound<T>(
    step: &Step<T>,
    element_index: usize,
    reduction_index: usize,
    arithmetic_step: TensorArithmeticStep,
    kind: PcuExecutionFaultKind,
) -> TensorError {
    TensorError::CompoundArithmeticFault {
        value: step.value,
        element_index,
        reduction_index,
        step: arithmetic_step,
        kind,
    }
}

fn elementwise<T: PcuCheckedFloat, const OP: u8>(
    storage: &mut [Vec<T>],
    step: &Step<T>,
) -> Result<(), TensorError> {
    for index in 0..step.count {
        let left = storage[step.left][index];
        let right = storage[step.right][index];
        // Cold preparation selects one monomorphized runner. OP has no per-element dispatch.
        let value = match OP {
            0 => left.pcu_checked_add_with_policy(right, step.policy),
            1 => left.pcu_checked_sub_with_policy(right, step.policy),
            2 => left.pcu_checked_mul_with_policy(right, step.policy),
            3 => left.pcu_checked_div_with_policy(right, step.policy),
            _ => unreachable!("only four cold-frozen binary runners exist"),
        }
        .map_err(|kind| fault(step, index, kind))?;
        storage[step.out][index] = value;
    }
    Ok(())
}
fn relu<T: PcuCheckedFloat>(storage: &mut [Vec<T>], step: &Step<T>) -> Result<(), TensorError> {
    for index in 0..step.count {
        storage[step.out][index] = storage[step.left][index]
            .pcu_checked_relu_with_policy(step.policy)
            .map_err(|kind| fault(step, index, kind))?;
    }
    Ok(())
}
fn relu_backward<T: PcuCheckedFloat>(
    storage: &mut [Vec<T>],
    step: &Step<T>,
) -> Result<(), TensorError> {
    for index in 0..step.count {
        storage[step.out][index] = storage[step.left][index]
            .pcu_checked_relu_backward_with_policy(storage[step.right][index], step.policy)
            .map_err(|kind| fault(step, index, kind))?;
    }
    Ok(())
}
fn sgd<T: PcuCheckedFloat>(storage: &mut [Vec<T>], step: &Step<T>) -> Result<(), TensorError> {
    for index in 0..step.count {
        let product = step
            .factor
            .pcu_checked_mul_with_policy(storage[step.right][index], step.policy)
            .map_err(|kind| compound(step, index, 0, TensorArithmeticStep::Multiply, kind))?;
        storage[step.out][index] = storage[step.left][index]
            .pcu_checked_sub_with_policy(product, step.policy)
            .map_err(|kind| compound(step, index, 0, TensorArithmeticStep::Subtract, kind))?;
    }
    Ok(())
}
fn matmul<
    T: TensorElement + PcuCheckedFloat,
    const TRANSPOSE_LEFT: bool,
    const TRANSPOSE_RIGHT: bool,
>(
    storage: &mut [Vec<T>],
    step: &Step<T>,
) -> Result<(), TensorError> {
    for row in 0..step.rows {
        for column in 0..step.columns {
            let index = row * step.columns + column;
            let mut total = scalar::<T>(0.0);
            for reduction in 0..step.inner {
                let left = if TRANSPOSE_LEFT {
                    reduction * step.left_stride + row
                } else {
                    row * step.left_stride + reduction
                };
                let right = if TRANSPOSE_RIGHT {
                    column * step.right_stride + reduction
                } else {
                    reduction * step.right_stride + column
                };
                let product = storage[step.left][left]
                    .pcu_checked_mul_with_policy(storage[step.right][right], step.policy)
                    .map_err(|kind| {
                        compound(step, index, reduction, TensorArithmeticStep::Multiply, kind)
                    })?;
                total = total
                    .pcu_checked_add_with_policy(product, step.policy)
                    .map_err(|kind| {
                        compound(step, index, reduction, TensorArithmeticStep::Add, kind)
                    })?;
            }
            storage[step.out][index] = total;
        }
    }
    Ok(())
}
fn mse<T: TensorElement + PcuCheckedFloat>(
    storage: &mut [Vec<T>],
    step: &Step<T>,
) -> Result<(), TensorError> {
    let mut total = scalar::<T>(0.0);
    for index in 0..step.inner {
        let difference = storage[step.left][index]
            .pcu_checked_sub_with_policy(storage[step.right][index], step.policy)
            .map_err(|kind| compound(step, 0, index, TensorArithmeticStep::Subtract, kind))?;
        let square = difference
            .pcu_checked_mul_with_policy(difference, step.policy)
            .map_err(|kind| compound(step, 0, index, TensorArithmeticStep::Multiply, kind))?;
        total = total
            .pcu_checked_add_with_policy(square, step.policy)
            .map_err(|kind| compound(step, 0, index, TensorArithmeticStep::Add, kind))?;
    }
    storage[step.out][0] = total
        .pcu_checked_div_with_policy(step.factor, step.policy)
        .map_err(|kind| compound(step, 0, step.inner, TensorArithmeticStep::Divide, kind))?;
    Ok(())
}
