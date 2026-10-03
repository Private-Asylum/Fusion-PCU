//! Independent direct checked host workload, with the same owned output/readback boundary.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuScalarType,
    dialect::tensor::{
        TensorElement,
        TensorScalarValue,
    },
};
use std::rc::Rc;

pub struct HostOutput<T> {
    data: Vec<T>,
    _shape: Rc<[usize]>,
}
impl<T> HostOutput<T> {
    pub fn new(data: Vec<T>, shape: &Rc<[usize]>) -> Self {
        Self {
            data,
            _shape: Rc::clone(shape),
        }
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
}
fn zero<T: TensorElement + PcuCheckedFloat>() -> T {
    scalar(0.0)
}
fn scalar<T: TensorElement + PcuCheckedFloat>(value: f32) -> T {
    T::as_scalar(if T::TYPE == PcuScalarType::F32 {
        TensorScalarValue::F32(value)
    } else {
        TensorScalarValue::F64(f64::from(value))
    })
    .unwrap()
}
#[allow(clippy::cast_precision_loss)] // Same declared nearest-even count conversion as the core compound contract.
fn divisor<T: TensorElement + PcuCheckedFloat>(count: usize) -> T {
    T::as_scalar(if T::TYPE == PcuScalarType::F32 {
        TensorScalarValue::F32(count as f32)
    } else {
        TensorScalarValue::F64(count as f64)
    })
    .unwrap()
}
fn loss_value<T: TensorElement + PcuCheckedFloat>(
    prediction: &[T],
    target: &[T],
) -> Result<T, PcuExecutionFaultKind> {
    let mut total = zero::<T>();
    for (&prediction, &target) in prediction.iter().zip(target) {
        let difference = prediction.pcu_checked_sub(target)?;
        let square = difference.pcu_checked_mul(difference)?;
        total = total.pcu_checked_add(square)?;
    }
    total.pcu_checked_div(divisor(prediction.len()))
}
pub fn loss<T: TensorElement + PcuCheckedFloat>(
    prediction: &[T],
    target: &[T],
    shape: &Rc<[usize]>,
) -> Result<HostOutput<T>, PcuExecutionFaultKind> {
    loss_value(prediction, target).map(|value| HostOutput::new(vec![value], shape))
}
pub fn training<T: TensorElement + PcuCheckedFloat>(
    input: &[[T; 2]; 2],
    transpose: &[[T; 2]; 2],
    weights: &[[T; 1]; 2],
    target: &[[T; 1]; 2],
    shape: &Rc<[usize]>,
) -> Result<HostOutput<T>, PcuExecutionFaultKind> {
    let mut activation = [zero::<T>(); 2];
    for (row, input) in input.iter().enumerate() {
        for (&input, weight) in input.iter().zip(weights) {
            activation[row] = activation[row].pcu_checked_add(input.pcu_checked_mul(weight[0])?)?;
        }
    }
    let prediction = [
        activation[0].pcu_checked_relu()?,
        activation[1].pcu_checked_relu()?,
    ];
    let _loss = loss_value(&prediction, &[target[0][0], target[1][0]])?;
    let difference = [
        prediction[0].pcu_checked_sub(target[0][0])?,
        prediction[1].pcu_checked_sub(target[1][0])?,
    ];
    let derivative = [
        activation[0].pcu_checked_relu_backward_with_policy(
            difference[0],
            PcuFloatUnderflowPolicy::default(),
        )?,
        activation[1].pcu_checked_relu_backward_with_policy(
            difference[1],
            PcuFloatUnderflowPolicy::default(),
        )?,
    ];
    let mut gradient = [zero::<T>(); 2];
    for (row, input) in transpose.iter().enumerate() {
        for (&input, &derivative) in input.iter().zip(&derivative) {
            gradient[row] = gradient[row].pcu_checked_add(input.pcu_checked_mul(derivative)?)?;
        }
    }
    let rate = scalar::<T>(0.5);
    let output = vec![
        weights[0][0].pcu_checked_sub(rate.pcu_checked_mul(gradient[0])?)?,
        weights[1][0].pcu_checked_sub(rate.pcu_checked_mul(gradient[1])?)?,
    ];
    Ok(HostOutput::new(output, shape))
}
