//! Ordered strict mean squared error, independent of vendor reduction order.
//!
//! PCU specifies increasing flattened element order: rounded checked subtraction,
//! squared difference, addition to a +0 accumulator, then one checked division.
//! No FMA, pairwise reduction or reciprocal substitution is permitted. IEEE 754-2019
//! clauses 4.3.1 and 7.5 govern scalar rounding and tininess; this order and the
//! element-count-to-float conversion are PCU's compound contract, not IEEE mandates.
#[rustfmt::skip]
use super::{
    Tensor,
    TensorElement,
    TensorError,
    ValueId,
};
#[rustfmt::skip]
use crate::{
    PcuCheckedFloat,
    PcuFloatUnderflowPolicy,
    dialect::tensor::TensorArithmeticStep,
};

pub(super) fn checked_mse<T: PcuCheckedFloat + TensorElement>(
    prediction: &Tensor<T>,
    target: &Tensor<T>,
    output: ValueId,
    policy: PcuFloatUnderflowPolicy,
    zero: T,
    denominator: T,
) -> Result<Tensor<T>, TensorError> {
    let mut total = zero;
    for (index, (&prediction, &target)) in prediction.data.iter().zip(&target.data).enumerate() {
        let fault = |step, kind| TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 0,
            reduction_index: index,
            step,
            kind,
        };
        let difference = prediction
            .pcu_checked_sub_with_policy(target, policy)
            .map_err(|kind| fault(TensorArithmeticStep::Subtract, kind))?;
        let square = difference
            .pcu_checked_mul_with_policy(difference, policy)
            .map_err(|kind| fault(TensorArithmeticStep::Multiply, kind))?;
        total = total
            .pcu_checked_add_with_policy(square, policy)
            .map_err(|kind| fault(TensorArithmeticStep::Add, kind))?;
    }
    total
        .pcu_checked_div_with_policy(denominator, policy)
        .map(Tensor::scalar)
        .map_err(|kind| TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 0,
            reduction_index: prediction.data.len(),
            step: TensorArithmeticStep::Divide,
            kind,
        })
}

#[cfg(test)]
mod tests;
