//! Strict SGD: independently rounded, checked multiply then subtract, without contraction.
//!
//! IEEE 754-derived scalar round-to-nearest-even and tininess-after-rounding rules are
//! delegated to the integer-only checked scalar oracle. Element/step ordering is PCU policy.
use alloc::vec::Vec;
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

pub(super) fn checked_sgd<T: PcuCheckedFloat + TensorElement>(
    weights: &Tensor<T>,
    gradient: &Tensor<T>,
    rate: T,
    value: ValueId,
    policy: PcuFloatUnderflowPolicy,
) -> Result<Tensor<T>, TensorError> {
    let mut data = Vec::with_capacity(weights.data.len());
    for (element_index, (&weight, &gradient)) in weights.data.iter().zip(&gradient.data).enumerate()
    {
        let fault = |step, kind| TensorError::CompoundArithmeticFault {
            value,
            element_index,
            reduction_index: 0,
            step,
            kind,
        };
        let product = rate
            .pcu_checked_mul_with_policy(gradient, policy)
            .map_err(|kind| fault(TensorArithmeticStep::Multiply, kind))?;
        let updated = weight
            .pcu_checked_sub_with_policy(product, policy)
            .map_err(|kind| fault(TensorArithmeticStep::Subtract, kind))?;
        data.push(updated);
    }
    Tensor::new(weights.shape.clone(), data)
}

#[cfg(test)]
mod tests;
