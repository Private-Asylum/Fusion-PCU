//! Typed construction of the exact joint division contract.
#[rustfmt::skip]
use super::{
    PcuError,
    PcuScalarLowering,
    PcuScalarValue,
};
#[rustfmt::skip]
use crate::{
    PcuCheckedIntegerDivision,
    PcuDispatchDataOp,
    PcuRangePolicy,
    PcuValueType,
};
use crate::model::PcuIntegerDivFlags;

impl<const MAX_OPS: usize> PcuScalarLowering<'_, MAX_OPS> {
    /// Emits checked exact-width quotient and remainder with distinct typed SSA values.
    ///
    /// This cold construction does not evaluate operands or hide arithmetic errors:
    /// zero divisor and signed MIN/-1 remain terminal execution faults. It always
    /// retains the joint CHECKED contract, never a wrapping or total alternative.
    /// The current joint operation has no range-recovery law, so a Clamp lowering
    /// context rejects before reserving values or emitting an instruction.
    /// Complete assembled verification and backend admission remain necessary;
    /// this helper does not imply composed-division or Portable execution support.
    ///
    /// # Errors
    /// Returns an unsupported-policy error for Clamp or resource exhaustion when value/operation
    /// capacity is exhausted. As with other lowering errors, discard the context
    /// after capacity failure; no runtime or usable arithmetic payload is produced.
    pub fn checked_div_rem_values<T: PcuCheckedIntegerDivision>(
        &mut self,
        lhs: PcuScalarValue<T>,
        rhs: PcuScalarValue<T>,
    ) -> Result<(PcuScalarValue<T>, PcuScalarValue<T>), PcuError> {
        if self.range_policy() != PcuRangePolicy::Reject {
            return Err(PcuError::unsupported());
        }
        let quotient = self.fresh_value()?;
        let remainder = self.fresh_value()?;
        self.push(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(T::TYPE),
            flags: PcuIntegerDivFlags::CHECKED,
            quotient,
            remainder,
            lhs: lhs.id(),
            rhs: rhs.id(),
        })?;
        Ok((
            PcuScalarValue::from_id(quotient),
            PcuScalarValue::from_id(remainder),
        ))
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
