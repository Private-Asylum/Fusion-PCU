//! Exact integer quotient and remainder across the sealed PCU integer family.
//!
//! Quotients truncate toward zero and signed remainders retain the dividend's sign.
//! Zero divisors and signed MIN/-1 reject before invalid native arithmetic. PCU
//! deliberately rejects MIN%-1 as well, matching its shared checked divide contract.
//! This finite-width integer contract is independent of IEEE floating division.
#[rustfmt::skip]
use crate::{
    PcuCheckedInteger,
    PcuExecutionFaultKind,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};

/// Explicit checked division and remainder; no implicit total or wrapping alternative.
///
/// Implementations are sealed by [`PcuCheckedInteger`]. Host support does not
/// claim execution support from every backend or extend a backend's physical ABI.
pub trait PcuCheckedIntegerDivision: PcuCheckedInteger {
    /// Validates the exact division domain without computing either result.
    ///
    /// For these sealed, same-width integers, every nonzero unsigned divisor is
    /// valid. Signed division additionally excludes MIN/-1. This also covers
    /// remainder under PCU's joint contract. Executors may preflight an entire
    /// immutable input map before computing and publishing each result once.
    /// # Errors
    /// Returns `DivideByZero` or `SignedDivisionOverflow`; no division is evaluated.
    fn pcu_div_rem_domain(self, rhs: Self) -> Result<(), PcuExecutionFaultKind>;
    /// Returns the quotient and remainder from one exact checked division.
    ///
    /// Wide carriers compute their limb division once. Primitive implementations
    /// guard the quotient first, so the remainder never evaluates a zero divisor
    /// or signed MIN/-1. This is the same joint fault contract as checked `DivRem`.
    /// # Errors
    /// Returns `DivideByZero` or `SignedDivisionOverflow` without either result.
    fn pcu_checked_div_rem(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind>;
    /// Returns a quotient truncated toward zero.
    /// # Errors
    /// Returns `DivideByZero` or `SignedDivisionOverflow` without producing a result.
    fn pcu_checked_div(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Returns the remainder with the dividend's sign.
    /// # Errors
    /// Returns `DivideByZero` or `SignedDivisionOverflow` without producing a result.
    fn pcu_checked_rem(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
}

macro_rules! unsigned {
    ($($ty:ty),+ $(,)?) => {$(
        impl PcuCheckedIntegerDivision for $ty {
            #[inline]
            fn pcu_div_rem_domain(self, rhs: Self) -> Result<(), PcuExecutionFaultKind> {
                if rhs == 0 { Err(PcuExecutionFaultKind::DivideByZero) } else { Ok(()) }
            }
            fn pcu_checked_div_rem(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                let quotient = self.pcu_checked_div(rhs)?;
                Ok((quotient, self % rhs))
            }
            fn pcu_checked_div(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_div(rhs).ok_or(PcuExecutionFaultKind::DivideByZero)
            }
            fn pcu_checked_rem(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_rem(rhs).ok_or(PcuExecutionFaultKind::DivideByZero)
            }
        }
    )+};
}
macro_rules! signed {
    ($($ty:ty),+ $(,)?) => {$(
        impl PcuCheckedIntegerDivision for $ty {
            #[inline]
            fn pcu_div_rem_domain(self, rhs: Self) -> Result<(), PcuExecutionFaultKind> {
                if rhs == 0 { Err(PcuExecutionFaultKind::DivideByZero) }
                else if self == Self::MIN && rhs == -1 { Err(PcuExecutionFaultKind::SignedDivisionOverflow) }
                else { Ok(()) }
            }
            fn pcu_checked_div_rem(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                let quotient = self.pcu_checked_div(rhs)?;
                Ok((quotient, self % rhs))
            }
            fn pcu_checked_div(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_div(rhs).ok_or(if rhs == 0 { PcuExecutionFaultKind::DivideByZero }
                    else { PcuExecutionFaultKind::SignedDivisionOverflow })
            }
            fn pcu_checked_rem(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_rem(rhs).ok_or(if rhs == 0 { PcuExecutionFaultKind::DivideByZero }
                    else { PcuExecutionFaultKind::SignedDivisionOverflow })
            }
        }
    )+};
}
unsigned!(u8, u16, u32, u64, u128);
signed!(i8, i16, i32, i64, i128);
macro_rules! wide {
    ($(($ty:ty, $min:expr, $limbs:literal)),+ $(,)?) => {$(
        impl PcuCheckedIntegerDivision for $ty {
            #[inline]
            fn pcu_div_rem_domain(self, rhs: Self) -> Result<(), PcuExecutionFaultKind> {
                if rhs == Self::ZERO {
                    return Err(PcuExecutionFaultKind::DivideByZero);
                }
                let minimum: Option<Self> = $min;
                if minimum == Some(self) && rhs == Self::from_limbs_le([u64::MAX; $limbs]) {
                    return Err(PcuExecutionFaultKind::SignedDivisionOverflow);
                }
                Ok(())
            }
            fn pcu_checked_div_rem(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                self.checked_div_rem(rhs)
            }
            fn pcu_checked_div(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_div_rem(rhs).map(|(quotient, _)| quotient)
            }
            fn pcu_checked_rem(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_div_rem(rhs).map(|(_, remainder)| remainder)
            }
        }
    )+};
}
wide!(
    (PcuU256, None, 4),
    (PcuI256, Some(PcuI256::MIN), 4),
    (PcuU512, None, 8),
    (PcuI512, Some(PcuI512::MIN), 8),
);

#[cfg(test)]
mod tests;
