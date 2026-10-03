//! Deterministic checked integer arithmetic, independent of Rust build profiles.
//!
//! Below-minimum and above-maximum results are distinct faults. No invalid native
//! arithmetic is executed, and no wrapping alternative is selected implicitly.

#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuScalar,
};

pub(crate) mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
    impl Sealed for u128 {}
    impl Sealed for i128 {}
}

/// Checked Add/Sub/Mul semantics for sealed native and limb-based PCU integers.
///
/// Implementations are sealed. Underflow means an exact integer result below
/// the representable minimum, including negative results for unsigned scalars.
/// Every supported representation also provides explicit saturated recovery through
/// [`crate::PcuClampedInteger`]. Checked methods here remain rejecting; a device instruction
/// must separately carry and admit its range policy before recovery is enabled.
pub trait PcuCheckedInteger: PcuScalar + crate::PcuClampedInteger + sealed::Sealed + Copy {
    /// Adds without wrapping.
    ///
    /// # Errors
    /// Returns underflow or overflow if the exact sum is outside the scalar's range.
    fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Subtracts without wrapping.
    ///
    /// # Errors
    /// Returns underflow or overflow if the exact difference is outside the scalar's range.
    fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Multiplies without wrapping.
    ///
    /// # Errors
    /// Returns underflow or overflow if the exact product is outside the scalar's range.
    fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
}

macro_rules! unsigned_integer {
    ($($ty:ty),+ $(,)?) => {$(
        impl PcuCheckedInteger for $ty {
            fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_add(rhs).ok_or(PcuExecutionFaultKind::ArithmeticOverflow)
            }
            fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_sub(rhs).ok_or(PcuExecutionFaultKind::ArithmeticUnderflow)
            }
            fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_mul(rhs).ok_or(PcuExecutionFaultKind::ArithmeticOverflow)
            }
        }
    )+};
}

macro_rules! signed_integer {
    ($($ty:ty),+ $(,)?) => {$(
        impl PcuCheckedInteger for $ty {
            fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_add(rhs).ok_or(if rhs < 0 {
                    PcuExecutionFaultKind::ArithmeticUnderflow
                } else {
                    PcuExecutionFaultKind::ArithmeticOverflow
                })
            }
            fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_sub(rhs).ok_or(if rhs > 0 {
                    PcuExecutionFaultKind::ArithmeticUnderflow
                } else {
                    PcuExecutionFaultKind::ArithmeticOverflow
                })
            }
            fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.checked_mul(rhs).ok_or(if (self < 0) != (rhs < 0) {
                    PcuExecutionFaultKind::ArithmeticUnderflow
                } else {
                    PcuExecutionFaultKind::ArithmeticOverflow
                })
            }
        }
    )+};
}

unsigned_integer!(u8, u16, u32, u64, u128);
signed_integer!(i8, i16, i32, i64, i128);

#[cfg(test)]
mod tests;
