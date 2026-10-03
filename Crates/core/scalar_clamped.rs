//! Checked scalar operations with explicit recovery values for supported range faults.
//!
//! Unlike saturating arithmetic, these operations report whether recovery was required. Integer
//! operations recover to a saturated endpoint; floating conversions may recover to a gradual
//! tiny result. The value is available by consuming the fault, so callers do not need a
//! `Clone`/`Copy` bound on the wrapper.
//!
//! ```
//! use fusion_pcu_core::{PcuClampedInteger, PcuExecutionFaultKind};
//!
//! let value = u8::MAX.pcu_clamped_add(1).unwrap_or_else(|fault| {
//!     assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
//!     fault.clamped_value()
//! });
//! assert_eq!(value, u8::MAX);
//! ```
//!
//! These explicit host scalar operations do not themselves enable a device kernel flag.

#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuScalar,
};

/// Policy for recovering from checked integer or floating result-range faults.
///
/// This is independent of [`crate::PcuFloatUnderflowPolicy`]: it controls recovery for either
/// overflow or underflow. Integer recovery saturates to the nearest representable endpoint;
/// the floating underflow policy independently classifies tiny results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PcuRangePolicy {
    /// Report a range fault and reject the operation result.
    #[default]
    Reject,
    /// Report a range fault while recovering to a defined boundary or gradual result.
    Clamp,
}

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

/// A range fault and its defined recovery value.
///
/// Fields are private so core implementations pair only meaningful faults and recovery values.
#[derive(Debug, PartialEq, Eq)]
pub struct PcuClampedFault<T> {
    kind: PcuExecutionFaultKind,
    clamped_value: T,
}

impl<T> PcuClampedFault<T> {
    pub(crate) const fn new(kind: PcuExecutionFaultKind, clamped_value: T) -> Self {
        Self {
            kind,
            clamped_value,
        }
    }

    /// Classification of the exact result's range violation.
    #[must_use]
    pub const fn kind(&self) -> PcuExecutionFaultKind {
        self.kind
    }

    /// Returns the defined recovery value, consuming the fault wrapper.
    /// Integer recovery uses a saturated endpoint; floating underflow can retain
    /// the correctly rounded gradual result, including signed zero.
    #[must_use]
    pub fn clamped_value(self) -> T {
        self.clamped_value
    }
}

/// Error returned by clamped scalar adapters when they can distinguish range recovery from a
/// fatal arithmetic condition.
#[derive(Debug, PartialEq, Eq)]
pub enum PcuClampedError<T> {
    /// The operation violated its range policy and has a defined recovery value.
    Range(PcuClampedFault<T>),
    /// The operation failed without a meaningful clamped scalar value.
    Fatal(PcuExecutionFaultKind),
}

impl<T> PcuClampedError<T> {
    /// Classification of either the recoverable range fault or fatal arithmetic condition.
    #[must_use]
    pub const fn kind(&self) -> PcuExecutionFaultKind {
        match self {
            Self::Range(fault) => fault.kind(),
            Self::Fatal(kind) => *kind,
        }
    }
}

/// Checked Add/Sub/Mul operations that return a saturated endpoint on range faults.
///
/// An in-range exact result is returned as `Ok`. An out-of-range result is always `Err`, with
/// [`PcuClampedFault::kind`] preserving its classification and
/// [`PcuClampedFault::clamped_value`] yielding the nearest representable integer endpoint.
#[allow(private_bounds)] // The sealed set is the contract boundary for supported integer types.
pub trait PcuClampedInteger: PcuScalar + sealed::Sealed + Copy {
    /// Adds, returning the maximum endpoint on overflow or minimum endpoint on underflow.
    ///
    /// # Errors
    /// Returns the range-fault classification and saturated endpoint when the exact sum is
    /// outside the scalar's range.
    fn pcu_clamped_add(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>>;
    /// Subtracts, returning the endpoint in the direction of the range fault.
    ///
    /// # Errors
    /// Returns the range-fault classification and saturated endpoint when the exact difference
    /// is outside the scalar's range.
    fn pcu_clamped_sub(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>>;
    /// Multiplies, returning the endpoint in the direction of the range fault.
    ///
    /// # Errors
    /// Returns the range-fault classification and saturated endpoint when the exact product is
    /// outside the scalar's range.
    fn pcu_clamped_mul(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>>;
}

macro_rules! unsigned_integer {
    ($($ty:ty),+ $(,)?) => {$ (
        impl PcuClampedInteger for $ty {
            fn pcu_clamped_add(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>> {
                self.checked_add(rhs).ok_or_else(|| {
                    PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Self::MAX)
                })
            }

            fn pcu_clamped_sub(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>> {
                self.checked_sub(rhs).ok_or_else(|| {
                    PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticUnderflow, Self::MIN)
                })
            }

            fn pcu_clamped_mul(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>> {
                self.checked_mul(rhs).ok_or_else(|| {
                    PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Self::MAX)
                })
            }
        }
    )+};
}

macro_rules! signed_integer {
    ($($ty:ty),+ $(,)?) => {$ (
        impl PcuClampedInteger for $ty {
            fn pcu_clamped_add(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>> {
                self.checked_add(rhs).ok_or_else(|| {
                    if rhs < 0 {
                        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticUnderflow, Self::MIN)
                    } else {
                        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Self::MAX)
                    }
                })
            }

            fn pcu_clamped_sub(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>> {
                self.checked_sub(rhs).ok_or_else(|| {
                    if rhs > 0 {
                        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticUnderflow, Self::MIN)
                    } else {
                        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Self::MAX)
                    }
                })
            }

            fn pcu_clamped_mul(self, rhs: Self) -> Result<Self, PcuClampedFault<Self>> {
                self.checked_mul(rhs).ok_or_else(|| {
                    if (self < 0) != (rhs < 0) {
                        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticUnderflow, Self::MIN)
                    } else {
                        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Self::MAX)
                    }
                })
            }
        }
    )+};
}

unsigned_integer!(u8, u16, u32, u64, u128);
signed_integer!(i8, i16, i32, i64, i128);

#[cfg(test)]
mod tests;
