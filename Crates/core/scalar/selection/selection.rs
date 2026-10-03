//! Exact finite encoding operations for half, FP8 and wide floating carriers.
//!
//! Negation changes one sign bit; `ReLU` and its derivative select existing bits.
//! No conversion, rounding, FTZ or narrower intermediate is introduced. IEEE
//! 754-2019 clause 7.5 governs exact subnormal classification where applicable;
//! rejecting nonfinite operands and `ReLU`'s positive-zero rule are PCU policies.
//! OFP8 formats use their named OCP encodings, not an invented IEEE FP8 format.
//! These references do not assert backend admission or general wide arithmetic.
#[rustfmt::skip]
use crate::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatClass,
    PcuFloatUnderflowClassification,
    PcuFloatUnderflowPolicy,
    PcuExecutionFaultKind,
    PcuClampedError,
    PcuClampedFault,
};

const fn classify_half(bits: u16, fraction_bits: u32, exponent_mask: u16) -> PcuFloatClass {
    let fraction = bits & ((1 << fraction_bits) - 1);
    let exponent = (bits >> fraction_bits) & exponent_mask;
    match (exponent, fraction) {
        (0, 0) => PcuFloatClass::Zero,
        (0, _) => PcuFloatClass::Subnormal,
        (exponent, 0) if exponent == exponent_mask => PcuFloatClass::Infinite,
        (exponent, _) if exponent == exponent_mask => {
            if fraction & (1 << (fraction_bits - 1)) == 0 {
                PcuFloatClass::SignalingNaN
            } else {
                PcuFloatClass::QuietNaN
            }
        }
        _ => PcuFloatClass::Normal,
    }
}
impl PcuF16Bits {
    /// Classifies representation without evaluating a host floating value.
    #[must_use]
    pub const fn classify(self) -> PcuFloatClass {
        classify_half(self.to_bits(), 10, 31)
    }
}
impl PcuBf16Bits {
    /// Classifies representation without evaluating a host floating value.
    #[must_use]
    pub const fn classify(self) -> PcuFloatClass {
        classify_half(self.to_bits(), 7, 255)
    }
}

trait Encoding: Copy {
    fn class(self) -> PcuFloatClass;
    fn negative(self) -> bool;
    fn negated(self) -> Self;
    fn zero() -> Self;
}
fn validate<T: Encoding>(value: T) -> Result<(), PcuExecutionFaultKind> {
    match value.class() {
        PcuFloatClass::Infinite | PcuFloatClass::QuietNaN | PcuFloatClass::SignalingNaN => {
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        }
        _ => Ok(()),
    }
}
fn publish<T: Encoding>(
    value: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, PcuClampedError<T>> {
    let subnormal = value.class() == PcuFloatClass::Subnormal;
    if policy.rejects(PcuFloatUnderflowClassification {
        is_tiny_after_rounding: subnormal,
        is_inexact: false,
        result_is_subnormal: subnormal,
    }) {
        Err(PcuClampedError::Range(PcuClampedFault::new(
            PcuExecutionFaultKind::ArithmeticUnderflow,
            value,
        )))
    } else {
        Ok(value)
    }
}
fn checked<T>(result: Result<T, PcuClampedError<T>>) -> Result<T, PcuExecutionFaultKind> {
    result.map_err(|fault| fault.kind())
}
fn neg<T: Encoding>(value: T, policy: PcuFloatUnderflowPolicy) -> Result<T, PcuClampedError<T>> {
    validate(value).map_err(PcuClampedError::Fatal)?;
    publish(value.negated(), policy)
}
fn relu<T: Encoding>(value: T, policy: PcuFloatUnderflowPolicy) -> Result<T, PcuClampedError<T>> {
    validate(value).map_err(PcuClampedError::Fatal)?;
    publish(
        if value.negative() || value.class() == PcuFloatClass::Zero {
            T::zero()
        } else {
            value
        },
        policy,
    )
}
fn backward<T: Encoding>(
    input: T,
    upstream: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, PcuExecutionFaultKind> {
    validate(input)?;
    validate(upstream)?;
    checked(publish(
        if input.negative() || input.class() == PcuFloatClass::Zero {
            T::zero()
        } else {
            upstream
        },
        policy,
    ))
}

macro_rules! narrow {
    ($ty:ty, $sign:expr) => {
        impl Encoding for $ty {
            fn class(self) -> PcuFloatClass {
                self.classify()
            }
            fn negative(self) -> bool {
                self.to_bits() & $sign != 0
            }
            fn negated(self) -> Self {
                Self::from_bits(self.to_bits() ^ $sign)
            }
            fn zero() -> Self {
                Self::from_bits(0)
            }
        }
    };
}
narrow!(PcuF16Bits, 0x8000);
narrow!(PcuBf16Bits, 0x8000);
narrow!(PcuF8E4M3FnBits, 0x80);
narrow!(PcuF8E5M2Bits, 0x80);
macro_rules! wide {
    ($ty:ty, $n:literal) => {
        impl Encoding for $ty {
            fn class(self) -> PcuFloatClass {
                self.classify()
            }
            fn negative(self) -> bool {
                self.to_limbs_le()[$n - 1] >> 63 != 0
            }
            fn negated(self) -> Self {
                let mut words = self.to_limbs_le();
                words[$n - 1] ^= 1 << 63;
                Self::from_limbs_le(words)
            }
            fn zero() -> Self {
                Self::from_limbs_le([0; $n])
            }
        }
    };
}
wide!(PcuF128Bits, 2);
wide!(PcuF256Bits, 4);

macro_rules! operations {
    ($($ty:ty),+ $(,)?) => {$(
        impl $ty {
            /// Exact sign inversion of finite stored bits with the default underflow policy.
            /// # Errors
            /// Returns `InvalidFloatingOperand` for infinity or NaN.
            pub fn pcu_checked_neg(self) -> Result<Self, PcuExecutionFaultKind> { self.pcu_checked_neg_with_policy(PcuFloatUnderflowPolicy::default()) }
            /// Exact sign inversion without changing subnormal representation.
            /// # Errors
            /// Returns invalid operand or policy-rejected subnormal result.
            pub fn pcu_checked_neg_with_policy(self, policy: PcuFloatUnderflowPolicy) -> Result<Self, PcuExecutionFaultKind> { checked(neg(self, policy)) }
            /// Selects this finite positive value, otherwise positive zero.
            /// # Errors
            /// Returns `InvalidFloatingOperand` for infinity or NaN.
            pub fn pcu_checked_relu(self) -> Result<Self, PcuExecutionFaultKind> { self.pcu_checked_relu_with_policy(PcuFloatUnderflowPolicy::default()) }
            /// Selects this finite positive value or +0 using an independent underflow policy.
            /// # Errors
            /// Returns invalid operand or policy-rejected selected subnormal result.
            pub fn pcu_checked_relu_with_policy(self, policy: PcuFloatUnderflowPolicy) -> Result<Self, PcuExecutionFaultKind> { checked(relu(self, policy)) }
            /// Selects upstream bits for strictly positive input; otherwise +0.
            /// Both operands must be finite even when the upstream value is masked out.
            /// # Errors
            /// Returns invalid operand or policy-rejected selected subnormal result.
            pub fn pcu_checked_relu_backward_with_policy(self, upstream: Self, policy: PcuFloatUnderflowPolicy) -> Result<Self, PcuExecutionFaultKind> { backward(self, upstream, policy) }
            /// Exact sign inversion retaining an observable policy-rejected subnormal payload.
            /// # Errors
            /// Returns fatal invalid operand or a range error containing the selected bits.
            pub fn pcu_clamped_neg_with_policy(self, policy: PcuFloatUnderflowPolicy) -> Result<Self, PcuClampedError<Self>> { neg(self, policy) }
            /// `ReLU` selection retaining an observable policy-rejected subnormal payload.
            /// # Errors
            /// Returns fatal invalid operand or a range error containing the selected bits.
            pub fn pcu_clamped_relu_with_policy(self, policy: PcuFloatUnderflowPolicy) -> Result<Self, PcuClampedError<Self>> { relu(self, policy) }
        }
    )+};
}
operations!(
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits
);

#[cfg(test)]
mod tests;
