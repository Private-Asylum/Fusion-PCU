//! Checked floating-point conversions with integer-only classification.
//!
//! IEEE Std 754-2019 references: format conversion (5.4.2), nearest ties-to-even
//! rounding (4.3.1/4.3.3), overflow (7.4), and tininess after rounding (7.5(a)).
//! Finite widening is exact; narrowing rounds at the destination precision and classifies
//! inexactness including final exponent-limited packing. Non-finite input rejection and
//! `Result` errors are PCU policies, not IEEE default exception handling.

#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuClampedError,
    PcuClampedFault,
    PcuFloatUnderflowClassification,
    PcuFloatUnderflowPolicy,
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f64 {}

    pub trait WidenSealed {}
    impl WidenSealed for f32 {}
}

/// Checked conversions from binary64 to binary32.
///
/// Finite inexact values are rounded to nearest, ties to even and are accepted. Non-finite
/// inputs and results that overflow binary32 are faults. Underflow uses the supplied policy.
#[allow(private_bounds)] // Downstream implementations must not bypass the integer contract.
pub trait PcuCheckedFloatConversion: sealed::Sealed + Copy {
    /// Converts to binary32 using round-to-nearest, ties-to-even and the default underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_to_f32(self) -> Result<f32, PcuExecutionFaultKind>;

    /// Converts to binary32 using round-to-nearest, ties-to-even and an explicit underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_to_f32_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<f32, PcuExecutionFaultKind>;
}

/// Explicit range recovery when narrowing finite binary64 to binary32.
///
/// Range faults remain `Err` and retain a rounded gradual-underflow result or a
/// sign-preserving maximum-finite overflow result. Non-finite inputs remain fatal.
/// This is PCU alternative handling, not IEEE default overflow handling (7.4).
#[allow(private_bounds)] // Keep recovery tied to the sealed conversion implementation.
pub trait PcuClampedFloatConversion: sealed::Sealed + Copy {
    /// Narrows with observable range recovery and the default underflow classification.
    ///
    /// # Errors
    /// Returns a recoverable range fault or a fatal non-finite-input fault.
    fn pcu_clamped_to_f32(self) -> Result<f32, PcuClampedError<f32>>;

    /// Narrows with observable range recovery and the specified underflow classification.
    ///
    /// # Errors
    /// Returns a recoverable range fault or a fatal non-finite-input fault.
    fn pcu_clamped_to_f32_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<f32, PcuClampedError<f32>>;
}

/// Checked exact widening from finite binary32 to binary64.
#[allow(private_bounds)] // Keep checked conversion semantics sealed to the core implementations.
pub trait PcuCheckedFloatWidening: sealed::WidenSealed + Copy {
    /// Widens a finite binary32 value exactly, rejecting NaN and infinity.
    ///
    /// # Errors
    /// Returns `InvalidFloatingOperand` for non-finite inputs.
    fn pcu_checked_to_f64(self) -> Result<f64, PcuExecutionFaultKind>;
}

fn widen_f32(bits: u32) -> Result<f64, PcuExecutionFaultKind> {
    let sign = u64::from(bits & 0x8000_0000) << 32;
    let exponent_field = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent_field == 0xff {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    if exponent_field == 0 && fraction == 0 {
        return Ok(f64::from_bits(sign));
    }
    if exponent_field == 0 {
        let top_bit = fraction.ilog2();
        let exponent = -149 + top_bit.cast_signed();
        let normalized_fraction = u64::from(fraction) << (52 - top_bit);
        let exp_field = (exponent + 1023).cast_unsigned();
        return Ok(f64::from_bits(
            sign | (u64::from(exp_field) << 52) | (normalized_fraction & 0x000f_ffff_ffff_ffff),
        ));
    }
    let exponent = exponent_field.cast_signed() - 127;
    let exp_field = (exponent + 1023).cast_unsigned();
    Ok(f64::from_bits(
        sign | (u64::from(exp_field) << 52) | (u64::from(fraction) << 29),
    ))
}

impl PcuCheckedFloatWidening for f32 {
    fn pcu_checked_to_f64(self) -> Result<f64, PcuExecutionFaultKind> {
        widen_f32(self.to_bits())
    }
}

// IEEE Std 754-2019, clause 4.3.1: retain parity for an exact halfway case.
fn round_right_even(value: u64, shift: u32) -> (u64, bool) {
    if shift == 0 {
        return (value, false);
    }
    if shift > 64 {
        return (0, value != 0);
    }
    let quotient = if shift == 64 { 0 } else { value >> shift };
    let mask = if shift == 64 {
        u64::MAX
    } else {
        (1_u64 << shift) - 1
    };
    let remainder = value & mask;
    let halfway = 1_u64 << (shift - 1);
    let increment = remainder > halfway || (remainder == halfway && quotient & 1 != 0);
    (quotient + u64::from(increment), remainder != 0)
}

#[allow(clippy::cast_possible_truncation)] // Values are masked or range-checked to binary32 width.
fn convert<const CLAMP: bool>(
    bits: u64,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let sign = ((bits >> 32) as u32) & 0x8000_0000;
    let exponent_field = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent_field == 0x7ff {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    if exponent_field == 0 && fraction == 0 {
        return Ok(f32::from_bits(sign));
    }

    let (significand, exponent) = if exponent_field == 0 {
        (fraction, -1022)
    } else {
        (fraction | (1_u64 << 52), exponent_field - 1023)
    };
    let top_bit = significand.ilog2();
    let unbiased = exponent - 52 + top_bit.cast_signed();

    // IEEE Std 754-2019, clause 7.5(a): classify tininess after rounding to 24
    // significant bits with an unbounded exponent, independently of stored bits.
    let precision_shift = top_bit.saturating_sub(23);
    let (mut rounded, _) = round_right_even(significand, precision_shift);
    let mut rounded_exponent = unbiased;
    if rounded == (1_u64 << 24) {
        rounded >>= 1;
        rounded_exponent += 1;
    }
    let tiny_after_rounding = rounded_exponent < -126;

    let (magnitude, inexact, subnormal) = if unbiased >= -126 {
        if rounded_exponent > 127 {
            return Err(if CLAMP {
                PcuClampedError::Range(PcuClampedFault::new(
                    PcuExecutionFaultKind::ArithmeticOverflow,
                    f32::from_bits(sign | 0x7f7f_ffff),
                ))
            } else {
                PcuClampedError::Fatal(PcuExecutionFaultKind::ArithmeticOverflow)
            });
        }
        let exp_field = (rounded_exponent + 127).cast_unsigned();
        (
            (exp_field << 23) | ((rounded as u32) & 0x007f_ffff),
            round_right_even(significand, precision_shift).1,
            false,
        )
    } else {
        // Subnormals are integer multiples of 2^-149.
        let shift = (-(exponent + 97)).max(0).cast_unsigned();
        let (sub_sig, discarded) = round_right_even(significand, shift);
        if sub_sig >= (1_u64 << 23) {
            (0x0080_0000, discarded, false)
        } else {
            (sub_sig as u32, discarded, sub_sig != 0)
        }
    };
    let classification = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: tiny_after_rounding,
        is_inexact: inexact,
        result_is_subnormal: subnormal,
    };
    if policy.rejects(classification) {
        return Err(if CLAMP {
            PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                f32::from_bits(sign | magnitude),
            ))
        } else {
            PcuClampedError::Fatal(PcuExecutionFaultKind::ArithmeticUnderflow)
        });
    }
    Ok(f32::from_bits(sign | magnitude))
}

impl PcuCheckedFloatConversion for f64 {
    fn pcu_checked_to_f32(self) -> Result<f32, PcuExecutionFaultKind> {
        self.pcu_checked_to_f32_with_policy(PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_to_f32_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<f32, PcuExecutionFaultKind> {
        convert::<false>(self.to_bits(), policy).map_err(|fault| fault.kind())
    }
}

#[cfg(test)]
mod tests;

impl PcuClampedFloatConversion for f64 {
    fn pcu_clamped_to_f32(self) -> Result<f32, PcuClampedError<f32>> {
        self.pcu_clamped_to_f32_with_policy(PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_to_f32_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<f32, PcuClampedError<f32>> {
        convert::<true>(self.to_bits(), policy)
    }
}
