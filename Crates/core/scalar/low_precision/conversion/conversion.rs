//! Exact-input narrowing into binary16/bfloat16/OFP8, without an intermediate float cast.
//!
//! IEEE 754-2019 5.4.2 specifies conversion; 4.3.1 supplies nearest/ties-even,
//! 7.4 overflow, and 7.5(a) destination-precision tininess with unbounded exponents.
//! BF16 uses the same declared PCU rounding/exception contract despite not being
//! an IEEE basic interchange format. Nonfinite rejection and observable clamp
//! errors are PCU departures. None of these references admits backend conversion.
//! Named OFP8 encodings use the same declared PCU rounding/fault rules;
//! E4M3FN's finite final exponent and reserved NaN are handled explicitly.
#[rustfmt::skip]
use super::{
    BF16,
    F16,
    E4M3FN,
    E5M2,
    Format,
    convert_error,
    pack,
};
#[rustfmt::skip]
use crate::{
    PcuBf16Bits,
    PcuClampedError,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
};

fn narrow_f64(
    bits: u64,
    format: Format,
    policy: PcuFloatUnderflowPolicy,
) -> Result<u16, PcuClampedError<u16>> {
    let exponent_field = ((bits >> 52) & 0x7ff).cast_signed();
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent_field == 0x7ff {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    let (significand, scale) = if exponent_field == 0 {
        (fraction, -1074)
    } else {
        (
            fraction | (1_u64 << 52),
            i32::try_from(exponent_field).expect("masked 11-bit exponent") - 1075,
        )
    };
    // The source significand is <=53 bits. pack rounds that exact dyadic once;
    // its normal quotient is <=12 bits, and very small shifts are bounded before use.
    pack(bits >> 63 != 0, significand, 1, scale, format, policy)
}

macro_rules! conversions {
    ($ty:ty, $format:ident) => {
        impl $ty {
            /// Narrows finite binary64 directly to destination precision, using default underflow policy.
            ///
            /// # Errors
            /// Rejects nonfinite input, overflow and tiny inexact results; ordinary inexact rounding succeeds.
            pub fn pcu_checked_from_f64(value: f64) -> Result<Self, PcuExecutionFaultKind> {
                Self::pcu_checked_from_f64_with_policy(value, PcuFloatUnderflowPolicy::default())
            }
            /// Narrows finite binary64 directly using the selected underflow policy.
            ///
            /// # Errors
            /// Rejects nonfinite input, overflow and policy-rejected underflow.
            pub fn pcu_checked_from_f64_with_policy(
                value: f64,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuExecutionFaultKind> {
                Self::pcu_clamped_from_f64_with_policy(value, policy).map_err(|fault| fault.kind())
            }
            /// Narrows binary64 retaining observable overflow or underflow recovery.
            ///
            /// # Errors
            /// Range faults retain signed maximum finite or the correctly rounded gradual result;
            /// nonfinite input is fatal with no payload.
            pub fn pcu_clamped_from_f64(value: f64) -> Result<Self, PcuClampedError<Self>> {
                Self::pcu_clamped_from_f64_with_policy(value, PcuFloatUnderflowPolicy::default())
            }
            /// Narrows binary64 retaining range recovery under the selected policy.
            ///
            /// # Errors
            /// Returns a recoverable range fault or fatal nonfinite input.
            pub fn pcu_clamped_from_f64_with_policy(
                value: f64,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuClampedError<Self>> {
                #[allow(clippy::cast_possible_truncation)]
                // Successful/recovered bits fit the selected destination encoding.
                let convert = |bits| Self::from_bits(bits as _);
                narrow_f64(value.to_bits(), $format, policy)
                    .map(convert)
                    .map_err(|fault| convert_error(fault, convert))
            }
            /// Narrows finite binary32 once, using the default underflow policy.
            ///
            /// # Errors
            /// Rejects nonfinite input, overflow and tiny inexact results.
            pub fn pcu_checked_from_f32(value: f32) -> Result<Self, PcuExecutionFaultKind> {
                Self::pcu_checked_from_f32_with_policy(value, PcuFloatUnderflowPolicy::default())
            }
            /// Narrows finite binary32 using the selected underflow policy.
            ///
            /// # Errors
            /// Rejects nonfinite input, overflow and policy-rejected underflow.
            pub fn pcu_checked_from_f32_with_policy(
                value: f32,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuExecutionFaultKind> {
                // Exact widening preserves every finite source bit before one destination rounding.
                Self::pcu_checked_from_f64_with_policy(crate::widen_f32_exact(value), policy)
            }
            /// Narrows binary32 retaining observable range recovery.
            ///
            /// # Errors
            /// Returns a recoverable range fault or fatal nonfinite input.
            pub fn pcu_clamped_from_f32(value: f32) -> Result<Self, PcuClampedError<Self>> {
                Self::pcu_clamped_from_f32_with_policy(value, PcuFloatUnderflowPolicy::default())
            }
            /// Narrows binary32 retaining range recovery under the selected policy.
            ///
            /// # Errors
            /// Returns a recoverable range fault or fatal nonfinite input.
            pub fn pcu_clamped_from_f32_with_policy(
                value: f32,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuClampedError<Self>> {
                Self::pcu_clamped_from_f64_with_policy(crate::widen_f32_exact(value), policy)
            }
            /// Widens a finite source exactly to binary32, preserving signed zero.
            ///
            /// # Errors
            /// Returns `InvalidFloatingOperand` for NaN or infinity.
            pub const fn pcu_checked_to_f32(self) -> Result<f32, PcuExecutionFaultKind> {
                let bits = self.to_bits() as u16;
                if bits & ($format.sign_mask - 1) > $format.max_finite {
                    return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
                }
                Ok(self.to_f32())
            }
            /// Widens a finite source exactly to binary64, preserving signed zero.
            ///
            /// # Errors
            /// Returns `InvalidFloatingOperand` for NaN or infinity.
            pub const fn pcu_checked_to_f64(self) -> Result<f64, PcuExecutionFaultKind> {
                match self.pcu_checked_to_f32() {
                    Ok(value) => Ok(crate::widen_f32_exact(value)),
                    Err(fault) => Err(fault),
                }
            }
        }
    };
}
conversions!(PcuF16Bits, F16);
conversions!(PcuBf16Bits, BF16);
conversions!(PcuF8E4M3FnBits, E4M3FN);
conversions!(PcuF8E5M2Bits, E5M2);

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
