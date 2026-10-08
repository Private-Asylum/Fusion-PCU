//! Checked binary64 arithmetic implemented with integer significands.
//!
//! IEEE Std 754-2019 references: nearest ties-to-even (4.3.1), overflow (7.4),
//! and tininess-after-rounding with final-result inexactness (7.5(a)). Error delivery
//! and the finite-input restriction are PCU policies; see the parent module.

use super::{PcuCheckedFloat, PcuClampedFloat};

#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuFloatUnderflowClassification,
    PcuFloatUnderflowPolicy,
    PcuClampedError,
    PcuClampedFault,
};

const SIGN_MASK: u64 = 1 << 63;
const FRACTION_MASK: u64 = (1 << 52) - 1;
const HIDDEN_BIT: u64 = 1 << 52;
const NORMALIZED_EXTENT: u32 = 55;
const NORMALIZED_LIMIT: u64 = 1 << (NORMALIZED_EXTENT + 1);
const MIN_NORMAL_EXPONENT: i32 = -1022;
const MAX_NORMAL_EXPONENT: i32 = 1023;

fn neg<const CLAMP: bool>(
    value: f64,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    let bits = value.to_bits();
    if bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    let result_bits = bits ^ SIGN_MASK;
    let subnormal = result_bits & !SIGN_MASK != 0 && result_bits & 0x7ff0_0000_0000_0000 == 0;
    let class = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: subnormal,
        is_inexact: false,
        result_is_subnormal: subnormal,
    };
    let result = f64::from_bits(result_bits);
    if policy.rejects(class) {
        if CLAMP {
            Err(PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                result,
            )))
        } else {
            Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ))
        }
    } else {
        Ok(result)
    }
}

fn relu<const CLAMP: bool>(
    value: f64,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    let bits = value.to_bits();
    if bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    let result_bits = if bits & SIGN_MASK != 0 || bits & !SIGN_MASK == 0 {
        0
    } else {
        bits
    };
    let subnormal = result_bits != 0 && result_bits & 0x7ff0_0000_0000_0000 == 0;
    let class = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: subnormal,
        is_inexact: false,
        result_is_subnormal: subnormal,
    };
    let result = f64::from_bits(result_bits);
    if policy.rejects(class) {
        if CLAMP {
            Err(PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                result,
            )))
        } else {
            Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ))
        }
    } else {
        Ok(result)
    }
}

#[derive(Clone, Copy)]
struct Parts {
    sign: u32,
    exponent: i32,
    significand: u64,
}

fn decode(bits: u64) -> Result<Option<Parts>, PcuExecutionFaultKind> {
    let exponent_field = (bits >> 52) & 0x7ff;
    let fraction = bits & FRACTION_MASK;
    if exponent_field == 0x7ff {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    if exponent_field == 0 && fraction == 0 {
        return Ok(None);
    }
    let (exponent, significand) = if exponent_field == 0 {
        (MIN_NORMAL_EXPONENT, fraction)
    } else {
        (
            i32::try_from(exponent_field).expect("binary64 exponent fits i32") - 1023,
            fraction | HIDDEN_BIT,
        )
    };
    Ok(Some(Parts {
        sign: u32::try_from(bits >> 63).expect("sign bit fits u32"),
        exponent,
        significand,
    }))
}

/// Right-shift while retaining whether any discarded bit was nonzero.
fn shift_right_jam(value: u64, distance: u32) -> u64 {
    if distance == 0 {
        value
    } else if distance < u64::BITS {
        (value >> distance) | u64::from(value & ((1_u64 << distance) - 1) != 0)
    } else {
        u64::from(value != 0)
    }
}

fn shift_right_jam_u128(value: u128, distance: u32) -> u64 {
    if distance == 0 {
        u64::try_from(value).expect("normalized product fits u64")
    } else if distance < u128::BITS {
        let retained = value >> distance;
        let discarded_mask = (1_u128 << distance) - 1;
        u64::try_from(retained | u128::from(value & discarded_mask != 0))
            .expect("normalized product fits u64")
    } else {
        u64::from(value != 0)
    }
}

// IEEE Std 754-2019, clause 4.3.1: ties select the even significand.
const fn round_increment(significand: u64, discarded: u64) -> bool {
    discarded > 4 || (discarded == 4 && significand & 1 != 0)
}

// Called only by pack, which already established the normalized significand range.
// Unbounded-exponent rounding for IEEE 754-2019 clause 7.5(a) needs no second normalization.
fn round_to_precision(ext: u64, mut exponent: i32) -> (u64, i32, bool) {
    debug_assert!((1_u64 << NORMALIZED_EXTENT..NORMALIZED_LIMIT).contains(&ext));
    let discarded = ext & 7;
    let mut significand = ext >> 3;
    let inexact = discarded != 0;
    if round_increment(significand, discarded) {
        significand += 1;
    }
    if significand == 1_u64 << 53 {
        significand >>= 1;
        exponent += 1;
    }
    (significand, exponent, inexact)
}

fn pack<const CLAMP: bool>(
    sign: u32,
    mut ext: u64,
    mut exponent: i32,
) -> Result<(f64, PcuFloatUnderflowClassification), PcuClampedError<f64>> {
    while ext >= NORMALIZED_LIMIT {
        ext = shift_right_jam(ext, 1);
        exponent += 1;
    }
    while ext < 1_u64 << NORMALIZED_EXTENT {
        ext <<= 1;
        exponent -= 1;
    }

    // IEEE Std 754-2019, clause 7.5(a): destination precision, unbounded exponent.
    let (_, rounded_exponent, _) = round_to_precision(ext, exponent);
    let tiny_after_rounding = rounded_exponent < MIN_NORMAL_EXPONENT;
    if exponent < MIN_NORMAL_EXPONENT {
        ext = shift_right_jam(ext, (MIN_NORMAL_EXPONENT - exponent).cast_unsigned());
        exponent = MIN_NORMAL_EXPONENT;
    }

    let discarded = ext & 7;
    let mut significand = ext >> 3;
    let inexact = discarded != 0;
    if round_increment(significand, discarded) {
        significand += 1;
    }

    let sign_bits = u64::from(sign) << 63;
    let (bits, result_is_subnormal) = if exponent == MIN_NORMAL_EXPONENT && significand < HIDDEN_BIT
    {
        (sign_bits | significand, significand != 0)
    } else {
        if significand >= 1_u64 << 53 {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > MAX_NORMAL_EXPONENT {
            if CLAMP {
                return Err(PcuClampedError::Range(PcuClampedFault::new(
                    PcuExecutionFaultKind::ArithmeticOverflow,
                    f64::from_bits(sign_bits | 0x7fef_ffff_ffff_ffff),
                )));
            }
            return Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticOverflow,
            ));
        }
        let exponent_field = u64::try_from(exponent + 1023)
            .expect("nonnegative normal exponent fits binary64 field");
        (
            sign_bits | (exponent_field << 52) | (significand & FRACTION_MASK),
            false,
        )
    };

    Ok((
        f64::from_bits(bits),
        PcuFloatUnderflowClassification {
            is_tiny_after_rounding: tiny_after_rounding,
            is_inexact: inexact,
            result_is_subnormal,
        },
    ))
}

fn passthrough<const CLAMP: bool>(
    value: f64,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    let magnitude = value.to_bits() & !SIGN_MASK;
    let subnormal = magnitude != 0 && magnitude < (1_u64 << 52);
    let classification = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: subnormal,
        is_inexact: false,
        result_is_subnormal: subnormal,
    };
    if policy.rejects(classification) {
        if CLAMP {
            Err(PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                value,
            )))
        } else {
            Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ))
        }
    } else {
        Ok(value)
    }
}

fn add<const CLAMP: bool>(
    left: f64,
    right: f64,
    subtract: bool,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    let a = decode(left.to_bits()).map_err(PcuClampedError::Fatal)?;
    let mut b = decode(right.to_bits()).map_err(PcuClampedError::Fatal)?;
    if let Some(parts) = &mut b {
        parts.sign ^= u32::from(subtract);
    }

    let Some(mut a) = a else {
        if let Some(b) = b {
            let value = f64::from_bits((u64::from(b.sign) << 63) | (right.to_bits() & !SIGN_MASK));
            return passthrough::<CLAMP>(value, policy);
        }
        let left_sign = u32::try_from(left.to_bits() >> 63).expect("sign bit fits u32");
        let right_sign = u32::try_from((right.to_bits() >> 63) ^ u64::from(subtract))
            .expect("sign bit fits u32");
        return Ok(f64::from_bits(
            u64::from(left_sign == right_sign) * (u64::from(left_sign) << 63),
        ));
    };
    let Some(mut b) = b else {
        return passthrough::<CLAMP>(left, policy);
    };
    if a.exponent < b.exponent {
        core::mem::swap(&mut a, &mut b);
    }
    let distance = (a.exponent - b.exponent).cast_unsigned();
    let a_ext = a.significand << 3;
    let b_ext = shift_right_jam(b.significand << 3, distance);
    let (result_sign, ext) = if a.sign == b.sign {
        (a.sign, a_ext + b_ext)
    } else if a_ext >= b_ext {
        (a.sign, a_ext - b_ext)
    } else {
        (b.sign, b_ext - a_ext)
    };
    if ext == 0 {
        return Ok(0.0);
    }
    let (result, classification) = pack::<CLAMP>(result_sign, ext, a.exponent)?;
    if policy.rejects(classification) {
        if CLAMP {
            Err(PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                result,
            )))
        } else {
            Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ))
        }
    } else {
        Ok(result)
    }
}

fn multiply<const CLAMP: bool>(
    left: f64,
    right: f64,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    let a = decode(left.to_bits())
        .map_err(PcuClampedError::Fatal)?
        .unwrap_or_else(|| Parts {
            sign: u32::try_from(left.to_bits() >> 63).expect("sign bit fits u32"),
            exponent: MIN_NORMAL_EXPONENT,
            significand: 0,
        });
    let b = decode(right.to_bits())
        .map_err(PcuClampedError::Fatal)?
        .unwrap_or_else(|| Parts {
            sign: u32::try_from(right.to_bits() >> 63).expect("sign bit fits u32"),
            exponent: MIN_NORMAL_EXPONENT,
            significand: 0,
        });
    if a.significand == 0 || b.significand == 0 {
        return Ok(f64::from_bits(u64::from(a.sign ^ b.sign) << 63));
    }
    let product = u128::from(a.significand) * u128::from(b.significand);
    let top = u128::BITS - 1 - product.leading_zeros();
    let exponent =
        a.exponent + b.exponent + i32::try_from(top).expect("product top fits i32") - 104;
    let ext = if top > NORMALIZED_EXTENT {
        shift_right_jam_u128(product, top - NORMALIZED_EXTENT)
    } else {
        u64::try_from(product << (NORMALIZED_EXTENT - top)).expect("normalized product fits u64")
    };
    let (result, classification) = pack::<CLAMP>(a.sign ^ b.sign, ext, exponent)?;
    if policy.rejects(classification) {
        if CLAMP {
            Err(PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                result,
            )))
        } else {
            Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ))
        }
    } else {
        Ok(result)
    }
}

// The denominator is already proved nonzero; retain the quotient/remainder pair
// that classifies rounding inexactness rather than adding a separate divisibility API.
#[allow(clippy::manual_is_multiple_of)]
fn divide<const CLAMP: bool>(
    left: f64,
    right: f64,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f64, PcuClampedError<f64>> {
    let a = decode(left.to_bits()).map_err(PcuClampedError::Fatal)?;
    let b = decode(right.to_bits()).map_err(PcuClampedError::Fatal)?;
    let sign = u32::try_from((left.to_bits() ^ right.to_bits()) >> 63).expect("sign bit fits u32");
    let Some(mut b) = b else {
        return Err(PcuClampedError::Fatal(PcuExecutionFaultKind::DivideByZero));
    };
    let Some(mut a) = a else {
        return Ok(f64::from_bits(u64::from(sign) << 63));
    };
    let a_shift = 52 - a.significand.ilog2();
    a.significand <<= a_shift;
    a.exponent -= i32::try_from(a_shift).expect("normalization shift fits i32");
    let b_shift = 52 - b.significand.ilog2();
    b.significand <<= b_shift;
    b.exponent -= i32::try_from(b_shift).expect("normalization shift fits i32");

    let mut numerator = u128::from(a.significand);
    let denominator = u128::from(b.significand);
    let mut exponent = a.exponent - b.exponent;
    if numerator < denominator {
        numerator <<= 1;
        exponent -= 1;
    }
    let scaled = numerator << 55;
    let mut ext = u64::try_from(scaled / denominator).expect("quotient fits binary64 ext");
    if scaled % denominator != 0 {
        ext |= 1;
    }
    let (result, classification) = pack::<CLAMP>(sign, ext, exponent)?;
    if policy.rejects(classification) {
        if CLAMP {
            Err(PcuClampedError::Range(PcuClampedFault::new(
                PcuExecutionFaultKind::ArithmeticUnderflow,
                result,
            )))
        } else {
            Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ))
        }
    } else {
        Ok(result)
    }
}

impl PcuCheckedFloat for f64 {
    fn pcu_checked_relu_backward_with_policy(
        self,
        upstream: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        if !self.is_finite() || !upstream.is_finite() {
            return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
        }
        // IEEE 754 binary64 selection is exact. Classify the encoding so a
        // caller's flush-input mode cannot turn a positive subnormal into zero.
        let bits = self.to_bits();
        let positive = bits & 0x8000_0000_0000_0000 == 0 && bits & 0x7fff_ffff_ffff_ffff != 0;
        let result = if positive { upstream } else { 0.0 };
        result
            .pcu_checked_neg_with_policy(policy)?
            .pcu_checked_neg()
    }
    fn pcu_checked_neg(self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_neg_with_policy(PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_neg_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        neg::<false>(self, policy).map_err(|fault| fault.kind())
    }

    fn pcu_checked_relu(self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_relu_with_policy(PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_relu_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        relu::<false>(self, policy).map_err(|fault| fault.kind())
    }
    fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_add_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_sub_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_mul_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_div(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_div_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_checked_add_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        add::<false>(self, rhs, false, policy).map_err(|fault| fault.kind())
    }

    fn pcu_checked_sub_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        add::<false>(self, rhs, true, policy).map_err(|fault| fault.kind())
    }

    fn pcu_checked_mul_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        multiply::<false>(self, rhs, policy).map_err(|fault| fault.kind())
    }

    fn pcu_checked_div_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        divide::<false>(self, rhs, policy).map_err(|fault| fault.kind())
    }
}

impl PcuClampedFloat for f64 {
    fn pcu_clamped_neg(self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_neg_with_policy(PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_neg_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        neg::<true>(self, policy)
    }

    fn pcu_clamped_relu(self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_relu_with_policy(PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_relu_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        relu::<true>(self, policy)
    }
    fn pcu_clamped_add(self, rhs: Self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_add_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_sub(self, rhs: Self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_sub_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_mul(self, rhs: Self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_mul_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_div(self, rhs: Self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_div_with_policy(rhs, PcuFloatUnderflowPolicy::default())
    }

    fn pcu_clamped_add_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        add::<true>(self, rhs, false, policy)
    }

    fn pcu_clamped_sub_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        add::<true>(self, rhs, true, policy)
    }

    fn pcu_clamped_mul_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        multiply::<true>(self, rhs, policy)
    }

    fn pcu_clamped_div_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        divide::<true>(self, rhs, policy)
    }
}

#[cfg(test)]
#[path = "f64_tests.rs"]
mod tests;
