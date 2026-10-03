//! Independent IEEE format packing oracle: direct integer rounding, no core conversion or host cast.
#[rustfmt::skip]
use pcu_facade::{PcuExecutionFaultKind as Kind,PcuFloatUnderflowPolicy as Policy};
pub fn narrow(bits: u64, policy: Policy) -> Result<(u32, Option<Kind>), Kind> {
    let sign = u32::try_from(bits >> 32).unwrap() & 0x8000_0000;
    let exponent = i32::try_from((bits >> 52) & 0x7ff).unwrap();
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if exponent == 0x7ff {
        return Err(Kind::InvalidFloatingOperand);
    }
    if exponent == 0 && fraction == 0 {
        return Ok((sign, None));
    }
    let unbiased = if exponent == 0 {
        -1022
    } else {
        exponent - 1023
    };
    let mantissa = if exponent == 0 {
        fraction
    } else {
        fraction | (1_u64 << 52)
    };
    if unbiased > 127 {
        return Ok((sign | 0x7f7f_ffff, Some(Kind::ArithmeticOverflow)));
    }
    let shift = u32::try_from(29 + (-126 - unbiased).max(0)).unwrap();
    // A binary64 significand has at most53 bits: larger shifts cannot reach a half-ULP.
    let (rounded, inexact) = if shift > 53 {
        (0, true)
    } else {
        let quotient = mantissa >> shift;
        let remainder = mantissa & ((1_u64 << shift) - 1);
        let half = 1_u64 << (shift - 1);
        (
            quotient + u64::from(remainder > half || (remainder == half && quotient & 1 != 0)),
            remainder != 0,
        )
    };
    let mut field = (unbiased + 127).max(0);
    let mut encoded = u32::try_from(rounded).unwrap();
    if field > 0 {
        if encoded >= 1 << 24 {
            field += 1;
            encoded >>= 1;
        }
        if field >= 255 {
            return Ok((sign | 0x7f7f_ffff, Some(Kind::ArithmeticOverflow)));
        }
        encoded = (u32::try_from(field).unwrap() << 23) | (encoded & 0x7f_ffff);
    }
    // Independently characterize the nearest-24-bit carry threshold. For exponent
    // -127 only, significand >=2^53-2^28 rounds to exponent -126 before any
    // constrained subnormal packing. Stored normal bits alone do not prove non-tininess.
    let tiny_after_precision =
        unbiased < -126 && !(unbiased == -127 && mantissa >= (1_u64 << 53) - (1_u64 << 28));
    let stored_subnormal = encoded != 0 && encoded & 0x7f80_0000 == 0;
    let rejected = match policy {
        Policy::IeeeAfterRounding => tiny_after_precision && inexact,
        Policy::RejectSubnormalResult => stored_subnormal || (tiny_after_precision && inexact),
        Policy::AllowGradualUnderflow => false,
    };
    Ok((
        sign | encoded,
        rejected.then_some(Kind::ArithmeticUnderflow),
    ))
}
pub fn widen(bits: u32) -> Result<u64, Kind> {
    let sign = u64::from(bits & 0x8000_0000) << 32;
    let exponent = (bits >> 23) & 255;
    let fraction = bits & 0x7f_ffff;
    if exponent == 255 {
        return Err(Kind::InvalidFloatingOperand);
    }
    if exponent == 0 {
        if fraction == 0 {
            return Ok(sign);
        }
        let leading = fraction.ilog2();
        let exponent = 874 + leading;
        Ok(sign
            | (u64::from(exponent) << 52)
            | (u64::from(fraction - (1 << leading)) << (52 - leading)))
    } else {
        Ok(sign | (u64::from(exponent + 896) << 52) | (u64::from(fraction) << 29))
    }
}
