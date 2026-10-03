//! Bounded binary16/bfloat16 and named OFP8 finite-input arithmetic reference.
//!
//! IEEE 754-2019 4.3.1 supplies nearest/ties-even, 7.4 overflow and 7.5(a)
//! destination-precision tininess with unbounded exponents. Packing is rounded
//! directly from the exact rational, never from a rounded host float. Rejecting
//! nonfinite inputs and reporting exceptions as errors are PCU departures.
//! See the [IEEE working-group exception notes](https://grouper.ieee.org/groups/msc/ANSI_IEEE-Std-754-2019/background/exceptions.txt).
//! This allocation-free reference makes no backend-admission or full-IEEE claim.
//! OFP8 E4M3FN/E5M2 use their OCP encodings with explicitly specified PCU
//! nearest/even rounding and observable range faults. E4M3FN's final exponent
//! contains finite values, and its reserved NaN must never become a finite result.
#[rustfmt::skip]
use crate::{
    PcuBf16Bits,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuExecutionFaultKind,
    PcuFloatUnderflowClassification,
    PcuFloatUnderflowPolicy,
    PcuClampedError,
    PcuClampedFault,
};

#[derive(Clone, Copy)]
struct Format {
    fraction: u32,
    bias: i32,
    exponent_mask: u16,
    sign_mask: u16,
    max_finite: u16,
}
const F16: Format = Format {
    fraction: 10,
    bias: 15,
    exponent_mask: 31,
    sign_mask: 0x8000,
    max_finite: 0x7bff,
};
const BF16: Format = Format {
    fraction: 7,
    bias: 127,
    exponent_mask: 255,
    sign_mask: 0x8000,
    max_finite: 0x7f7f,
};
const E4M3FN: Format = Format {
    fraction: 3,
    bias: 7,
    exponent_mask: 15,
    sign_mask: 0x80,
    max_finite: 0x7e,
};
const E5M2: Format = Format {
    fraction: 2,
    bias: 15,
    exponent_mask: 31,
    sign_mask: 0x80,
    max_finite: 0x7b,
};

#[derive(Clone, Copy)]
struct Finite {
    negative: bool,
    significand: u64,
    scale: i32,
}

// Every cast is bounded by a masked field, <=53-bit exact input or intermediate, or
// destination <=11-bit rounded significand. Exponent differences are nonnegative.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn decode(bits: u16, format: Format) -> Result<Finite, PcuClampedError<u16>> {
    let exponent = (bits >> format.fraction) & format.exponent_mask;
    if bits & (format.sign_mask - 1) > format.max_finite {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    let fraction = u64::from(bits & ((1 << format.fraction) - 1));
    Ok(Finite {
        negative: bits & format.sign_mask != 0,
        significand: fraction
            | if exponent == 0 {
                0
            } else {
                1 << format.fraction
            },
        scale: i32::from(exponent.max(1)) - format.bias - format.fraction as i32,
    })
}

#[derive(Clone, Copy)]
enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

// n/d has at most 53 numerator bits and 11 denominator bits. The normal
// precision shift is in [-52, 21]; subnormal shifts are bounded above by it.
// Extremely negative shifts are handled before shifting, so no shift can panic.
#[allow(clippy::cast_sign_loss)]
fn round_rational(n: u64, d: u64, shift: i32) -> (u64, bool) {
    if shift < -64 {
        return (0, n != 0);
    }
    let (numerator, denominator) = if shift >= 0 {
        (u128::from(n) << shift as u32, u128::from(d))
    } else {
        (u128::from(n), u128::from(d) << (-shift) as u32)
    };
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let increment = remainder > denominator - remainder
        || (remainder == denominator - remainder && quotient & 1 != 0);
    // Destination rounding bounds the quotient to <=12 bits.
    #[allow(clippy::cast_possible_truncation)]
    let result = (quotient + u128::from(increment)) as u64;
    (result, remainder != 0)
}

#[allow(clippy::cast_possible_wrap, clippy::cast_sign_loss)]
fn ratio_exponent(n: u64, d: u64) -> i32 {
    let estimate = d.leading_zeros() as i32 - n.leading_zeros() as i32;
    let below = if estimate >= 0 {
        u128::from(n) < u128::from(d) << estimate as u32
    } else {
        u128::from(n) << ((-estimate) as u32) < u128::from(d)
    };
    estimate - i32::from(below)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]
fn pack(
    negative: bool,
    n: u64,
    d: u64,
    scale: i32,
    format: Format,
    policy: PcuFloatUnderflowPolicy,
) -> Result<u16, PcuClampedError<u16>> {
    let sign = if negative { format.sign_mask } else { 0 };
    if n == 0 {
        return Ok(sign);
    }
    let fraction = format.fraction as i32;
    let exponent = scale + ratio_exponent(n, d);
    let normal_scale = exponent - fraction;
    let (unbounded, _) = round_rational(n, d, scale - normal_scale);
    let unbounded_exponent = exponent + i32::from(unbounded == 1 << (format.fraction + 1));
    let min_exponent = 1 - format.bias;
    let final_scale = normal_scale.max(min_exponent - fraction);
    let (mut rounded, inexact) = round_rational(n, d, scale - final_scale);
    let mut packed_exponent = final_scale + fraction;
    if rounded == 1 << (format.fraction + 1) {
        rounded >>= 1;
        packed_exponent += 1;
    }
    // Compare the rounded destination fields, rather than assuming the final
    // exponent is reserved. E4M3FN admits exponent 1111 through fraction 110.
    let max_exponent = i32::from(format.max_finite >> format.fraction) - format.bias;
    let max_significand =
        (1 << format.fraction) | u64::from(format.max_finite & ((1 << format.fraction) - 1));
    if packed_exponent > max_exponent
        || (packed_exponent == max_exponent && rounded > max_significand)
    {
        let endpoint = sign | format.max_finite;
        return Err(PcuClampedError::Range(PcuClampedFault::new(
            PcuExecutionFaultKind::ArithmeticOverflow,
            endpoint,
        )));
    }
    let subnormal = rounded < 1 << format.fraction;
    let magnitude = if subnormal {
        rounded as u16
    } else {
        (((packed_exponent + format.bias) as u16) << format.fraction)
            | (rounded as u16 & ((1 << format.fraction) - 1))
    };
    let bits = sign | magnitude;
    if policy.rejects(PcuFloatUnderflowClassification {
        is_tiny_after_rounding: unbounded_exponent < min_exponent,
        is_inexact: inexact,
        result_is_subnormal: subnormal && rounded != 0,
    }) {
        Err(PcuClampedError::Range(PcuClampedFault::new(
            PcuExecutionFaultKind::ArithmeticUnderflow,
            bits,
        )))
    } else {
        Ok(bits)
    }
}

#[allow(clippy::cast_sign_loss)]
fn arithmetic(
    left: u16,
    right: u16,
    format: Format,
    operation: Operation,
    policy: PcuFloatUnderflowPolicy,
) -> Result<u16, PcuClampedError<u16>> {
    let mut a = decode(left, format)?;
    let mut b = decode(right, format)?;
    match operation {
        Operation::Mul => pack(
            a.negative ^ b.negative,
            a.significand * b.significand,
            1,
            a.scale + b.scale,
            format,
            policy,
        ),
        Operation::Div => {
            if b.significand == 0 {
                return Err(PcuClampedError::Fatal(PcuExecutionFaultKind::DivideByZero));
            }
            pack(
                a.negative ^ b.negative,
                a.significand,
                b.significand,
                a.scale - b.scale,
                format,
                policy,
            )
        }
        Operation::Add | Operation::Sub => {
            b.negative ^= matches!(operation, Operation::Sub);
            if a.significand == 0 && b.significand == 0 {
                return Ok(if a.negative && b.negative {
                    format.sign_mask
                } else {
                    0
                });
            }
            if a.significand == 0 {
                a = b;
                b.significand = 0;
            }
            if b.significand == 0 {
                return pack(a.negative, a.significand, 1, a.scale, format, policy);
            }
            if a.scale < b.scale {
                core::mem::swap(&mut a, &mut b);
            }
            let gap = a.scale - b.scale;
            if gap > 32 {
                // Precision <=11: the smaller operand is <2^-22 of one ulp
                // of the larger, even allowing a subnormal significand. It cannot
                // reach a midpoint or cancel a leading bit. Such a gap requires
                // the larger operand to be normal, so neither underflow policy
                // can reject its rounded result. Returning that exact encoding
                // is valid despite the mathematically inexact addition.
                return pack(a.negative, a.significand, 1, a.scale, format, policy);
            }
            let large = a.significand << gap as u32;
            let (negative, numerator) = if a.negative == b.negative {
                (a.negative, large + b.significand)
            } else if large >= b.significand {
                (a.negative && large != b.significand, large - b.significand)
            } else {
                (b.negative, b.significand - large)
            };
            pack(negative, numerator, 1, b.scale, format, policy)
        }
    }
}

fn convert_error<T>(error: PcuClampedError<u16>, convert: impl Fn(u16) -> T) -> PcuClampedError<T> {
    match error {
        PcuClampedError::Fatal(kind) => PcuClampedError::Fatal(kind),
        PcuClampedError::Range(fault) => PcuClampedError::Range(PcuClampedFault::new(
            fault.kind(),
            convert(fault.clamped_value()),
        )),
    }
}

macro_rules! operation {
    ($ty:ty, $format:ident, $op:ident, $checked:ident, $checked_policy:ident, $clamped:ident, $clamped_policy:ident) => {
        impl $ty {
            /// Applies finite-input nearest/ties-even arithmetic with default underflow errors.
            /// # Errors
            /// Rejects nonfinite operands, overflow, tiny inexact results, and zero divisors.
            pub fn $checked(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind> {
                self.$checked_policy(rhs, PcuFloatUnderflowPolicy::default())
            }
            /// Applies finite-input arithmetic using the independent underflow policy.
            /// # Errors
            /// Returns invalid operand, overflow, division by zero, or policy-rejected underflow.
            pub fn $checked_policy(
                self,
                rhs: Self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuExecutionFaultKind> {
                self.$clamped_policy(rhs, policy)
                    .map_err(|fault| fault.kind())
            }
            /// Applies arithmetic retaining observable finite overflow and rounded underflow payloads.
            /// # Errors
            /// Returns a fatal operand/division fault or an observable range fault.
            pub fn $clamped(self, rhs: Self) -> Result<Self, PcuClampedError<Self>> {
                self.$clamped_policy(rhs, PcuFloatUnderflowPolicy::default())
            }
            /// Applies arithmetic retaining range payloads under the selected underflow policy.
            /// # Errors
            /// Returns a fatal operand/division fault or an observable range fault.
            pub fn $clamped_policy(
                self,
                rhs: Self,
                policy: PcuFloatUnderflowPolicy,
            ) -> Result<Self, PcuClampedError<Self>> {
                #[allow(clippy::cast_possible_truncation)]
                // Successful/recovered bits are bounded by this format's sign and finite masks.
                let convert = |bits| Self::from_bits(bits as _);
                arithmetic(
                    u16::from(self.to_bits()),
                    u16::from(rhs.to_bits()),
                    $format,
                    Operation::$op,
                    policy,
                )
                .map(convert)
                .map_err(|fault| convert_error(fault, convert))
            }
        }
    };
}
macro_rules! operations {
    ($ty:ty, $format:ident) => {
        operation!(
            $ty,
            $format,
            Add,
            pcu_checked_add,
            pcu_checked_add_with_policy,
            pcu_clamped_add,
            pcu_clamped_add_with_policy
        );
        operation!(
            $ty,
            $format,
            Sub,
            pcu_checked_sub,
            pcu_checked_sub_with_policy,
            pcu_clamped_sub,
            pcu_clamped_sub_with_policy
        );
        operation!(
            $ty,
            $format,
            Mul,
            pcu_checked_mul,
            pcu_checked_mul_with_policy,
            pcu_clamped_mul,
            pcu_clamped_mul_with_policy
        );
        operation!(
            $ty,
            $format,
            Div,
            pcu_checked_div,
            pcu_checked_div_with_policy,
            pcu_clamped_div,
            pcu_clamped_div_with_policy
        );
    };
}
operations!(PcuF16Bits, F16);
operations!(PcuBf16Bits, BF16);
operations!(PcuF8E4M3FnBits, E4M3FN);
operations!(PcuF8E5M2Bits, E5M2);

mod traits;

#[cfg(test)]
mod tests;

#[path = "conversion/conversion.rs"]
mod conversion;
