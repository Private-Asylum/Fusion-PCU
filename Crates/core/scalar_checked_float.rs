//! Checked binary16/BF16/OFP8/binary32/binary64 arithmetic with integer significands.
//!
//! Numerical rules reference IEEE Std 754-2019: clauses 4.3.1/4.3.3 for nearest,
//! ties-to-even rounding, 7.4 for overflow, and 7.5(a) for tininess after rounding.
//! Destination precision is applied with an unbounded exponent before classifying tininess;
//! inexactness also accounts for final packing into the destination exponent range.
//!
//! Rejecting all non-finite operands and returning errors instead of IEEE default results
//! and status flags are PCU policies. In particular, this finite-input profile is stricter
//! than IEEE invalid-operation handling (clauses 6.2 and 7.2). These bounded operations do
//! not establish conformance for the full standard or for backend library reductions.
//! OFP8 is not an IEEE basic format; its named encodings use the declared PCU
//! rounding/exception contract, including E4M3FN's finite final exponent.

#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuFloatUnderflowClassification,
    PcuFloatUnderflowPolicy,
    PcuClampedError,
    PcuClampedFault,
    PcuScalar,
};

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
    impl Sealed for crate::PcuF16Bits {}
    impl Sealed for crate::PcuBf16Bits {}
    impl Sealed for crate::PcuF8E4M3FnBits {}
    impl Sealed for crate::PcuF8E5M2Bits {}
}

mod f64;

/// Checked scalar Add/Sub/Mul/Div for binary16/32/64 and named BF16/OFP8 formats.
///
/// This host-side arithmetic contract does not imply backend execution support.
#[allow(private_bounds)] // Prevents downstream checked-float implementations from bypassing the contract.
pub trait PcuCheckedFloat: PcuScalar + sealed::Sealed + Sized + Copy {
    /// Selects the upstream derivative when this finite input is strictly positive.
    ///
    /// Both operands must be finite even when the gradient would be masked out.
    /// Nonpositive inputs, including either signed zero, produce positive zero.
    /// Selection is a PCU differentiation rule, not IEEE arithmetic. An unchanged
    /// subnormal is exact under IEEE 754-2019 clause 7.5; the selected policy may
    /// deliberately reject it. No multiply or rounding is introduced.
    ///
    /// # Errors
    /// Returns invalid operand or policy-rejected selected subnormal output.
    fn pcu_checked_relu_backward_with_policy(
        self,
        upstream: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;
    /// Adds finite operands, rejecting invalid inputs and overflow by default.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_add(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Subtracts finite operands, rejecting invalid inputs and overflow by default.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_sub(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Multiplies finite operands, rejecting invalid inputs and overflow by default.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_mul(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Divides finite operands, rejecting zero denominators, overflow, and policy-rejected underflow.
    ///
    /// # Errors
    /// Returns invalid operand, divide by zero, overflow, or policy-rejected underflow.
    fn pcu_checked_div(self, rhs: Self) -> Result<Self, PcuExecutionFaultKind>;
    /// Inverts the sign bit of a finite value, preserving exact subnormals and signed zero.
    ///
    /// # Errors
    /// Returns invalid operand or policy-rejected subnormal output.
    fn pcu_checked_neg(self) -> Result<Self, PcuExecutionFaultKind>;
    /// Exact finite-input sign inversion with the selected underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand or policy-rejected subnormal output.
    fn pcu_checked_neg_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;

    /// Applies PCU `ReLU` selection to a finite value, canonicalizing nonpositive values to +0.
    ///
    /// This is a PCU selection contract, not an IEEE 754 arithmetic operation: NaNs and
    /// infinities are rejected, and both signed zeros produce positive zero. Subnormal
    /// classification follows IEEE Std 754-2019 clause 7.5(a) under the selected policy.
    ///
    /// # Errors
    /// Returns `InvalidFloatingOperand` for nonfinite inputs or `ArithmeticUnderflow` when the
    /// selected policy rejects the subnormal result.
    fn pcu_checked_relu(self) -> Result<Self, PcuExecutionFaultKind>;
    /// Adds using the supplied floating underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_add_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;
    /// Subtracts using the supplied floating underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_sub_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;
    /// Multiplies using the supplied floating underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand, overflow, or policy-rejected underflow.
    fn pcu_checked_mul_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;
    /// Divides using the supplied floating underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand, divide by zero, overflow, or policy-rejected underflow.
    fn pcu_checked_div_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;
    /// Applies PCU `ReLU` selection using the supplied underflow policy.
    ///
    /// # Errors
    /// Returns `InvalidFloatingOperand` for nonfinite inputs or `ArithmeticUnderflow` when the
    /// selected policy rejects the subnormal result.
    fn pcu_checked_relu_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind>;
}

/// Checked binary16/BF16/binary32/binary64 arithmetic retaining range-fault payloads.
///
/// Unlike saturation, a range violation remains an error. Its payload is the nearest finite
/// endpoint for overflow or the rounded result for policy-rejected underflow.
#[allow(private_bounds)] // The sealed set limits this behavior to explicitly implemented binary formats.
pub trait PcuClampedFloat: PcuCheckedFloat + sealed::Sealed {
    /// Adds, retaining the signed maximum finite endpoint on overflow.
    ///
    /// # Errors
    /// Returns a fatal operand error or a range fault with its clamped value.
    fn pcu_clamped_add(self, rhs: Self) -> Result<Self, PcuClampedError<Self>>;
    /// Subtracts, retaining the signed maximum finite endpoint on overflow.
    ///
    /// # Errors
    /// Returns a fatal operand error or a range fault with its clamped value.
    fn pcu_clamped_sub(self, rhs: Self) -> Result<Self, PcuClampedError<Self>>;
    /// Multiplies, retaining range-fault payloads.
    ///
    /// # Errors
    /// Returns a fatal operand error or a range fault with its clamped value.
    fn pcu_clamped_mul(self, rhs: Self) -> Result<Self, PcuClampedError<Self>>;
    /// Divides, retaining range-fault payloads.
    ///
    /// # Errors
    /// Returns a fatal operand/division error or a range fault with its clamped value.
    fn pcu_clamped_div(self, rhs: Self) -> Result<Self, PcuClampedError<Self>>;
    /// Inverts a finite value's sign, retaining policy-rejected exact subnormals as range faults.
    ///
    /// # Errors
    /// Returns a fatal invalid operand or an observable subnormal range fault.
    fn pcu_clamped_neg(self) -> Result<Self, PcuClampedError<Self>>;
    /// Exact finite-input sign inversion with the selected underflow policy.
    ///
    /// # Errors
    /// Returns invalid operand or policy-rejected subnormal output.
    fn pcu_clamped_neg_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>>;

    /// Applies PCU `ReLU` selection, retaining a policy-rejected subnormal result as a range fault.
    ///
    /// # Errors
    /// Returns a fatal invalid-operand error or a range fault that retains the subnormal value.
    fn pcu_clamped_relu(self) -> Result<Self, PcuClampedError<Self>>;
    /// Adds using the supplied underflow policy.
    ///
    /// # Errors
    /// Returns a fatal operand error or a range fault with its clamped value.
    fn pcu_clamped_add_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>>;
    /// Subtracts using the supplied underflow policy.
    ///
    /// # Errors
    /// Returns a fatal operand error or a range fault with its clamped value.
    fn pcu_clamped_sub_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>>;
    /// Multiplies using the supplied underflow policy.
    ///
    /// # Errors
    /// Returns a fatal operand error or a range fault with its clamped value.
    fn pcu_clamped_mul_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>>;
    /// Divides using the supplied underflow policy.
    ///
    /// # Errors
    /// Returns a fatal operand/division error or a range fault with its clamped value.
    fn pcu_clamped_div_with_policy(
        self,
        rhs: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>>;
    /// Applies PCU `ReLU` selection using the supplied underflow policy.
    ///
    /// # Errors
    /// Returns a fatal invalid-operand error or a range fault when the policy rejects the result.
    fn pcu_clamped_relu_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>>;
}

fn neg_f32<const CLAMP: bool>(
    value: f32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    let result_bits = bits ^ 0x8000_0000;
    let subnormal = result_bits & !0x8000_0000 != 0 && result_bits & 0x7f80_0000 == 0;
    let class = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: subnormal,
        is_inexact: false,
        result_is_subnormal: subnormal,
    };
    let result = f32::from_bits(result_bits);
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

fn relu_f32<const CLAMP: bool>(
    value: f32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let bits = value.to_bits();
    if bits & 0x7f80_0000 == 0x7f80_0000 {
        return Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ));
    }
    let result_bits = if bits & 0x8000_0000 != 0 || bits.trailing_zeros() >= 31 {
        0
    } else {
        bits
    };
    let subnormal = result_bits != 0 && result_bits & 0x7f80_0000 == 0;
    let class = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: subnormal,
        is_inexact: false,
        result_is_subnormal: subnormal,
    };
    let result = f32::from_bits(result_bits);
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

fn decode(bits: u32) -> Result<Option<Parts>, PcuExecutionFaultKind> {
    let exponent = (bits >> 23) & 0xff;
    let fraction = bits & 0x007f_ffff;
    if exponent == 0xff {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    if exponent == 0 && fraction == 0 {
        return Ok(None);
    }
    let (unbiased, sig) = if exponent == 0 {
        (-126, u64::from(fraction))
    } else {
        (
            exponent.cast_signed() - 127,
            u64::from(fraction | 0x0080_0000),
        )
    };
    Ok(Some(Parts {
        sign: bits >> 31,
        exponent: unbiased,
        significand: sig,
    }))
}

/// Right shift while retaining whether any discarded bit was nonzero.
fn shift_right_jam(value: u64, distance: u32) -> u64 {
    if distance == 0 {
        value
    } else if distance < 64 {
        (value >> distance) | u64::from(value & ((1_u64 << distance) - 1) != 0)
    } else {
        u64::from(value != 0)
    }
}

// IEEE Std 754-2019, clause 4.3.1: an exact halfway case selects the even significand.
const fn round_increment(significand: u64, discarded: u64) -> bool {
    discarded > 4 || (discarded == 4 && significand & 1 != 0)
}

fn round_to_precision(mut ext: u64, mut exponent: i32) -> (u64, i32, bool) {
    while ext >= (1_u64 << 27) {
        ext = shift_right_jam(ext, 1);
        exponent += 1;
    }
    while ext < (1_u64 << 26) {
        ext <<= 1;
        exponent -= 1;
    }
    let discarded = ext & 7;
    let mut significand = ext >> 3;
    let inexact = discarded != 0;
    if round_increment(significand, discarded) {
        significand += 1;
    }
    if significand == (1_u64 << 24) {
        significand >>= 1;
        exponent += 1;
    }
    (significand, exponent, inexact)
}

fn pack<const CLAMP: bool>(
    sign: u32,
    mut ext: u64,
    mut exponent: i32,
) -> Result<(f32, PcuFloatUnderflowClassification), PcuClampedError<f32>> {
    while ext >= (1_u64 << 27) {
        ext = shift_right_jam(ext, 1);
        exponent += 1;
    }
    while ext < (1_u64 << 26) {
        ext <<= 1;
        exponent -= 1;
    }

    // IEEE Std 754-2019, clause 7.5(a): test destination-precision rounding with
    // an unbounded exponent, before the final subnormal/zero packing step.
    let (_, unbounded_exp, _) = round_to_precision(ext, exponent);
    let tiny = unbounded_exp < -126;

    if exponent < -126 {
        ext = shift_right_jam(ext, (-126 - exponent).cast_unsigned());
        exponent = -126;
    }
    let discarded = ext & 7;
    let mut significand = ext >> 3;
    let mut inexact = discarded != 0;
    if round_increment(significand, discarded) {
        significand += 1;
    }

    let (bits, subnormal) = if exponent == -126 && significand < 0x0080_0000 {
        (
            sign << 31 | u32::try_from(significand).expect("rounded significand fits binary32"),
            significand != 0,
        )
    } else {
        if significand >= (1_u64 << 24) {
            significand >>= 1;
            exponent += 1;
        }
        if exponent > 127 {
            if CLAMP {
                return Err(PcuClampedError::Range(PcuClampedFault::new(
                    PcuExecutionFaultKind::ArithmeticOverflow,
                    f32::from_bits((sign << 31) | 0x7f7f_ffff),
                )));
            }
            return Err(PcuClampedError::Fatal(
                PcuExecutionFaultKind::ArithmeticOverflow,
            ));
        }
        let exp_field = (exponent + 127).cast_unsigned();
        (
            sign << 31
                | (exp_field << 23)
                | (u32::try_from(significand).expect("rounded significand fits binary32")
                    & 0x007f_ffff),
            false,
        )
    };
    // Zero can only be inexact when right-shifting discarded nonzero information.
    if bits.trailing_zeros() >= 31 {
        inexact = discarded != 0;
    }
    Ok((
        f32::from_bits(bits),
        PcuFloatUnderflowClassification {
            is_tiny_after_rounding: tiny,
            is_inexact: inexact,
            result_is_subnormal: subnormal,
        },
    ))
}

fn add<const CLAMP: bool>(
    left: f32,
    right: f32,
    subtract: bool,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let a = decode(left.to_bits()).map_err(PcuClampedError::Fatal)?;
    let mut b = decode(right.to_bits()).map_err(PcuClampedError::Fatal)?;
    if let Some(parts) = &mut b {
        parts.sign ^= u32::from(subtract);
    }
    let Some(mut a) = a else {
        if let Some(b) = b {
            let value = f32::from_bits((b.sign << 31) | (right.to_bits() & 0x7fff_ffff));
            return passthrough::<CLAMP>(value, policy);
        }
        let left_sign = left.to_bits() >> 31;
        let right_sign = (right.to_bits() >> 31) ^ u32::from(subtract);
        return Ok(f32::from_bits(
            u32::from(left_sign == right_sign) * (left_sign << 31),
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
    let (result, class) = pack::<CLAMP>(result_sign, ext, a.exponent)?;
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

fn passthrough<const CLAMP: bool>(
    value: f32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let magnitude = value.to_bits() & 0x7fff_ffff;
    let class = PcuFloatUnderflowClassification {
        is_tiny_after_rounding: magnitude != 0 && magnitude < 0x0080_0000,
        is_inexact: false,
        result_is_subnormal: magnitude != 0 && magnitude < 0x0080_0000,
    };
    if policy.rejects(class) {
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

fn multiply<const CLAMP: bool>(
    left: f32,
    right: f32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let a = decode(left.to_bits())
        .map_err(PcuClampedError::Fatal)?
        .unwrap_or_else(|| Parts {
            sign: left.to_bits() >> 31,
            exponent: -126,
            significand: 0,
        });
    let b = decode(right.to_bits())
        .map_err(PcuClampedError::Fatal)?
        .unwrap_or_else(|| Parts {
            sign: right.to_bits() >> 31,
            exponent: -126,
            significand: 0,
        });
    if a.significand == 0 || b.significand == 0 {
        return Ok(f32::from_bits((a.sign ^ b.sign) << 31));
    }
    let product = a.significand * b.significand;
    let top = 63 - product.leading_zeros().cast_signed();
    let exponent = a.exponent + b.exponent + top - 46;
    let ext = if top > 26 {
        shift_right_jam(product, (top - 26).cast_unsigned())
    } else {
        product << (26 - top)
    };
    let (result, class) = pack::<CLAMP>(a.sign ^ b.sign, ext, exponent)?;
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

// The denominator is already proved nonzero; retain the quotient/remainder pair
// that classifies rounding inexactness rather than adding a separate divisibility API.
#[allow(clippy::manual_is_multiple_of)]
fn divide<const CLAMP: bool>(
    left: f32,
    right: f32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuClampedError<f32>> {
    let a = decode(left.to_bits()).map_err(PcuClampedError::Fatal)?;
    let b = decode(right.to_bits()).map_err(PcuClampedError::Fatal)?;
    let sign = (left.to_bits() ^ right.to_bits()) >> 31;
    let Some(mut b) = b else {
        return Err(PcuClampedError::Fatal(PcuExecutionFaultKind::DivideByZero));
    };
    let Some(mut a) = a else {
        return Ok(f32::from_bits(sign << 31));
    };
    let a_shift = 23 - a.significand.ilog2();
    a.significand <<= a_shift;
    a.exponent -= a_shift.cast_signed();
    let b_shift = 23 - b.significand.ilog2();
    b.significand <<= b_shift;
    b.exponent -= b_shift.cast_signed();

    let mut numerator = a.significand;
    let mut exponent = a.exponent - b.exponent;
    if numerator < b.significand {
        numerator <<= 1;
        exponent -= 1;
    }
    let scaled = numerator << 26;
    let mut ext = scaled / b.significand;
    if scaled % b.significand != 0 {
        ext |= 1;
    }
    let (result, class) = pack::<CLAMP>(sign, ext, exponent)?;
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

impl PcuCheckedFloat for f32 {
    fn pcu_checked_relu_backward_with_policy(
        self,
        upstream: Self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        if !self.is_finite() || !upstream.is_finite() {
            return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
        }
        // IEEE 754 binary32 selection is exact: positive finite subnormals
        // remain positive even when the caller's hardware mode flushes inputs.
        // Inspect the encoding rather than inheriting that ambient mode.
        let bits = self.to_bits();
        let positive = bits & 0x8000_0000 == 0 && bits & 0x7fff_ffff != 0;
        let result = if positive { upstream } else { 0.0 };
        // Two exact sign inversions validate the selected representation without rounding it.
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
        neg_f32::<false>(self, policy).map_err(|fault| fault.kind())
    }

    fn pcu_checked_relu(self) -> Result<Self, PcuExecutionFaultKind> {
        self.pcu_checked_relu_with_policy(PcuFloatUnderflowPolicy::default())
    }
    fn pcu_checked_relu_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuExecutionFaultKind> {
        relu_f32::<false>(self, policy).map_err(|fault| fault.kind())
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

impl PcuClampedFloat for f32 {
    fn pcu_clamped_neg(self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_neg_with_policy(PcuFloatUnderflowPolicy::default())
    }
    fn pcu_clamped_neg_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        neg_f32::<true>(self, policy)
    }

    fn pcu_clamped_relu(self) -> Result<Self, PcuClampedError<Self>> {
        self.pcu_clamped_relu_with_policy(PcuFloatUnderflowPolicy::default())
    }
    fn pcu_clamped_relu_with_policy(
        self,
        policy: PcuFloatUnderflowPolicy,
    ) -> Result<Self, PcuClampedError<Self>> {
        relu_f32::<true>(self, policy)
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
#[path = "scalar_checked_float/clamped_tests.rs"]
mod clamped_tests;

#[cfg(test)]
mod tests;

#[cfg(test)]
#[path = "scalar_checked_float/neg_tests.rs"]
mod neg_tests;
