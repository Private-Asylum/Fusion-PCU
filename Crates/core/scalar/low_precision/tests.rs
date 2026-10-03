#[rustfmt::skip]
use super::{
    arithmetic,
    Operation,
    Format,
    F16,
    BF16,
    E4M3FN,
    E5M2,
};
#[rustfmt::skip]
use crate::{
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuExecutionFaultKind,
    PcuClampedError,
    PcuFloatUnderflowPolicy,
};

fn value(bits: u16, format: Format) -> f64 {
    if format.fraction == 10 {
        f64::from(PcuF16Bits::from_bits(bits).to_f32())
    } else if format.fraction == 7 {
        f64::from(PcuBf16Bits::from_bits(bits).to_f32())
    } else {
        // Independent OFP8 numerical decoder, not the production widening implementation.
        let magnitude = bits & (format.sign_mask - 1);
        if magnitude > format.max_finite {
            return f64::NAN;
        }
        let exponent = magnitude >> format.fraction;
        let significand = (magnitude & ((1 << format.fraction) - 1))
            | if exponent == 0 {
                0
            } else {
                1 << format.fraction
            };
        let scale =
            i32::from(exponent.max(1)) - format.bias - i32::try_from(format.fraction).unwrap();
        let power = f64::from_bits(u64::try_from(scale + 1023).unwrap() << 52);
        let decoded = f64::from(significand) * power;
        if bits & format.sign_mask == 0 {
            decoded
        } else {
            -decoded
        }
    }
}
// Independent oracle searches destination encodings against exact neighboring
// midpoints. Half inputs and products fit f64 exactly. Division's <=11-bit
// denominator separates non-dyadic half midpoints from the result by much more
// than f64 rounding error; dyadic ties are exact. Large-gap additions cannot
// cross a half midpoint; cancellation and near-midpoint sums fit 53 bits.
#[allow(clippy::float_cmp)] // Exact dyadic midpoint equality decides ties; tolerance would corrupt the oracle.
fn oracle(
    left: u16,
    right: u16,
    format: Format,
    operation: Operation,
) -> Result<u16, PcuExecutionFaultKind> {
    let a = value(left, format);
    let b = value(right, format);
    if !a.is_finite() || !b.is_finite() {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    if matches!(operation, Operation::Div) && b == 0.0 {
        return Err(PcuExecutionFaultKind::DivideByZero);
    }
    let result = match operation {
        Operation::Add => a + b,
        Operation::Sub => a - b,
        Operation::Mul => a * b,
        Operation::Div => a / b,
    };
    round_value(result, format)
}

#[allow(clippy::float_cmp)] // Exact dyadic midpoint equality determines rounding parity.
fn round_value(result: f64, format: Format) -> Result<u16, PcuExecutionFaultKind> {
    if !result.is_finite() {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    let sign = if result.is_sign_negative() {
        format.sign_mask
    } else {
        0
    };
    let magnitude = result.abs();
    let max = format.max_finite;
    let max_value = value(max, format);
    let overflow_midpoint = max_value + (max_value - value(max - 1, format)) / 2.0;
    if magnitude > overflow_midpoint || (magnitude == overflow_midpoint && max & 1 != 0) {
        return Err(PcuExecutionFaultKind::ArithmeticOverflow);
    }
    let mut lower = 0;
    let mut upper = max;
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        if value(midpoint, format) <= magnitude {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    if lower == max {
        return Ok(sign | lower);
    }
    let midpoint = value(lower, format).midpoint(value(lower + 1, format));
    Ok(sign
        | (lower + u16::from(magnitude > midpoint || (magnitude == midpoint && lower & 1 != 0))))
}
#[allow(clippy::float_cmp)] // Exact equality identifies representable results; no tolerance is valid.
fn compare(left: u16, right: u16, format: Format, operation: Operation) {
    let actual = arithmetic(
        left,
        right,
        format,
        operation,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    )
    .map_err(|fault| fault.kind());
    let expected = oracle(left, right, format, operation);
    assert_eq!(
        actual, expected,
        "left={left:#06x}, right={right:#06x}, fraction={}",
        format.fraction
    );
    let normal = 1 << format.fraction;
    let a = value(left, format);
    let b = value(right, format);
    let exact = match operation {
        Operation::Add => a + b,
        Operation::Sub => a - b,
        Operation::Mul => a * b,
        Operation::Div => a / b,
    };
    // Below minimum normal, unbounded precision spacing is half the stored
    // subnormal spacing. Its midpoint boundary therefore lies one quarter of
    // a subnormal ulp below minimum normal; equality rounds to the even normal.
    let tiny_boundary = value(normal, format) - value(1, format) / 4.0;
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        let classified = expected.and_then(|bits| {
            let magnitude = bits & (format.sign_mask - 1);
            let tiny_inexact =
                exact != 0.0 && exact.abs() < tiny_boundary && exact != value(bits, format);
            let tight = policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                && magnitude != 0
                && magnitude < normal;
            if tiny_inexact || tight {
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            } else {
                Ok(bits)
            }
        });
        assert_eq!(
            arithmetic(left, right, format, operation, policy).map_err(|fault| fault.kind()),
            classified,
            "classification left={left:#06x}, right={right:#06x}, fraction={}",
            format.fraction
        );
    }
}

#[test]
fn every_fp8_pair_operation_and_underflow_policy_matches_midpoint_oracle() {
    // 2 formats * 256^2 pairs * 4 operations * 3 policies = 1,572,864 comparisons.
    for format in [E4M3FN, E5M2] {
        for left in 0..=255 {
            for right in 0..=255 {
                for operation in [
                    Operation::Add,
                    Operation::Sub,
                    Operation::Mul,
                    Operation::Div,
                ] {
                    compare(left, right, format, operation);
                }
            }
        }
    }
}

#[test]
fn fp8_finite_final_exponent_overflow_ties_and_range_payloads_are_explicit() {
    let max = PcuF8E4M3FnBits::from_bits(0x7e);
    let zero = PcuF8E4M3FnBits::from_bits(0);
    assert_eq!(max.pcu_checked_add(zero), Ok(max)); // 448, not a reserved exponent.
    assert_eq!(PcuF8E4M3FnBits::pcu_checked_from_f64(464.0), Ok(max));
    let overflow = PcuF8E4M3FnBits::pcu_clamped_from_f64(464.0_f64.next_up()).unwrap_err();
    assert_eq!(overflow.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(recovered(overflow), max);
    let overflow = PcuF8E5M2Bits::pcu_clamped_from_f64(-61_440.0).unwrap_err();
    assert_eq!(overflow.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(recovered(overflow).to_bits(), 0xfb);
    let least = PcuF8E4M3FnBits::from_bits(1);
    let half = PcuF8E4M3FnBits::from_bits(0x30);
    let underflow = least.pcu_clamped_mul(half).unwrap_err();
    assert_eq!(underflow.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    assert_eq!(recovered(underflow).to_bits(), 0);
    assert_eq!(
        least.pcu_checked_mul(PcuF8E4M3FnBits::from_bits(0x38)),
        Ok(least)
    );
    assert_eq!(
        max.pcu_checked_div(zero),
        Err(PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(
        PcuF8E4M3FnBits::from_bits(0x7f).pcu_checked_add(zero),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
}

fn recovered<T>(error: PcuClampedError<T>) -> T {
    match error {
        PcuClampedError::Range(fault) => fault.clamped_value(),
        PcuClampedError::Fatal(kind) => panic!("expected useful range payload, got {kind:?}"),
    }
}

#[path = "fp8_conversion_tests.rs"]
mod fp8_conversion_tests;
#[test]
fn every_encoding_against_edges_and_independent_midpoint_oracle() {
    for format in [F16, BF16] {
        let one = u16::try_from(format.bias).unwrap() << format.fraction;
        let infinity = format.exponent_mask << format.fraction;
        let normal = 1 << format.fraction;
        let edges = [
            0,
            0x8000,
            1,
            normal - 1,
            normal,
            one - 1,
            one,
            one + 1,
            one | 0x8000,
            infinity - 1,
            infinity,
            infinity + 1,
        ];
        for left in 0..=u16::MAX {
            for right in edges {
                for op in [
                    Operation::Add,
                    Operation::Sub,
                    Operation::Mul,
                    Operation::Div,
                ] {
                    compare(left, right, format, op);
                }
            }
        }
    }
}
#[test]
fn randomized_pairs_against_independent_midpoint_oracle() {
    let mut state = 0x65a8_9391_16bc_712d_u64;
    for format in [F16, BF16] {
        for _ in 0..50_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bytes = state.to_le_bytes();
            let a = u16::from_le_bytes([bytes[0], bytes[1]]);
            let b = u16::from_le_bytes([bytes[2], bytes[3]]);
            for op in [
                Operation::Add,
                Operation::Sub,
                Operation::Mul,
                Operation::Div,
            ] {
                compare(a, b, format, op);
            }
        }
    }
}
#[test]
fn exact_subnormal_default_tiny_inexact_and_observable_clamps() {
    for format in [F16, BF16] {
        let one = u16::try_from(format.bias).unwrap() << format.fraction;
        let half = one - (1 << format.fraction);
        for op in [Operation::Add, Operation::Sub] {
            assert_eq!(
                arithmetic(1, 0, format, op, PcuFloatUnderflowPolicy::default()),
                Ok(1)
            );
        }
        for op in [Operation::Mul, Operation::Div] {
            assert_eq!(
                arithmetic(1, one, format, op, PcuFloatUnderflowPolicy::default()),
                Ok(1)
            );
        }
        match arithmetic(
            1,
            half,
            format,
            Operation::Mul,
            PcuFloatUnderflowPolicy::default(),
        ) {
            Err(PcuClampedError::Range(fault)) => {
                assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
                assert_eq!(fault.clamped_value(), 0);
            }
            other => panic!("expected tiny inexact range fault: {other:?}"),
        }
        assert_eq!(
            arithmetic(
                1,
                half,
                format,
                Operation::Mul,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            Ok(0)
        );
        assert_eq!(
            arithmetic(
                1,
                one,
                format,
                Operation::Mul,
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            )
            .map_err(|fault| fault.kind()),
            Err(PcuExecutionFaultKind::ArithmeticUnderflow)
        );
        let max = ((format.exponent_mask - 1) << format.fraction) | ((1 << format.fraction) - 1);
        match arithmetic(
            max | 0x8000,
            max | 0x8000,
            format,
            Operation::Add,
            PcuFloatUnderflowPolicy::default(),
        ) {
            Err(PcuClampedError::Range(fault)) => {
                assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
                assert_eq!(fault.clamped_value(), max | 0x8000);
            }
            other => panic!("expected overflow endpoint: {other:?}"),
        }
    }
}
#[test]
fn typed_surface_ties_cancellation_signed_zero_and_fatal_faults() {
    let one = PcuF16Bits::from_bits(0x3c00);
    assert_eq!(
        one.pcu_checked_add(PcuF16Bits::from_bits(0x1000))
            .unwrap()
            .to_bits(),
        0x3c00
    );
    assert_eq!(
        PcuF16Bits::from_bits(0x3c01)
            .pcu_checked_add(PcuF16Bits::from_bits(0x1000))
            .unwrap()
            .to_bits(),
        0x3c02
    );
    assert_eq!(one.pcu_checked_sub(one).unwrap().to_bits(), 0);
    assert_eq!(
        PcuF16Bits::from_bits(0x8000)
            .pcu_checked_mul(one)
            .unwrap()
            .to_bits(),
        0x8000
    );
    assert_eq!(
        one.pcu_checked_div(PcuF16Bits::from_bits(0)),
        Err(PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(
        one.pcu_checked_add(PcuF16Bits::from_bits(0x7c00)),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    let bf = PcuBf16Bits::from_bits(0x3f80);
    assert_eq!(bf.pcu_checked_mul(bf).unwrap().to_bits(), 0x3f80);
    assert_eq!(bf.pcu_checked_div(bf).unwrap().to_bits(), 0x3f80);
}

#[test]
fn tininess_before_exponent_limited_rounding_even_when_packed_result_is_normal() {
    for format in [F16, BF16] {
        let one = u16::try_from(format.bias).unwrap() << format.fraction;
        let normal = 1 << format.fraction;
        match arithmetic(
            normal,
            one - 1,
            format,
            Operation::Mul,
            PcuFloatUnderflowPolicy::default(),
        ) {
            Err(PcuClampedError::Range(fault)) => {
                assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
                assert_eq!(fault.clamped_value(), normal);
            }
            other => {
                panic!("expected tiny exact unbounded significand with inexact packing: {other:?}")
            }
        }
        assert_eq!(
            arithmetic(
                normal,
                one - 1,
                format,
                Operation::Mul,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            Ok(normal)
        );
        // This product is slightly below minimum normal but its unbounded
        // destination-precision rounding is normal: it must succeed by default.
        assert_eq!(
            arithmetic(
                normal - 1,
                one + 1,
                format,
                Operation::Mul,
                PcuFloatUnderflowPolicy::default()
            ),
            Ok(normal)
        );
    }
}

#[test]
fn generic_trait_bridges_use_the_same_finite_input_contracts() {
    fn generic<T: crate::PcuClampedFloat + core::fmt::Debug + Eq>(one: T, zero: T) {
        assert_eq!(
            <T as crate::PcuCheckedFloat>::pcu_checked_add(one, zero),
            Ok(one)
        );
        assert_eq!(
            <T as crate::PcuCheckedFloat>::pcu_checked_sub(one, one),
            Ok(zero)
        );
        assert_eq!(
            <T as crate::PcuCheckedFloat>::pcu_checked_mul(one, one),
            Ok(one)
        );
        assert_eq!(
            <T as crate::PcuCheckedFloat>::pcu_checked_div(one, one),
            Ok(one)
        );
        assert_eq!(
            <T as crate::PcuClampedFloat>::pcu_clamped_add(one, zero),
            Ok(one)
        );
        assert_eq!(
            <T as crate::PcuClampedFloat>::pcu_clamped_relu(one),
            Ok(one)
        );
        assert_eq!(
            <T as crate::PcuCheckedFloat>::pcu_checked_relu_backward_with_policy(
                one,
                one,
                PcuFloatUnderflowPolicy::default()
            ),
            Ok(one)
        );
        assert_eq!(
            <T as crate::PcuClampedFloat>::pcu_clamped_neg(
                <T as crate::PcuCheckedFloat>::pcu_checked_neg(one).unwrap()
            ),
            Ok(one)
        );
    }
    generic(PcuF16Bits::from_bits(0x3c00), PcuF16Bits::from_bits(0));
    generic(PcuBf16Bits::from_bits(0x3f80), PcuBf16Bits::from_bits(0));
}
