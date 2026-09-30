use super::PcuCheckedFloat;

#[test]
fn relu_is_a_bitwise_finite_selection_with_canonical_positive_zero() {
    let patterns = [
        0x0000_0000_0000_0000,
        0x8000_0000_0000_0000,
        0x0000_0000_0000_0001,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        0x8000_0000_0000_0001,
        0xbff0_0000_0000_0000,
        0xffef_ffff_ffff_ffff,
    ];
    for bits in patterns {
        let value = f64::from_bits(bits);
        let expected = if bits & (1 << 63) != 0 || bits & !(1 << 63) == 0 {
            0
        } else {
            bits
        };
        assert_eq!(value.pcu_checked_relu().unwrap().to_bits(), expected);
    }
    for bits in [
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff0_0000_0000_0001,
        0xfff8_1234_5678_9abc,
    ] {
        assert_eq!(
            f64::from_bits(bits).pcu_checked_relu(),
            Err(crate::PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }
    assert_eq!(
        f64::from_bits(1)
            .pcu_checked_relu_with_policy(crate::PcuFloatUnderflowPolicy::RejectSubnormalResult),
        Err(crate::PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        f64::from_bits(1)
            .pcu_checked_relu_with_policy(crate::PcuFloatUnderflowPolicy::AllowGradualUnderflow),
        Ok(f64::from_bits(1))
    );
}

#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};

fn assert_bits(actual: Result<f64, PcuExecutionFaultKind>, expected: f64) {
    assert_eq!(actual.map(f64::to_bits), Ok(expected.to_bits()));
}

#[test]
fn binary64_add_sub_round_ties_to_even_and_preserve_zero_signs() {
    let half_ulp_at_one = 2.0_f64.powi(-53);
    assert_bits(1.0_f64.pcu_checked_add(half_ulp_at_one), 1.0);
    let one_and_half_ulps = 3.0_f64 * half_ulp_at_one;
    assert_bits(
        1.0_f64.pcu_checked_add(one_and_half_ulps),
        2.0_f64.mul_add(f64::EPSILON, 1.0),
    );

    assert_bits(1.0_f64.pcu_checked_add(-1.0), 0.0_f64);
    assert_bits(
        f64::from_bits(1_u64 << 63).pcu_checked_add(f64::from_bits(1_u64 << 63)),
        f64::from_bits(1_u64 << 63),
    );
    assert_bits(
        f64::from_bits(1_u64 << 63).pcu_checked_sub(0.0),
        f64::from_bits(1_u64 << 63),
    );
    assert_bits(
        0.0_f64.pcu_checked_add(f64::from_bits(1_u64 << 63)),
        0.0_f64,
    );
}

#[test]
fn binary64_multiply_preserves_signed_zero_and_exact_cancellation() {
    assert_bits(
        f64::from_bits(1_u64 << 63).pcu_checked_mul(3.0),
        f64::from_bits(1_u64 << 63),
    );
    assert_bits((-3.0_f64).pcu_checked_mul(0.0), f64::from_bits(1_u64 << 63));
    assert_bits(1.25_f64.pcu_checked_sub(1.25), 0.0_f64);
}

#[test]
fn binary64_underflow_policies_distinguish_exact_subnormal_and_tiny_inexact() {
    let min_normal = f64::MIN_POSITIVE;
    let min_subnormal = f64::from_bits(1);
    let exact_subnormal = min_normal * 0.5;
    assert_bits(min_normal.pcu_checked_mul(0.5), exact_subnormal);
    assert_bits(
        min_normal.pcu_checked_mul_with_policy(0.5, PcuFloatUnderflowPolicy::AllowGradualUnderflow),
        exact_subnormal,
    );
    assert_eq!(
        min_normal
            .pcu_checked_mul_with_policy(0.5, PcuFloatUnderflowPolicy::RejectSubnormalResult,),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );

    assert_eq!(
        min_subnormal.pcu_checked_mul_with_policy(0.5, PcuFloatUnderflowPolicy::IeeeAfterRounding,),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_bits(
        (-min_subnormal)
            .pcu_checked_mul_with_policy(0.5, PcuFloatUnderflowPolicy::AllowGradualUnderflow),
        f64::from_bits(1_u64 << 63),
    );

    // The exact halfway value is tiny at the unbounded exponent. Exponent-limited
    // packing then rounds inexactly to the minimum normal value.
    let predecessor_of_one = f64::from_bits(1.0_f64.to_bits() - 1);
    assert_eq!(
        min_normal.pcu_checked_mul(predecessor_of_one),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        min_normal.pcu_checked_mul_with_policy(
            predecessor_of_one,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_bits(
        min_normal.pcu_checked_mul_with_policy(
            predecessor_of_one,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        min_normal,
    );
}

#[test]
fn binary64_rejects_non_finite_operands_and_finite_overflow() {
    for invalid in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert_eq!(
            1.0_f64.pcu_checked_add(invalid),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
        assert_eq!(
            1.0_f64.pcu_checked_sub(invalid),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
        assert_eq!(
            1.0_f64.pcu_checked_mul(invalid),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
        assert_eq!(
            invalid.pcu_checked_mul(0.0),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }

    assert_eq!(
        f64::MAX.pcu_checked_add(f64::MAX),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        f64::MAX.pcu_checked_mul(2.0),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        f64::MAX.pcu_checked_sub(-f64::MAX),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
}

#[test]
fn binary64_division_normalizes_subnormals_and_rounds_tiny_results_to_even() {
    let min_subnormal = f64::from_bits(1);
    let min_normal = f64::MIN_POSITIVE;
    assert_bits(
        min_subnormal.pcu_checked_div_with_policy(
            min_normal,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        2.0_f64.powi(-52),
    );
    assert_bits(
        f64::from_bits(min_normal.to_bits() - 1).pcu_checked_div_with_policy(
            min_normal,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        1.0_f64 - 2.0_f64.powi(-52),
    );
    assert_bits(
        min_normal.pcu_checked_div_with_policy(
            min_subnormal,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        2.0_f64.powi(52),
    );
    assert_eq!(
        min_subnormal
            .pcu_checked_div_with_policy(
                f64::from_bits(min_normal.to_bits() + 1),
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            )
            .unwrap()
            .to_bits(),
        (2.0_f64.powi(-52)).to_bits() - 2
    );
    assert_eq!(
        f64::from_bits(3)
            .pcu_checked_div_with_policy(2.0, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
            .to_bits(),
        2
    );
    assert_eq!(
        f64::from_bits(5)
            .pcu_checked_div_with_policy(2.0, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
            .to_bits(),
        2
    );
    assert_eq!(
        min_subnormal
            .pcu_checked_div_with_policy(2.0, PcuFloatUnderflowPolicy::AllowGradualUnderflow,),
        Ok(0.0)
    );
    assert_eq!(
        min_subnormal.pcu_checked_div(2.0),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
}

#[test]
fn binary64_division_error_order_and_signed_zero_are_checked() {
    for numerator in [0.0_f64, -0.0] {
        for denominator in [0.0_f64, -0.0] {
            assert_eq!(
                numerator.pcu_checked_div(denominator),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
        }
    }
    assert_eq!(
        f64::INFINITY.pcu_checked_div(0.0),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_eq!(
        0.0_f64.pcu_checked_div(f64::NAN),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_eq!(
        (-0.0_f64).pcu_checked_div(2.0).unwrap().to_bits(),
        1_u64 << 63
    );
    assert_eq!(
        f64::MAX.pcu_checked_div(f64::from_bits(1)),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
}

#[test]
fn binary64_division_matches_host_rne_for_deterministic_finite_pairs() {
    let mut state = 0x91e1_0da5_7b3c_2f19_u64;
    for _ in 0..50_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let left_bits = state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let right_bits = state;
        if left_bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000
            || right_bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000
            || right_bits << 1 == 0
        {
            continue;
        }
        let left = f64::from_bits(left_bits);
        let right = f64::from_bits(right_bits);
        let expected = left / right;
        let actual =
            left.pcu_checked_div_with_policy(right, PcuFloatUnderflowPolicy::AllowGradualUnderflow);
        if expected.is_infinite() {
            assert_eq!(actual, Err(PcuExecutionFaultKind::ArithmeticOverflow));
        } else {
            assert_eq!(actual.map(f64::to_bits), Ok(expected.to_bits()));
        }
    }
}

#[test]
fn randomized_finite_binary64_results_match_host_rne_under_gradual_policy() {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    let mut next_finite = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let mut bits = state;
        if bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
            bits ^= 1_u64 << 52;
        }
        f64::from_bits(bits)
    };

    for _ in 0..50_000 {
        let left = next_finite();
        let right = next_finite();
        for (actual, expected) in [
            (
                left.pcu_checked_add_with_policy(
                    right,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ),
                left + right,
            ),
            (
                left.pcu_checked_sub_with_policy(
                    right,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ),
                left - right,
            ),
            (
                left.pcu_checked_mul_with_policy(
                    right,
                    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                ),
                left * right,
            ),
        ] {
            if expected.is_infinite() {
                assert_eq!(actual, Err(PcuExecutionFaultKind::ArithmeticOverflow));
            } else {
                assert_bits(actual, expected);
            }
        }
    }
}
