use super::PcuCheckedFloat;
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};

#[test]
fn rejects_non_finite_inputs_by_bits() {
    for bits in [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345] {
        let value = f32::from_bits(bits);
        assert_eq!(
            value.pcu_checked_add(1.0),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
        assert_eq!(
            value.pcu_checked_sub(1.0),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
        assert_eq!(
            value.pcu_checked_mul(1.0),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }
}

#[test]
fn relu_is_a_bitwise_finite_selection_with_canonical_positive_zero() {
    let patterns = [
        0x0000_0000,
        0x8000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0x7f7f_ffff,
        0x8000_0001,
        0xbf80_0000,
        0xff7f_ffff,
    ];
    for bits in patterns {
        let value = f32::from_bits(bits);
        let expected = if bits & 0x8000_0000 != 0 || bits.trailing_zeros() >= 31 {
            0
        } else {
            bits
        };
        assert_eq!(value.pcu_checked_relu().unwrap().to_bits(), expected);
    }
    for bits in [0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xffc1_2345] {
        assert_eq!(
            f32::from_bits(bits).pcu_checked_relu(),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }
    assert_eq!(
        f32::from_bits(1)
            .pcu_checked_relu_with_policy(PcuFloatUnderflowPolicy::RejectSubnormalResult),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        f32::from_bits(1)
            .pcu_checked_relu_with_policy(PcuFloatUnderflowPolicy::AllowGradualUnderflow),
        Ok(f32::from_bits(1))
    );
}

#[test]
fn finite_results_and_overflow_are_checked() {
    assert_eq!(1.5_f32.pcu_checked_add(2.25), Ok(3.75));
    assert_eq!(1.5_f32.pcu_checked_sub(2.25), Ok(-0.75));
    assert_eq!(1.5_f32.pcu_checked_mul(2.25), Ok(3.375));
    assert_eq!(7.5_f32.pcu_checked_div(2.5), Ok(3.0));
    assert_eq!(1.0_f32.pcu_checked_div(3.0), Ok(1.0 / 3.0));
    assert_eq!(
        f32::MAX.pcu_checked_mul(2.0),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        f32::MAX.pcu_checked_add(f32::MAX),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
}

#[test]
fn division_normalizes_subnormals_and_rounds_tiny_results_to_even() {
    let min_subnormal = f32::from_bits(1);
    let min_normal = f32::MIN_POSITIVE;
    assert_eq!(
        min_subnormal
            .pcu_checked_div_with_policy(min_normal, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
            .to_bits(),
        (2.0_f32.powi(-23)).to_bits()
    );
    assert_eq!(
        f32::from_bits(min_normal.to_bits() - 1)
            .pcu_checked_div_with_policy(min_normal, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
            .to_bits(),
        (1.0_f32 - 2.0_f32.powi(-23)).to_bits()
    );
    assert_eq!(
        min_normal
            .pcu_checked_div_with_policy(
                min_subnormal,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            )
            .unwrap()
            .to_bits(),
        (2.0_f32.powi(23)).to_bits()
    );
    assert_eq!(
        min_subnormal
            .pcu_checked_div_with_policy(
                f32::from_bits(min_normal.to_bits() + 1),
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            )
            .unwrap()
            .to_bits(),
        (2.0_f32.powi(-23)).to_bits() - 2
    );
    assert_eq!(
        f32::from_bits(3)
            .pcu_checked_div_with_policy(2.0, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
            .unwrap()
            .to_bits(),
        2
    );
    assert_eq!(
        f32::from_bits(5)
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
fn division_error_order_and_signed_zero_are_checked() {
    for numerator in [0.0_f32, -0.0] {
        for denominator in [0.0_f32, -0.0] {
            assert_eq!(
                numerator.pcu_checked_div(denominator),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
        }
    }
    assert_eq!(
        f32::INFINITY.pcu_checked_div(0.0),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_eq!(
        0.0_f32.pcu_checked_div(f32::NAN),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
    assert_eq!(
        (-0.0_f32).pcu_checked_div(2.0).unwrap().to_bits(),
        0x8000_0000
    );
    assert_eq!(
        f32::MAX.pcu_checked_div(f32::from_bits(1)),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
}

#[test]
fn division_matches_host_rne_for_deterministic_finite_pairs() {
    let mut state = 0x91e1_0da5_u32;
    for _ in 0..50_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let left_bits = state;
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let right_bits = state;
        if left_bits & 0x7f80_0000 == 0x7f80_0000
            || right_bits & 0x7f80_0000 == 0x7f80_0000
            || right_bits << 1 == 0
        {
            continue;
        }
        let left = f32::from_bits(left_bits);
        let right = f32::from_bits(right_bits);
        let expected = left / right;
        let actual =
            left.pcu_checked_div_with_policy(right, PcuFloatUnderflowPolicy::AllowGradualUnderflow);
        if expected.is_infinite() {
            assert_eq!(actual, Err(PcuExecutionFaultKind::ArithmeticOverflow));
        } else {
            assert_eq!(actual.map(f32::to_bits), Ok(expected.to_bits()));
        }
    }
}

#[test]
fn underflow_policies_distinguish_exact_subnormals() {
    let exact = f32::from_bits(1).pcu_checked_mul(1.0);
    assert_eq!(exact, Ok(f32::from_bits(1)));
    assert_eq!(
        f32::from_bits(1).pcu_checked_mul(0.5),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        f32::from_bits(1)
            .pcu_checked_mul_with_policy(0.5, PcuFloatUnderflowPolicy::AllowGradualUnderflow),
        Ok(0.0)
    );
    assert_eq!(
        f32::from_bits(1)
            .pcu_checked_mul_with_policy(1.0, PcuFloatUnderflowPolicy::RejectSubnormalResult),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
}

#[test]
fn tininess_uses_unbounded_exponent_rounding_before_subnormal_pack() {
    let just_below_one = f32::from_bits(0x3f7f_ffff);
    assert_eq!(
        f32::MIN_POSITIVE.pcu_checked_mul(just_below_one),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        f32::MIN_POSITIVE.pcu_checked_mul_with_policy(
            just_below_one,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ),
        Ok(f32::MIN_POSITIVE)
    );
}

#[test]
fn signed_zero_is_preserved_for_exact_zero_results() {
    let negative_zero = f32::from_bits(0x8000_0000);
    let positive_zero = 0.0_f32;
    assert_eq!(
        negative_zero
            .pcu_checked_add(positive_zero)
            .unwrap()
            .to_bits(),
        0
    );
    assert_eq!(
        negative_zero
            .pcu_checked_add(negative_zero)
            .unwrap()
            .to_bits(),
        0x8000_0000
    );
    assert_eq!(
        positive_zero
            .pcu_checked_add(negative_zero)
            .unwrap()
            .to_bits(),
        0
    );
    assert_eq!(
        negative_zero
            .pcu_checked_sub(positive_zero)
            .unwrap()
            .to_bits(),
        0x8000_0000
    );
    assert_eq!(
        positive_zero
            .pcu_checked_sub(negative_zero)
            .unwrap()
            .to_bits(),
        0
    );
    assert_eq!(
        negative_zero.pcu_checked_mul(2.0).unwrap().to_bits(),
        0x8000_0000
    );
}

#[test]
fn zero_identity_obeys_reject_subnormal_policy() {
    let least_subnormal = f32::from_bits(1);
    assert_eq!(
        0.0_f32.pcu_checked_add_with_policy(
            least_subnormal,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    assert_eq!(
        least_subnormal
            .pcu_checked_add_with_policy(0.0, PcuFloatUnderflowPolicy::RejectSubnormalResult,),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
}

#[test]
fn randomized_finite_bits_match_host_binary32_for_all_operations() {
    let mut state = 0x9e37_79b9_u32;
    for _ in 0..100_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let left = f32::from_bits(state);
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let right = f32::from_bits(state);
        if !left.is_finite() || !right.is_finite() {
            continue;
        }

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
            match actual {
                Ok(actual) if expected.is_finite() => {
                    assert_eq!(actual.to_bits(), expected.to_bits());
                }
                Err(PcuExecutionFaultKind::ArithmeticOverflow) if expected.is_infinite() => {}
                other => panic!("unexpected checked result {other:?} for {left:?}, {right:?}"),
            }
        }
    }
}
