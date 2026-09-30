use super::PcuCheckedFloatConversion;
use super::PcuCheckedFloatWidening;
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind as Fault,
    PcuFloatUnderflowPolicy as Policy,
};

fn check(value: f64, expected: Result<u32, Fault>, policy: Policy) {
    assert_eq!(
        value
            .pcu_checked_to_f32_with_policy(policy)
            .map(f32::to_bits),
        expected,
        "conversion of {value:?} with {policy:?}"
    );
}

#[test]
fn conversion_handles_specials_overflow_signed_zero_and_inexact_rounding() {
    check(
        f64::INFINITY,
        Err(Fault::InvalidFloatingOperand),
        Policy::AllowGradualUnderflow,
    );
    check(
        f64::NAN,
        Err(Fault::InvalidFloatingOperand),
        Policy::default(),
    );
    check(f64::MAX, Err(Fault::ArithmeticOverflow), Policy::default());
    check(-0.0, Ok(0x8000_0000), Policy::RejectSubnormalResult);
    // Ordinary precision loss is expected conversion rounding, not an error.
    check(1.0 + f64::EPSILON, Ok(1.0_f32.to_bits()), Policy::default());
}

#[test]
fn conversion_obeys_underflow_policy_and_preserves_exact_subnormals() {
    let min_subnormal_f32 = f64::from(f32::from_bits(1));
    check(min_subnormal_f32, Ok(1), Policy::default());
    check(
        min_subnormal_f32,
        Err(Fault::ArithmeticUnderflow),
        Policy::RejectSubnormalResult,
    );
    check(min_subnormal_f32, Ok(1), Policy::AllowGradualUnderflow);

    let half_min_subnormal = f64::from_bits(873_u64 << 52);
    check(
        half_min_subnormal,
        Err(Fault::ArithmeticUnderflow),
        Policy::default(),
    );
    check(half_min_subnormal, Ok(0), Policy::AllowGradualUnderflow);
    check(
        -half_min_subnormal,
        Ok(0x8000_0000),
        Policy::AllowGradualUnderflow,
    );
    // Ties between the first and second subnormal round to the even second one.
    check(
        half_min_subnormal * 3.0,
        Ok(2),
        Policy::AllowGradualUnderflow,
    );
}

#[test]
fn normal_and_subnormal_halfway_values_use_ties_to_even() {
    let normal_midpoint = 1.0 + 2.0_f64.powi(-24);
    check(normal_midpoint, Ok(1.0_f32.to_bits()), Policy::default());
    check(
        -normal_midpoint,
        Ok((-1.0_f32).to_bits()),
        Policy::default(),
    );
    check(
        f64::from_bits(normal_midpoint.to_bits() + 1),
        Ok((1.0_f32 + f32::EPSILON).to_bits()),
        Policy::default(),
    );

    let min_normal = 2.0_f64.powi(-126);
    let largest_subnormal = f64::from(f32::from_bits(0x007f_ffff));
    // Tie between the last value with exponent -127 at binary32 precision and min-normal.
    let exponent_midpoint = min_normal - 2.0_f64.powi(-151);
    check(
        largest_subnormal,
        Ok(0x007f_ffff),
        Policy::AllowGradualUnderflow,
    );
    check(min_normal, Ok(0x0080_0000), Policy::default());
    // At the tie, RN-even selects minimum normal, so the unbounded rounded exponent is not tiny.
    check(exponent_midpoint, Ok(0x0080_0000), Policy::default());
    check(
        exponent_midpoint,
        Ok(0x0080_0000),
        Policy::AllowGradualUnderflow,
    );
    check(
        f64::from_bits(exponent_midpoint.to_bits() - 1),
        Err(Fault::ArithmeticUnderflow),
        Policy::default(),
    );
    check(
        f64::from_bits(exponent_midpoint.to_bits() + 1),
        Ok(0x0080_0000),
        Policy::default(),
    );
}

#[test]
fn overflow_midpoint_rounds_to_infinity_and_faults() {
    let midpoint = 2.0_f64.powi(128) - 2.0_f64.powi(103);
    let below = f64::from_bits(midpoint.to_bits() - 1);
    let above = f64::from_bits(midpoint.to_bits() + 1);
    check(below, Ok(f32::MAX.to_bits()), Policy::default());
    check(midpoint, Err(Fault::ArithmeticOverflow), Policy::default());
    check(above, Err(Fault::ArithmeticOverflow), Policy::default());
}

#[test]
fn smallest_binary64_subnormals_preserve_signed_zero_when_gradual() {
    let smallest = f64::from_bits(1);
    check(smallest, Err(Fault::ArithmeticUnderflow), Policy::default());
    check(
        -smallest,
        Err(Fault::ArithmeticUnderflow),
        Policy::default(),
    );
    check(smallest, Ok(0), Policy::AllowGradualUnderflow);
    check(-smallest, Ok(0x8000_0000), Policy::AllowGradualUnderflow);
}

#[test]
#[allow(clippy::cast_possible_truncation)] // The host IEEE cast is the differential oracle.
fn deterministic_finite_differential_matches_ieee_cast() {
    let mut state = 0x6a09_e667_f3bc_c909_u64;
    for _ in 0..20_000 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f64::from_bits(state);
        if !value.is_finite() {
            continue;
        }
        let expected = value as f32;
        let actual = value.pcu_checked_to_f32_with_policy(Policy::AllowGradualUnderflow);
        if expected.is_infinite() {
            assert_eq!(actual, Err(Fault::ArithmeticOverflow));
        } else {
            assert_eq!(
                actual.map(f32::to_bits),
                Ok(expected.to_bits()),
                "{value:?}"
            );
        }
    }
}

#[test]
fn checked_f32_to_f64_widens_finite_values_exactly_and_preserves_signed_zero() {
    assert_eq!(0.0_f32.pcu_checked_to_f64().unwrap().to_bits(), 0);
    assert_eq!(
        (-0.0_f32).pcu_checked_to_f64().unwrap().to_bits(),
        0x8000_0000_0000_0000
    );
    for bits in [1_u32, 0x007f_ffff, 0x0080_0000, 0x3f80_0000, 0x7f7f_ffff] {
        let value = f32::from_bits(bits);
        let widened = value.pcu_checked_to_f64().expect("finite F32 widens");
        assert_eq!(widened.to_bits(), f64::from(value).to_bits());
        if value.is_subnormal() {
            assert!(widened.is_normal(), "{bits:#010x}");
        }
    }
}

#[test]
fn checked_f32_to_f64_rejects_nonfinite_values() {
    assert_eq!(
        f32::INFINITY.pcu_checked_to_f64(),
        Err(Fault::InvalidFloatingOperand)
    );
    assert_eq!(
        f32::from_bits(0x7fc0_1234).pcu_checked_to_f64(),
        Err(Fault::InvalidFloatingOperand)
    );
    assert_eq!(
        f32::from_bits(0x7f80_1234).pcu_checked_to_f64(),
        Err(Fault::InvalidFloatingOperand)
    );
}

#[test]
fn deterministic_finite_f32_widening_matches_ieee_cast_bits() {
    let mut state = 0x9e37_79b9_u32;
    for _ in 0..20_000 {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        let value = f32::from_bits(state);
        if value.is_finite() {
            assert_eq!(
                value.pcu_checked_to_f64().map(f64::to_bits),
                Ok(f64::from(value).to_bits()),
                "{value:?}"
            );
        }
    }
}

#[test]
fn clamped_narrowing_retains_observable_signed_overflow_payloads() {
    use super::PcuClampedFloatConversion;
    use crate::PcuClampedError;

    for value in [f64::MAX, -f64::MAX] {
        let Err(PcuClampedError::Range(fault)) = value.pcu_clamped_to_f32() else {
            panic!("overflow must remain an observable recoverable error");
        };
        assert_eq!(fault.kind(), Fault::ArithmeticOverflow);
        let expected = if value.is_sign_negative() {
            -f32::MAX
        } else {
            f32::MAX
        };
        assert_eq!(fault.clamped_value().to_bits(), expected.to_bits());
        assert_eq!(value.pcu_checked_to_f32(), Err(Fault::ArithmeticOverflow));
    }
}

#[test]
fn clamped_narrowing_retains_gradual_underflow_and_fatal_distinction() {
    use super::PcuClampedFloatConversion;
    use crate::PcuClampedError;

    for value in [2.0_f64.powi(-150), -2.0_f64.powi(-150)] {
        let Err(PcuClampedError::Range(fault)) = value.pcu_clamped_to_f32() else {
            panic!("tiny inexact conversion must retain its range fault");
        };
        assert_eq!(fault.kind(), Fault::ArithmeticUnderflow);
        let expected = if value.is_sign_negative() {
            -0.0_f32
        } else {
            0.0_f32
        };
        assert_eq!(fault.clamped_value().to_bits(), expected.to_bits());
    }
    let exact = 2.0_f64.powi(-149);
    assert_eq!(exact.pcu_clamped_to_f32().unwrap().to_bits(), 1);
    let Err(PcuClampedError::Range(fault)) =
        exact.pcu_clamped_to_f32_with_policy(Policy::RejectSubnormalResult)
    else {
        panic!("stricter PCU policy must preserve its exact subnormal payload");
    };
    assert_eq!(fault.clamped_value().to_bits(), 1);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(matches!(
            value.pcu_clamped_to_f32(),
            Err(PcuClampedError::Fatal(Fault::InvalidFloatingOperand))
        ));
    }
}

#[test]
fn clamped_narrowing_preserves_checked_status_and_independent_value_oracle() {
    use super::PcuClampedFloatConversion;
    use crate::PcuClampedError;

    let mut bits = 0x6a09_e667_f3bc_c909_u64;
    for _ in 0..20_000 {
        bits = bits.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let value = f64::from_bits(bits);
        let checked = value.pcu_checked_to_f32();
        match value.pcu_clamped_to_f32() {
            Ok(result) => assert_eq!(checked.map(f32::to_bits), Ok(result.to_bits())),
            Err(PcuClampedError::Fatal(kind)) => {
                assert_eq!(kind, Fault::InvalidFloatingOperand);
                assert_eq!(checked, Err(kind));
            }
            Err(PcuClampedError::Range(fault)) => {
                let kind = fault.kind();
                assert_eq!(checked, Err(kind));
                let result = fault.clamped_value();
                let expected = match kind {
                    Fault::ArithmeticOverflow => {
                        if value.is_sign_negative() {
                            -f32::MAX
                        } else {
                            f32::MAX
                        }
                    }
                    // Independently use the host conversion only for payload bits, not
                    // IEEE 754-2019 7.5 classification: stored bits cannot prove inexactness.
                    #[allow(clippy::cast_possible_truncation)]
                    Fault::ArithmeticUnderflow => value as f32,
                    other => panic!("unexpected recoverable conversion fault: {other:?}"),
                };
                assert_eq!(result.to_bits(), expected.to_bits());
            }
        }
    }
}
