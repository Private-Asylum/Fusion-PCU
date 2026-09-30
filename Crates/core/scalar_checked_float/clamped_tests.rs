use super::{PcuCheckedFloat, PcuClampedFloat};
use crate::{PcuClampedError, PcuExecutionFaultKind, PcuFloatUnderflowPolicy};

fn assert_f32_matches(
    checked: Result<f32, PcuExecutionFaultKind>,
    clamped: Result<f32, PcuClampedError<f32>>,
) {
    match (checked, clamped) {
        (Ok(checked), Ok(clamped)) => assert_eq!(checked.to_bits(), clamped.to_bits()),
        (Err(checked), Err(clamped)) => assert_eq!(checked, clamped.kind()),
        _ => panic!("checked and clamped routes must agree on success versus fault"),
    }
}

fn assert_f64_matches(
    checked: Result<f64, PcuExecutionFaultKind>,
    clamped: Result<f64, PcuClampedError<f64>>,
) {
    match (checked, clamped) {
        (Ok(checked), Ok(clamped)) => assert_eq!(checked.to_bits(), clamped.to_bits()),
        (Err(checked), Err(clamped)) => assert_eq!(checked, clamped.kind()),
        _ => panic!("checked and clamped routes must agree on success versus fault"),
    }
}

#[test]
fn f32_clamped_routes_match_checked_add_sub_mul_div() {
    assert_f32_matches(1.25_f32.pcu_checked_add(2.5), 1.25_f32.pcu_clamped_add(2.5));
    assert_f32_matches(1.25_f32.pcu_checked_sub(2.5), 1.25_f32.pcu_clamped_sub(2.5));
    assert_f32_matches(1.25_f32.pcu_checked_mul(2.5), 1.25_f32.pcu_clamped_mul(2.5));
    assert_f32_matches(1.25_f32.pcu_checked_div(2.5), 1.25_f32.pcu_clamped_div(2.5));
    assert_f32_matches(
        f32::MAX.pcu_checked_add(f32::MAX),
        f32::MAX.pcu_clamped_add(f32::MAX),
    );
    assert_f32_matches(
        f32::MAX.pcu_checked_sub(-f32::MAX),
        f32::MAX.pcu_clamped_sub(-f32::MAX),
    );
    assert_f32_matches(f32::MAX.pcu_checked_mul(2.0), f32::MAX.pcu_clamped_mul(2.0));
    assert_f32_matches(
        f32::MAX.pcu_checked_div(f32::from_bits(1)),
        f32::MAX.pcu_clamped_div(f32::from_bits(1)),
    );
}

#[test]
fn f64_clamped_routes_match_checked_add_sub_mul_div() {
    assert_f64_matches(1.25_f64.pcu_checked_add(2.5), 1.25_f64.pcu_clamped_add(2.5));
    assert_f64_matches(1.25_f64.pcu_checked_sub(2.5), 1.25_f64.pcu_clamped_sub(2.5));
    assert_f64_matches(1.25_f64.pcu_checked_mul(2.5), 1.25_f64.pcu_clamped_mul(2.5));
    assert_f64_matches(1.25_f64.pcu_checked_div(2.5), 1.25_f64.pcu_clamped_div(2.5));
    assert_f64_matches(
        f64::MAX.pcu_checked_add(f64::MAX),
        f64::MAX.pcu_clamped_add(f64::MAX),
    );
    assert_f64_matches(
        f64::MAX.pcu_checked_sub(-f64::MAX),
        f64::MAX.pcu_clamped_sub(-f64::MAX),
    );
    assert_f64_matches(f64::MAX.pcu_checked_mul(2.0), f64::MAX.pcu_clamped_mul(2.0));
    assert_f64_matches(
        f64::MAX.pcu_checked_div(f64::from_bits(1)),
        f64::MAX.pcu_clamped_div(f64::from_bits(1)),
    );
}

#[test]
fn clamped_relu_has_identical_bits_and_retains_rejected_subnormal() {
    for bits in [
        0,
        0x8000_0000,
        1,
        0x007f_ffff,
        0x0080_0000,
        0x3f80_0000,
        0xbf80_0000,
    ] {
        let value = f32::from_bits(bits);
        assert_f32_matches(value.pcu_checked_relu(), value.pcu_clamped_relu());
    }
    let f32_fault = f32::from_bits(1)
        .pcu_clamped_relu_with_policy(PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap_err();
    assert_eq!(f32_fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    let PcuClampedError::Range(f32_fault) = f32_fault else {
        panic!("a policy-rejected subnormal has a recovery result");
    };
    assert_eq!(f32_fault.clamped_value().to_bits(), 1);

    for bits in [
        0,
        1 << 63,
        1,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0xbff0_0000_0000_0000,
    ] {
        let value = f64::from_bits(bits);
        assert_f64_matches(value.pcu_checked_relu(), value.pcu_clamped_relu());
    }
    let f64_fault = f64::from_bits(1)
        .pcu_clamped_relu_with_policy(PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap_err();
    assert_eq!(f64_fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    let PcuClampedError::Range(f64_fault) = f64_fault else {
        panic!("a policy-rejected subnormal has a recovery result");
    };
    assert_eq!(f64_fault.clamped_value().to_bits(), 1);
}

#[test]
fn f32_range_errors_retain_signed_finite_endpoint_and_underflow_rounding() {
    let positive = f32::MAX.pcu_clamped_mul(2.0).unwrap_err();
    assert_eq!(positive.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    let PcuClampedError::Range(fault) = positive else {
        panic!("overflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), f32::MAX.to_bits());

    let negative = (-f32::MAX).pcu_clamped_mul(2.0).unwrap_err();
    assert_eq!(negative.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    let PcuClampedError::Range(fault) = negative else {
        panic!("negative overflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), (-f32::MAX).to_bits());

    let tiny = f32::from_bits(1).pcu_clamped_div(2.0).unwrap_err();
    assert_eq!(tiny.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    let PcuClampedError::Range(fault) = tiny else {
        panic!("underflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), 0);
    let negative_tiny = (-f32::from_bits(1)).pcu_clamped_div(2.0).unwrap_err();
    let PcuClampedError::Range(fault) = negative_tiny else {
        panic!("negative underflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), 0x8000_0000);
    assert_eq!(
        f32::from_bits(1).pcu_checked_div(2.0),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );

    let exact_subnormal = f32::from_bits(1).pcu_clamped_mul(1.0).unwrap();
    assert_eq!(exact_subnormal.to_bits(), 1);
    let rejected_exact = f32::from_bits(1)
        .pcu_clamped_mul_with_policy(1.0, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap_err();
    assert_eq!(
        rejected_exact.kind(),
        PcuExecutionFaultKind::ArithmeticUnderflow
    );
    let PcuClampedError::Range(fault) = rejected_exact else {
        panic!("policy-rejected subnormal must retain a range payload")
    };
    assert_eq!(fault.clamped_value().to_bits(), 1);

    assert_eq!(
        f32::from_bits(1)
            .pcu_clamped_div_with_policy(2.0, PcuFloatUnderflowPolicy::AllowGradualUnderflow,),
        Ok(0.0)
    );
}

#[test]
fn f64_range_errors_retain_signed_finite_endpoint_and_underflow_rounding() {
    let positive = f64::MAX.pcu_clamped_mul(2.0).unwrap_err();
    assert_eq!(positive.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    let PcuClampedError::Range(fault) = positive else {
        panic!("overflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), f64::MAX.to_bits());

    let negative = (-f64::MAX).pcu_clamped_mul(2.0).unwrap_err();
    assert_eq!(negative.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    let PcuClampedError::Range(fault) = negative else {
        panic!("negative overflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), (-f64::MAX).to_bits());

    let tiny = f64::from_bits(1).pcu_clamped_div(2.0).unwrap_err();
    assert_eq!(tiny.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    let PcuClampedError::Range(fault) = tiny else {
        panic!("underflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), 0);
    let negative_tiny = (-f64::from_bits(1)).pcu_clamped_div(2.0).unwrap_err();
    let PcuClampedError::Range(fault) = negative_tiny else {
        panic!("negative underflow must be a recoverable range error")
    };
    assert_eq!(fault.clamped_value().to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(
        f64::from_bits(1).pcu_checked_div(2.0),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );

    let exact_subnormal = f64::from_bits(1).pcu_clamped_mul(1.0).unwrap();
    assert_eq!(exact_subnormal.to_bits(), 1);
    let rejected_exact = f64::from_bits(1)
        .pcu_clamped_mul_with_policy(1.0, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap_err();
    assert_eq!(
        rejected_exact.kind(),
        PcuExecutionFaultKind::ArithmeticUnderflow
    );
    let PcuClampedError::Range(fault) = rejected_exact else {
        panic!("policy-rejected subnormal must retain a range payload")
    };
    assert_eq!(fault.clamped_value().to_bits(), 1);

    assert_eq!(
        f64::from_bits(1)
            .pcu_clamped_div_with_policy(2.0, PcuFloatUnderflowPolicy::AllowGradualUnderflow,),
        Ok(0.0)
    );
}

#[test]
fn clamped_float_keeps_fatal_faults_without_payload() {
    let zero_division = 1.0_f32.pcu_clamped_div(0.0).unwrap_err();
    assert_eq!(zero_division.kind(), PcuExecutionFaultKind::DivideByZero);
    assert!(matches!(zero_division, PcuClampedError::Fatal(_)));

    let nonfinite = f64::INFINITY.pcu_clamped_add(1.0).unwrap_err();
    assert_eq!(
        nonfinite.kind(),
        PcuExecutionFaultKind::InvalidFloatingOperand
    );
    assert!(matches!(nonfinite, PcuClampedError::Fatal(_)));
}
