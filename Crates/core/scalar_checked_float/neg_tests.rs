//! Sign-bit semantics and policy/recovery parity for the checked negation oracle.
#[rustfmt::skip]
use super::{
    PcuCheckedFloat,
    PcuClampedFloat,
    PcuClampedError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};
#[test]
fn f32_neg_preserves_signed_zero_and_exact_finite_bits() {
    for bits in [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x7f7f_ffff,
        0xff7f_ffff,
    ] {
        assert_eq!(
            f32::from_bits(bits).pcu_checked_neg().unwrap().to_bits(),
            bits ^ 0x8000_0000
        );
    }
    for bits in [0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xff80_0001] {
        assert_eq!(
            f32::from_bits(bits).pcu_checked_neg(),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }
}
#[test]
fn f64_neg_preserves_signed_zero_and_exact_finite_bits() {
    for bits in [
        0,
        1 << 63,
        1,
        (1 << 63) | 1,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
    ] {
        assert_eq!(
            f64::from_bits(bits).pcu_checked_neg().unwrap().to_bits(),
            bits ^ (1 << 63)
        );
    }
    for bits in [
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0000,
        0xfff0_0000_0000_0001,
    ] {
        assert_eq!(
            f64::from_bits(bits).pcu_checked_neg(),
            Err(PcuExecutionFaultKind::InvalidFloatingOperand)
        );
    }
}
#[test]
fn neg_subnormal_rejection_retains_exact_signed_recovery() {
    let policy = PcuFloatUnderflowPolicy::RejectSubnormalResult;
    for bits in [1, 0x8000_0001] {
        let value = f32::from_bits(bits);
        assert_eq!(
            value.pcu_checked_neg_with_policy(policy),
            Err(PcuExecutionFaultKind::ArithmeticUnderflow)
        );
        let PcuClampedError::Range(fault) = value.pcu_clamped_neg_with_policy(policy).unwrap_err()
        else {
            panic!("range recovery required")
        };
        assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
        assert_eq!(fault.clamped_value().to_bits(), bits ^ 0x8000_0000);
    }
    for bits in [1, (1 << 63) | 1] {
        let value = f64::from_bits(bits);
        assert_eq!(
            value.pcu_checked_neg_with_policy(policy),
            Err(PcuExecutionFaultKind::ArithmeticUnderflow)
        );
        let PcuClampedError::Range(fault) = value.pcu_clamped_neg_with_policy(policy).unwrap_err()
        else {
            panic!("range recovery required")
        };
        assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
        assert_eq!(fault.clamped_value().to_bits(), bits ^ (1 << 63));
    }
    assert!(matches!(
        f64::INFINITY.pcu_clamped_neg(),
        Err(PcuClampedError::Fatal(
            PcuExecutionFaultKind::InvalidFloatingOperand
        ))
    ));
}
