use super::*;

#[test]
fn every_half_encoding_is_classified_and_exactly_selected_without_float_hardware() {
    macro_rules! check {
        ($ty:ty, $fraction:expr, $exp_mask:expr) => {
            for bits in 0..=u16::MAX {
                let value = <$ty>::from_bits(bits);
                let exponent = (bits >> $fraction) & $exp_mask;
                let finite = exponent != $exp_mask;
                if !finite {
                    assert_eq!(
                        value.pcu_checked_neg(),
                        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
                    );
                    assert_eq!(
                        value.pcu_checked_relu(),
                        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
                    );
                } else {
                    assert_eq!(value.pcu_checked_neg().unwrap().to_bits(), bits ^ 0x8000);
                    let expected = if bits & 0x8000 != 0 { 0 } else { bits };
                    assert_eq!(value.pcu_checked_relu().unwrap().to_bits(), expected);
                    let tiny = exponent == 0 && bits & ((1 << $fraction) - 1) != 0;
                    let strict = value.pcu_checked_neg_with_policy(
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    );
                    assert_eq!(strict.is_err(), tiny);
                }
            }
        };
    }
    check!(PcuF16Bits, 10, 31);
    check!(PcuBf16Bits, 7, 255);
}

#[test]
fn every_fp8_encoding_obeys_its_own_special_value_and_selection_contract() {
    for bits in 0..=u8::MAX {
        let finite4 = bits & 0x7f != 0x7f;
        let finite5 = bits & 0x7f < 0x7c;
        let four = PcuF8E4M3FnBits::from_bits(bits);
        let five = PcuF8E5M2Bits::from_bits(bits);
        assert_eq!(
            four.pcu_checked_neg().map(PcuF8E4M3FnBits::to_bits),
            if finite4 {
                Ok(bits ^ 0x80)
            } else {
                Err(PcuExecutionFaultKind::InvalidFloatingOperand)
            }
        );
        assert_eq!(
            five.pcu_checked_neg().map(PcuF8E5M2Bits::to_bits),
            if finite5 {
                Ok(bits ^ 0x80)
            } else {
                Err(PcuExecutionFaultKind::InvalidFloatingOperand)
            }
        );
    }
}

#[test]
fn wide_selection_preserves_low_bits_and_lifts_clamped_tiny_payload() {
    let tiny = PcuF256Bits::from_limbs_le([1, 0, 0, 0]);
    assert_eq!(
        tiny.pcu_checked_neg().unwrap().to_limbs_le(),
        [1, 0, 0, 1 << 63]
    );
    assert_eq!(tiny.pcu_checked_relu(), Ok(tiny));
    let fault = tiny
        .pcu_clamped_neg_with_policy(PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap_err();
    assert_eq!(fault.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    let PcuClampedError::Range(fault) = fault else {
        panic!("expected range payload")
    };
    assert_eq!(fault.clamped_value().to_limbs_le(), [1, 0, 0, 1 << 63]);
    let positive = PcuF128Bits::from_limbs_le([0, 0x3fff_0000_0000_0000]);
    let negative_zero = PcuF128Bits::from_limbs_le([0, 1 << 63]);
    assert_eq!(
        positive.pcu_checked_relu_backward_with_policy(
            negative_zero,
            PcuFloatUnderflowPolicy::default()
        ),
        Ok(negative_zero)
    );
    assert_eq!(
        negative_zero.pcu_checked_relu().unwrap().to_limbs_le(),
        [0, 0]
    );
    let nan = PcuF128Bits::from_limbs_le([1, 0x7fff_0000_0000_0000]);
    assert_eq!(
        negative_zero
            .pcu_checked_relu_backward_with_policy(nan, PcuFloatUnderflowPolicy::default()),
        Err(PcuExecutionFaultKind::InvalidFloatingOperand)
    );
}
