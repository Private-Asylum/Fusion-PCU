use super::*;

#[test]
fn carries_borrows_and_signed_range_faults_cross_all_limbs() {
    let one = PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    let prefix = PcuU512::from_limbs_le([u64::MAX, u64::MAX, 0, 0, 0, 0, 0, 0]);
    let carried = prefix.pcu_checked_add(one).unwrap();
    assert_eq!(carried.to_limbs_le(), [0, 0, 1, 0, 0, 0, 0, 0]);
    assert_eq!(carried.pcu_checked_sub(one), Ok(prefix));
    assert_eq!(
        PcuU512::MAX.pcu_checked_add(one),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        PcuU512::ZERO.pcu_checked_sub(one),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    let one = PcuI512::from_limbs_le(one.to_limbs_le());
    assert_eq!(
        PcuI512::MAX.pcu_checked_add(one),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        PcuI512::MIN.pcu_checked_sub(one),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
    let minus_one = PcuI512::from_limbs_le([u64::MAX; 8]);
    assert_eq!(
        PcuI512::MIN.pcu_checked_mul(minus_one),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(PcuI512::MIN.pcu_checked_mul(one), Ok(PcuI512::MIN));
    assert_eq!(
        PcuI512::MIN.checked_div_rem(minus_one),
        Err(PcuExecutionFaultKind::SignedDivisionOverflow)
    );
}

#[test]
fn multiplication_retains_high_cross_products_and_detects_lost_high_half() {
    let shift128 = PcuU512::from_limbs_le([0, 0, 1, 0, 0, 0, 0, 0]);
    assert_eq!(
        shift128.pcu_checked_mul(shift128).unwrap().to_limbs_le(),
        [0, 0, 0, 0, 1, 0, 0, 0]
    );
    let top = PcuU512::from_limbs_le([0, 0, 0, 0, 0, 0, 0, 1 << 63]);
    let two = PcuU512::from_limbs_le([2, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(
        top.pcu_checked_mul(two),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    let pattern = PcuU256::from_limbs_le([u64::MAX, u64::MAX, 0, 0]);
    assert_eq!(
        pattern.pcu_checked_mul(pattern).unwrap().to_limbs_le(),
        [1, 0, u64::MAX - 1, u64::MAX]
    );
}

#[test]
fn unsigned_division_handles_high_bit_carry_and_roundtrips_large_products() {
    let divisor = PcuU256::from_limbs_le([7, 0, 0, 1 << 63]);
    let (quotient, remainder) = PcuU256::MAX.checked_div_rem(divisor).unwrap();
    assert_eq!(quotient.to_limbs_le(), [1, 0, 0, 0]);
    assert_eq!(
        quotient
            .pcu_checked_mul(divisor)
            .unwrap()
            .pcu_checked_add(remainder),
        Ok(PcuU256::MAX)
    );
    assert_eq!(
        PcuU256::MAX.checked_div_rem(PcuU256::ZERO),
        Err(PcuExecutionFaultKind::DivideByZero)
    );
    let divisor = PcuU512::from_limbs_le([17, 29, 0, 0, 0, 0, 0, 0]);
    let (q, r) = PcuU512::MAX.checked_div_rem(divisor).unwrap();
    assert_eq!(
        q.pcu_checked_mul(divisor).unwrap().pcu_checked_add(r),
        Ok(PcuU512::MAX)
    );
    assert_eq!(
        compare(&r.to_limbs_le(), &divisor.to_limbs_le()),
        Ordering::Less
    );
}

#[test]
fn deterministic_low_domain_matches_independent_native_u128_and_i128_oracles() {
    let mut seed = 0x31e9_574d_862a_407b_u64;
    for _ in 0..2048 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let left = seed;
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let right = seed | 1;
        let a = PcuU256::from_limbs_le([left, 0, 0, 0]);
        let b = PcuU256::from_limbs_le([right, 0, 0, 0]);
        let product = u128::from(left) * u128::from(right);
        let bytes = product.to_le_bytes();
        let mut expected = [0; 32];
        expected[..16].copy_from_slice(&bytes);
        assert_eq!(a.pcu_checked_mul(b).unwrap().encode_le(), expected);
        let (q, r) = a.checked_div_rem(b).unwrap();
        assert_eq!(q.to_limbs_le(), [left / right, 0, 0, 0]);
        assert_eq!(r.to_limbs_le(), [left % right, 0, 0, 0]);
        let left = i128::from(left) - i128::from(u64::MAX / 2);
        let right = i128::from(right) - i128::from(u64::MAX / 2);
        let signed = |value: i128| {
            let bytes = value.to_le_bytes();
            let mut full = [if value < 0 { 255 } else { 0 }; 32];
            full[..16].copy_from_slice(&bytes);
            PcuI256::decode_le(full)
        };
        assert_eq!(
            signed(left).pcu_checked_mul(signed(right)),
            Ok(signed(left * right))
        );
        if right != 0 {
            assert_eq!(
                signed(left).checked_div_rem(signed(right)),
                Ok((signed(left / right), signed(left % right)))
            );
        }
    }
}

#[test]
fn canonical_encoding_preserves_nan_payloads_and_wide_signed_patterns() {
    let value = PcuI512::from_limbs_le([1, 2, 3, 4, 5, 6, 7, u64::MAX]);
    assert_eq!(PcuI512::decode_le(value.encode_le()), value);
    assert_eq!(value.encode_le()[0], 1);
    assert_eq!(PcuI512::HOST_SIZE, 64);
    assert_eq!(PcuI512::HOST_ALIGNMENT, 8);
    let bits = PcuF256Bits::from_limbs_le([1, 2, 3, u64::MAX]);
    assert_eq!(PcuF256Bits::decode_le(bits.encode_le()), bits);
    assert_eq!(
        u128::MAX.pcu_checked_add(1),
        Err(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        i128::MIN.pcu_checked_sub(1),
        Err(PcuExecutionFaultKind::ArithmeticUnderflow)
    );
}

#[test]
fn explicit_wrapping_and_clamping_preserve_wide_classification() {
    #[rustfmt::skip]
    use crate::{PcuClampedInteger, PcuWrappingInteger};
    let one = PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]);
    assert_eq!(PcuU512::MAX.wrapping_add(one), PcuU512::ZERO);
    assert_eq!(PcuU512::ZERO.wrapping_sub(one), PcuU512::MAX);
    assert_eq!(PcuU512::MAX.wrapping_mul(one), PcuU512::MAX);
    let overflow = PcuU512::MAX.pcu_clamped_add(one).unwrap_err();
    assert_eq!(overflow.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(overflow.clamped_value(), PcuU512::MAX);
    let underflow = PcuU512::ZERO.pcu_clamped_sub(one).unwrap_err();
    assert_eq!(underflow.kind(), PcuExecutionFaultKind::ArithmeticUnderflow);
    assert_eq!(underflow.clamped_value(), PcuU512::ZERO);
    let signed_one = PcuI256::from_limbs_le([1, 0, 0, 0]);
    assert_eq!(PcuI256::MAX.wrapping_add(signed_one), PcuI256::MIN);
    let negative_one = PcuI256::from_limbs_le([u64::MAX; 4]);
    let positive_fault = PcuI256::MIN.pcu_clamped_mul(negative_one).unwrap_err();
    assert_eq!(
        positive_fault.kind(),
        PcuExecutionFaultKind::ArithmeticOverflow
    );
    assert_eq!(positive_fault.clamped_value(), PcuI256::MAX);
    let negative_fault = PcuI256::MIN.pcu_clamped_sub(signed_one).unwrap_err();
    assert_eq!(
        negative_fault.kind(),
        PcuExecutionFaultKind::ArithmeticUnderflow
    );
    assert_eq!(negative_fault.clamped_value(), PcuI256::MIN);
    let native = u128::MAX.pcu_clamped_add(1).unwrap_err();
    assert_eq!(native.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(native.clamped_value(), u128::MAX);
}
