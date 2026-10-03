use super::*;

#[test]
fn exhaustive_native_byte_division_matches_widened_signed_oracle() {
    for lhs in i8::MIN..=i8::MAX {
        for rhs in i8::MIN..=i8::MAX {
            let expected = if rhs == 0 {
                Err(PcuExecutionFaultKind::DivideByZero)
            } else if lhs == i8::MIN && rhs == -1 {
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            } else {
                Ok((
                    i16::from(lhs) / i16::from(rhs),
                    i16::from(lhs) % i16::from(rhs),
                ))
            };
            assert_eq!(
                lhs.pcu_checked_div(rhs).map(i16::from),
                expected.map(|(q, _)| q)
            );
            assert_eq!(
                lhs.pcu_checked_rem(rhs).map(i16::from),
                expected.map(|(_, r)| r)
            );
            assert_eq!(
                lhs.pcu_checked_div_rem(rhs)
                    .map(|(q, r)| (i16::from(q), i16::from(r))),
                expected
            );
            assert_eq!(lhs.pcu_div_rem_domain(rhs), expected.map(|_| ()));
        }
    }
}

#[test]
fn exhaustive_unsigned_byte_domain_matches_widened_joint_arithmetic() {
    for lhs in u8::MIN..=u8::MAX {
        for rhs in u8::MIN..=u8::MAX {
            let expected = if rhs == 0 {
                Err(PcuExecutionFaultKind::DivideByZero)
            } else {
                Ok((
                    u16::from(lhs) / u16::from(rhs),
                    u16::from(lhs) % u16::from(rhs),
                ))
            };
            assert_eq!(lhs.pcu_div_rem_domain(rhs), expected.map(|_| ()));
            assert_eq!(
                lhs.pcu_checked_div_rem(rhs)
                    .map(|(q, r)| (u16::from(q), u16::from(r))),
                expected
            );
        }
    }
}

#[test]
#[allow(clippy::cognitive_complexity)] // Expand the same independent boundary oracle across every signed/unsigned width.
fn every_width_classifies_zero_and_signed_overflow_and_preserves_remainder_sign() {
    macro_rules! unsigned {
        ($ty:ty) => {
            assert_eq!(
                <$ty>::MAX.pcu_checked_div(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            assert_eq!(
                <$ty>::MAX.pcu_checked_rem(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            let seven: $ty = 7;
            assert_eq!(seven.pcu_div_rem_domain(3), Ok(()));
            assert_eq!(
                <$ty>::MAX.pcu_div_rem_domain(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            assert_eq!(seven.pcu_checked_div(3), Ok(2));
            assert_eq!(seven.pcu_checked_div_rem(3), Ok((2, 1)));
            assert_eq!(
                <$ty>::MAX.pcu_checked_div_rem(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
        };
    }
    unsigned!(u8);
    unsigned!(u16);
    unsigned!(u32);
    unsigned!(u64);
    unsigned!(u128);
    assert_eq!(u128::MAX.pcu_checked_div(1), Ok(u128::MAX));
    for (lhs, rhs, q, r) in [(-7_i128, 3, -2, -1), (7, -3, -2, 1), (-7, -3, 2, -1)] {
        assert_eq!(lhs.pcu_checked_div(rhs), Ok(q));
        assert_eq!(lhs.pcu_checked_rem(rhs), Ok(r));
    }
    macro_rules! signed {
        ($ty:ty) => {
            assert_eq!(
                <$ty>::MIN.pcu_div_rem_domain(-1),
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!(
                <$ty>::MIN.pcu_div_rem_domain(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            assert_eq!(<$ty>::MIN.pcu_div_rem_domain(1), Ok(()));
            assert_eq!(
                <$ty>::MIN.pcu_checked_div(-1),
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!(
                <$ty>::MIN.pcu_checked_rem(-1),
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!(
                <$ty>::MIN.pcu_checked_div(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            assert_eq!(
                <$ty>::MIN.pcu_checked_div_rem(-1),
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!(
                <$ty>::MIN.pcu_checked_div_rem(0),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
        };
    }
    signed!(i8);
    signed!(i16);
    signed!(i32);
    signed!(i64);
    signed!(i128);
    let zero = PcuU512::ZERO;
    assert_eq!(
        PcuU512::MAX.pcu_checked_rem(zero),
        Err(PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(
        PcuU256::MAX.pcu_checked_div(PcuU256::ZERO),
        Err(PcuExecutionFaultKind::DivideByZero)
    );
    assert_eq!(
        PcuI256::MIN.pcu_checked_div(PcuI256::from_limbs_le([u64::MAX; 4])),
        Err(PcuExecutionFaultKind::SignedDivisionOverflow)
    );
    assert_eq!(
        PcuI512::MIN.pcu_checked_rem(PcuI512::from_limbs_le([u64::MAX; 8])),
        Err(PcuExecutionFaultKind::SignedDivisionOverflow)
    );
}

#[test]
fn fused_wide_pair_retains_high_limbs_and_exact_endpoint_faults() {
    macro_rules! unsigned {
        ($ty:ty, $limbs:expr) => {
            let divisor = <$ty>::from_limbs_le($limbs);
            assert_eq!(<$ty>::MAX.pcu_div_rem_domain(divisor), Ok(()));
            assert_eq!(
                <$ty>::MAX.pcu_div_rem_domain(<$ty>::ZERO),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            let (q, r) = <$ty>::MAX.pcu_checked_div_rem(divisor).unwrap();
            assert_eq!(
                q.pcu_checked_mul(divisor).unwrap().pcu_checked_add(r),
                Ok(<$ty>::MAX)
            );
            assert_eq!(
                <$ty>::MAX.pcu_checked_div_rem(<$ty>::ZERO),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
        };
    }
    unsigned!(PcuU256, [7, 0, 0, 1 << 63]);
    unsigned!(PcuU512, [7, 0, 0, 0, 0, 0, 0, 1 << 63]);
    macro_rules! signed {
        ($ty:ty, $limbs:expr) => {
            let minus_one = <$ty>::from_limbs_le($limbs);
            assert_eq!(
                <$ty>::MIN.pcu_div_rem_domain(minus_one),
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!(
                <$ty>::MIN.pcu_div_rem_domain(<$ty>::ZERO),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            assert_eq!(minus_one.pcu_div_rem_domain(minus_one), Ok(()));
            assert_eq!(
                <$ty>::MIN.pcu_checked_div_rem(minus_one),
                Err(PcuExecutionFaultKind::SignedDivisionOverflow)
            );
            assert_eq!(
                <$ty>::MIN.pcu_checked_div_rem(<$ty>::ZERO),
                Err(PcuExecutionFaultKind::DivideByZero)
            );
            assert_eq!(
                minus_one.pcu_checked_div_rem(minus_one),
                Ok((minus_one.pcu_checked_mul(minus_one).unwrap(), <$ty>::ZERO))
            );
        };
    }
    signed!(PcuI256, [u64::MAX; 4]);
    signed!(PcuI512, [u64::MAX; 8]);
}
