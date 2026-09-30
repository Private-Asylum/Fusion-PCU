#[rustfmt::skip]
use super::{
    PcuClampedError,
    PcuClampedFault,
    PcuClampedInteger,
};
use crate::PcuExecutionFaultKind;
use std::string::String;

fn assert_fault<T: core::fmt::Debug + PartialEq + Copy>(
    result: Result<T, PcuClampedFault<T>>,
    expected_kind: PcuExecutionFaultKind,
    expected_value: T,
) {
    match result {
        Ok(_) => panic!("expected a range fault"),
        Err(fault) => {
            assert_eq!(fault.kind(), expected_kind);
            assert_eq!(fault.clamped_value(), expected_value);
        }
    }
}

#[test]
fn all_integer_widths_report_faults_and_return_the_saturated_endpoint() {
    macro_rules! unsigned_boundaries {
        ($ty:ty) => {{
            assert_fault(
                <$ty>::MAX.pcu_clamped_add(1),
                PcuExecutionFaultKind::ArithmeticOverflow,
                <$ty>::MAX,
            );
            assert_fault(
                <$ty>::MIN.pcu_clamped_sub(1),
                PcuExecutionFaultKind::ArithmeticUnderflow,
                <$ty>::MIN,
            );
            assert_fault(
                <$ty>::MAX.pcu_clamped_mul(2),
                PcuExecutionFaultKind::ArithmeticOverflow,
                <$ty>::MAX,
            );
            assert_eq!(<$ty>::MAX.pcu_clamped_add(0), Ok(<$ty>::MAX));
            assert_eq!(<$ty>::MIN.pcu_clamped_sub(0), Ok(<$ty>::MIN));
            assert_eq!(<$ty>::MAX.pcu_clamped_mul(1), Ok(<$ty>::MAX));
        }};
    }
    macro_rules! signed_boundaries {
        ($ty:ty) => {{
            assert_fault(
                <$ty>::MAX.pcu_clamped_add(1),
                PcuExecutionFaultKind::ArithmeticOverflow,
                <$ty>::MAX,
            );
            assert_fault(
                <$ty>::MIN.pcu_clamped_add(-1),
                PcuExecutionFaultKind::ArithmeticUnderflow,
                <$ty>::MIN,
            );
            assert_fault(
                <$ty>::MIN.pcu_clamped_sub(1),
                PcuExecutionFaultKind::ArithmeticUnderflow,
                <$ty>::MIN,
            );
            assert_fault(
                <$ty>::MAX.pcu_clamped_sub(-1),
                PcuExecutionFaultKind::ArithmeticOverflow,
                <$ty>::MAX,
            );
            assert_fault(
                <$ty>::MIN.pcu_clamped_mul(2),
                PcuExecutionFaultKind::ArithmeticUnderflow,
                <$ty>::MIN,
            );
            assert_fault(
                <$ty>::MIN.pcu_clamped_mul(-1),
                PcuExecutionFaultKind::ArithmeticOverflow,
                <$ty>::MAX,
            );
            assert_eq!(<$ty>::MAX.pcu_clamped_add(0), Ok(<$ty>::MAX));
            assert_eq!(<$ty>::MIN.pcu_clamped_sub(0), Ok(<$ty>::MIN));
            assert_eq!(<$ty>::MIN.pcu_clamped_mul(1), Ok(<$ty>::MIN));
        }};
    }

    unsigned_boundaries!(u8);
    unsigned_boundaries!(u16);
    unsigned_boundaries!(u32);
    unsigned_boundaries!(u64);
    signed_boundaries!(i8);
    signed_boundaries!(i16);
    signed_boundaries!(i32);
    signed_boundaries!(i64);
}

#[test]
fn u8_clamped_operations_match_exhaustive_widened_integer_oracle() {
    for lhs in u8::MIN..=u8::MAX {
        for rhs in u8::MIN..=u8::MAX {
            let left = u16::from(lhs);
            let right = u16::from(rhs);
            let add = left + right;
            if let Ok(expected) = u8::try_from(add) {
                assert_eq!(lhs.pcu_clamped_add(rhs), Ok(expected));
            } else {
                assert_fault(
                    lhs.pcu_clamped_add(rhs),
                    PcuExecutionFaultKind::ArithmeticOverflow,
                    u8::MAX,
                );
            }

            if lhs >= rhs {
                assert_eq!(lhs.pcu_clamped_sub(rhs), Ok(lhs - rhs));
            } else {
                assert_fault(
                    lhs.pcu_clamped_sub(rhs),
                    PcuExecutionFaultKind::ArithmeticUnderflow,
                    u8::MIN,
                );
            }

            let mul = left * right;
            if let Ok(expected) = u8::try_from(mul) {
                assert_eq!(lhs.pcu_clamped_mul(rhs), Ok(expected));
            } else {
                assert_fault(
                    lhs.pcu_clamped_mul(rhs),
                    PcuExecutionFaultKind::ArithmeticOverflow,
                    u8::MAX,
                );
            }
        }
    }
}

#[test]
fn i8_clamped_operations_match_exhaustive_widened_integer_oracle() {
    for lhs in i8::MIN..=i8::MAX {
        for rhs in i8::MIN..=i8::MAX {
            let left = i16::from(lhs);
            let right = i16::from(rhs);
            assert_i8_result(lhs.pcu_clamped_add(rhs), left + right);
            assert_i8_result(lhs.pcu_clamped_sub(rhs), left - right);
            assert_i8_result(lhs.pcu_clamped_mul(rhs), left * right);
        }
    }
}

fn assert_i8_result(result: Result<i8, PcuClampedFault<i8>>, exact: i16) {
    if exact < i16::from(i8::MIN) {
        assert_fault(result, PcuExecutionFaultKind::ArithmeticUnderflow, i8::MIN);
    } else if exact > i16::from(i8::MAX) {
        assert_fault(result, PcuExecutionFaultKind::ArithmeticOverflow, i8::MAX);
    } else {
        assert_eq!(result, Ok(i8::try_from(exact).expect("in range")));
    }
}

#[test]
fn clamped_error_exposes_kind_without_copying_or_cloning_payload() {
    let fatal: PcuClampedError<String> =
        PcuClampedError::Fatal(PcuExecutionFaultKind::InvalidFloatingOperand);
    assert_eq!(fatal.kind(), PcuExecutionFaultKind::InvalidFloatingOperand);

    let range = PcuClampedError::Range(PcuClampedFault::new(
        PcuExecutionFaultKind::ArithmeticOverflow,
        String::from("recovered endpoint"),
    ));
    assert_eq!(range.kind(), PcuExecutionFaultKind::ArithmeticOverflow);
    match range {
        PcuClampedError::Range(fault) => {
            assert_eq!(fault.clamped_value(), "recovered endpoint");
        }
        PcuClampedError::Fatal(_) => panic!("expected range recovery"),
    }
}

#[test]
fn recovery_transfers_non_clone_ownership_and_drops_exactly_once() {
    struct Payload<'a>(&'a core::cell::Cell<usize>);
    impl Drop for Payload<'_> {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let drops = core::cell::Cell::new(0);
    let fault = PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Payload(&drops));
    let recovered = fault.clamped_value();
    assert_eq!(drops.get(), 0);
    drop(recovered);
    assert_eq!(drops.get(), 1);
    let discarded =
        PcuClampedFault::new(PcuExecutionFaultKind::ArithmeticOverflow, Payload(&drops));
    drop(discarded);
    assert_eq!(drops.get(), 2);
}
