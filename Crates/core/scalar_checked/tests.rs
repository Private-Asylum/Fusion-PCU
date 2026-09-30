use super::*;

macro_rules! unsigned_boundaries {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            let max = <$ty>::MAX;
            assert_eq!(max.pcu_checked_add(0), Ok(max));
            assert_eq!(
                max.pcu_checked_add(1),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(
                (0 as $ty).pcu_checked_sub(1),
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            );
            assert_eq!(max.pcu_checked_sub(max), Ok(0));
            assert_eq!(max.pcu_checked_mul(1), Ok(max));
            assert_eq!(
                max.pcu_checked_mul(2),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(
                max.pcu_checked_mul(max),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(max.pcu_checked_mul(0), Ok(0));
        }
    };
}

macro_rules! signed_boundaries {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            let min = <$ty>::MIN;
            let max = <$ty>::MAX;
            assert_eq!(max.pcu_checked_add(0), Ok(max));
            assert_eq!(min.pcu_checked_sub(0), Ok(min));
            assert_eq!(
                max.pcu_checked_add(1),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(
                min.pcu_checked_add(-1),
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            );
            assert_eq!(
                max.pcu_checked_sub(-1),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(
                min.pcu_checked_sub(1),
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            );
            assert_eq!(
                min.pcu_checked_mul(-1),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(
                min.pcu_checked_mul(2),
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            );
            assert_eq!(
                max.pcu_checked_mul(-2),
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            );
            assert_eq!((min / 2).pcu_checked_mul(2), Ok(min));
            assert_eq!(
                min.pcu_checked_mul(min),
                Err(PcuExecutionFaultKind::ArithmeticOverflow)
            );
            assert_eq!(
                min.pcu_checked_mul(max),
                Err(PcuExecutionFaultKind::ArithmeticUnderflow)
            );
            assert_eq!(min.pcu_checked_mul(0), Ok(0));
            assert_eq!(min.pcu_checked_mul(1), Ok(min));
            assert_eq!(max.pcu_checked_mul(-1), Ok(-max));
        }
    };
}

unsigned_boundaries!(u8_boundaries, u8);
unsigned_boundaries!(u16_boundaries, u16);
unsigned_boundaries!(u32_boundaries, u32);
unsigned_boundaries!(u64_boundaries, u64);
signed_boundaries!(i8_boundaries, i8);
signed_boundaries!(i16_boundaries, i16);
signed_boundaries!(i32_boundaries, i32);
signed_boundaries!(i64_boundaries, i64);

fn mathematical_result<T: TryFrom<i128>>(
    exact: i128,
    minimum: i128,
) -> Result<T, PcuExecutionFaultKind> {
    T::try_from(exact).map_err(|_| {
        if exact < minimum {
            PcuExecutionFaultKind::ArithmeticUnderflow
        } else {
            PcuExecutionFaultKind::ArithmeticOverflow
        }
    })
}

#[test]
fn exhaustive_i8_matches_widened_mathematical_oracle() {
    for left in i8::MIN..=i8::MAX {
        for right in i8::MIN..=i8::MAX {
            let a = i128::from(left);
            let b = i128::from(right);
            let minimum = i128::from(i8::MIN);
            assert_eq!(
                left.pcu_checked_add(right),
                mathematical_result(a + b, minimum)
            );
            assert_eq!(
                left.pcu_checked_sub(right),
                mathematical_result(a - b, minimum)
            );
            assert_eq!(
                left.pcu_checked_mul(right),
                mathematical_result(a * b, minimum)
            );
        }
    }
}

#[test]
fn exhaustive_u8_matches_widened_mathematical_oracle() {
    for left in u8::MIN..=u8::MAX {
        for right in u8::MIN..=u8::MAX {
            let a = i128::from(left);
            let b = i128::from(right);
            assert_eq!(left.pcu_checked_add(right), mathematical_result(a + b, 0));
            assert_eq!(left.pcu_checked_sub(right), mathematical_result(a - b, 0));
            assert_eq!(left.pcu_checked_mul(right), mathematical_result(a * b, 0));
        }
    }
}
