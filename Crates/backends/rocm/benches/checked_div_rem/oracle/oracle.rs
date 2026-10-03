//! Native Rust checked division is an independent oracle for the GPU lowering.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFaultKind,
    PcuScalar,
    PcuCheckedIntegerDivision,
};
pub trait Integer:
    PcuScalar + PcuCheckedIntegerDivision + Default + Copy + PartialEq + std::fmt::Debug
{
    const LABEL: &'static str;
    const MIN: Self;
    const MAX: Self;
    const ZERO: Self;
    const ONE: Self;
    const SENTINEL: Self;
    const NEGATIVE_ONE: Option<Self>;
    fn from_bits(bits: u64) -> Self;
    fn checked(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind>;
}
macro_rules! integers {
    ($($ty:ty: $minus:expr),* $(,)?) => {$(
        impl Integer for $ty {
            const LABEL: &'static str = stringify!($ty);
            const MIN: Self = Self::MIN;
            const MAX: Self = Self::MAX;
            const ZERO: Self = 0;
            const ONE: Self = 1;
            const SENTINEL: Self = 99;
            const NEGATIVE_ONE: Option<Self> = $minus;
            fn from_bits(bits: u64) -> Self {
                Self::from_le_bytes(bits.to_le_bytes()[..size_of::<Self>()].try_into().unwrap())
            }
            fn checked(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                match (self.checked_div(rhs), self.checked_rem(rhs)) {
                    (Some(q), Some(r)) => Ok((q, r)),
                    _ => Err(if rhs == 0 { PcuExecutionFaultKind::DivideByZero }
                        else { PcuExecutionFaultKind::SignedDivisionOverflow }),
                }
            }
        }
    )*};
}
integers!(i8: Some(-1), u8: None, i16: Some(-1), u16: None,
    i32: Some(-1), u32: None, i64: Some(-1), u64: None);
pub fn inputs<T: Integer>(count: usize, phase: u64) -> (Vec<T>, Vec<T>) {
    let mut state = 0xc13a_9d5e_7710_ab37_u64.wrapping_add(phase);
    let mut lhs = Vec::with_capacity(count);
    let mut rhs = Vec::with_capacity(count);
    for index in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let a = match index % 7 {
            0 => T::MIN,
            1 => T::MAX,
            _ => T::from_bits(state),
        };
        let candidate = T::from_bits(state.rotate_left(23));
        let b = if a.checked(candidate).is_ok() {
            candidate
        } else {
            T::ONE
        };
        lhs.push(a);
        rhs.push(b);
    }
    (lhs, rhs)
}
pub fn verify<T: Integer>(lhs: &[T], rhs: &[T], q: &[T], r: &[T]) {
    for (index, (&a, &b)) in lhs.iter().zip(rhs).enumerate() {
        let expected = a.checked(b).unwrap();
        assert_eq!(a.pcu_checked_div(b), Ok(expected.0));
        assert_eq!(a.pcu_checked_rem(b), Ok(expected.1));
        assert_eq!(
            (q[index], r[index]),
            expected,
            "{} element={index}",
            T::LABEL
        );
    }
    assert!(q[lhs.len()..].iter().all(|&value| value == T::SENTINEL));
    assert!(r[lhs.len()..].iter().all(|&value| value == T::SENTINEL));
}
