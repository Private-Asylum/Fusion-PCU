//! Independent base-256 division and frozen `BigInt` vectors for every checked width.
#[path = "division/division.rs"]
mod division;
pub use division::evaluate;
type Golden<T> = (T, T, Result<(T, T), PcuExecutionFaultKind>);
pub fn goldens<T: Integer>() -> Vec<Golden<T>> {
    division::goldens()
}
pub fn domain<T: Integer>(lhs: T, rhs: T) -> Result<(), PcuExecutionFaultKind> {
    division::domain(lhs, rhs)
}
use division::Wide;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFaultKind,
    PcuScalar,
    PcuCheckedIntegerDivision,
};
pub trait Integer:
    PcuScalar + PcuCheckedIntegerDivision + Wide + Copy + PartialEq + std::fmt::Debug
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
                { let mut bytes = [0; size_of::<Self>()]; let n = bytes.len().min(8); bytes[..n].copy_from_slice(&bits.to_le_bytes()[..n]); Self::from_le_bytes(bytes) }
            }
            fn checked(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                evaluate(self, rhs)
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
        let mut raw = [0; 64];
        for chunk in raw.as_chunks_mut::<8>().0 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            chunk.copy_from_slice(&state.to_le_bytes());
        }
        let a = match index % 7 {
            0 => T::MIN,
            1 => T::MAX,
            _ => T::from_bytes(raw),
        };
        raw.reverse();
        let candidate = match index % 5 {
            0 => T::MAX,
            1 => a,
            2 => T::ONE,
            3 => T::from_bytes(raw),
            _ => T::from_bits(state.rotate_left(23)),
        };
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

macro_rules! wide {
    ($ty:ty, $minimum:expr, $one:expr, $sentinel:expr, $negative:expr) => {
        impl Integer for $ty {
            const LABEL: &'static str = stringify!($ty);
            const MIN: Self = $minimum;
            const MAX: Self = Self::MAX;
            const ZERO: Self = Self::ZERO;
            const ONE: Self = $one;
            const SENTINEL: Self = $sentinel;
            const NEGATIVE_ONE: Option<Self> = $negative;
            fn from_bits(bits: u64) -> Self {
                let mut bytes = [0; 64];
                bytes[..8].copy_from_slice(&bits.to_le_bytes());
                Self::from_bytes(bytes)
            }
            fn checked(self, rhs: Self) -> Result<(Self, Self), PcuExecutionFaultKind> {
                evaluate(self, rhs)
            }
        }
    };
}
integers!(i128: Some(-1), u128: None);
wide!(
    fusion_pcu::PcuI256,
    fusion_pcu::PcuI256::MIN,
    fusion_pcu::PcuI256::from_limbs_le([1, 0, 0, 0]),
    fusion_pcu::PcuI256::from_limbs_le([99, 0, 0, 0]),
    Some(fusion_pcu::PcuI256::from_limbs_le([u64::MAX; 4]))
);
wide!(
    fusion_pcu::PcuU256,
    fusion_pcu::PcuU256::ZERO,
    fusion_pcu::PcuU256::from_limbs_le([1, 0, 0, 0]),
    fusion_pcu::PcuU256::from_limbs_le([99, 0, 0, 0]),
    None
);
wide!(
    fusion_pcu::PcuI512,
    fusion_pcu::PcuI512::MIN,
    fusion_pcu::PcuI512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
    fusion_pcu::PcuI512::from_limbs_le([99, 0, 0, 0, 0, 0, 0, 0]),
    Some(fusion_pcu::PcuI512::from_limbs_le([u64::MAX; 8]))
);
wide!(
    fusion_pcu::PcuU512,
    fusion_pcu::PcuU512::ZERO,
    fusion_pcu::PcuU512::from_limbs_le([1, 0, 0, 0, 0, 0, 0, 0]),
    fusion_pcu::PcuU512::from_limbs_le([99, 0, 0, 0, 0, 0, 0, 0]),
    None
);
