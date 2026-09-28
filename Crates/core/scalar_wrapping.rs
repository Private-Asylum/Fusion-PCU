//! Sealed integer scalars for the bounded generic wrapping arithmetic profile.

use crate::PcuScalar;

mod sealed {
    pub trait Sealed {}
    impl Sealed for u8 {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for i8 {}
    impl Sealed for i16 {}
    impl Sealed for i32 {}
    impl Sealed for i64 {}
}

/// A `PcuScalar` whose integer wrapping add, subtract, and multiply are part of its generic PCU
/// contract. Implementations are sealed to the eight built-in integer scalar types.
pub trait PcuWrappingInteger: PcuScalar + sealed::Sealed + Sized {
    /// Adds two values modulo the scalar's bit width.
    #[must_use]
    fn wrapping_add(self, rhs: Self) -> Self;
    /// Subtracts two values modulo the scalar's bit width.
    #[must_use]
    fn wrapping_sub(self, rhs: Self) -> Self;
    /// Multiplies two values modulo the scalar's bit width.
    #[must_use]
    fn wrapping_mul(self, rhs: Self) -> Self;
}

macro_rules! impl_wrapping_integer {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl PcuWrappingInteger for $ty {
                fn wrapping_add(self, rhs: Self) -> Self { <$ty>::wrapping_add(self, rhs) }
                fn wrapping_sub(self, rhs: Self) -> Self { <$ty>::wrapping_sub(self, rhs) }
                fn wrapping_mul(self, rhs: Self) -> Self { <$ty>::wrapping_mul(self, rhs) }
            }
        )+
    };
}

impl_wrapping_integer!(u8, u16, u32, u64, i8, i16, i32, i64);
