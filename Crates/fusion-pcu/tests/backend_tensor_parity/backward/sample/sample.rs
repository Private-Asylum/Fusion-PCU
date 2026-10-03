//! Representation constants form an independent bit oracle, not reference arithmetic.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
};

pub trait Sample: PcuCheckedFloat {
    const TINY: Self;
    const INVALID: Self;
    const NEGATIVE_ZERO: Self;
    const MAX: Self;
    fn finite(value: f32) -> Self;
}

macro_rules! samples {
    ($($ty:ty, $max:expr, $invalid:expr, $negative_zero:expr;)+) => {$(
        impl Sample for $ty {
            const TINY: Self = Self::from_bits(1);
            const INVALID: Self = Self::from_bits($invalid);
            const NEGATIVE_ZERO: Self = Self::from_bits($negative_zero);
            const MAX: Self = Self::from_bits($max);
            fn finite(value: f32) -> Self {
                Self::pcu_checked_from_f32(value).unwrap()
            }
        }
    )+};
}
samples! {
    PcuF16Bits, 0x7bff, 0x7c00, 0x8000;
    PcuBf16Bits, 0x7f7f, 0x7f80, 0x8000;
    PcuF8E4M3FnBits, 0x7e, 0x7f, 0x80;
    PcuF8E5M2Bits, 0x7b, 0x7c, 0x80;
}
impl Sample for f32 {
    const TINY: Self = Self::from_bits(1);
    const INVALID: Self = Self::INFINITY;
    const NEGATIVE_ZERO: Self = -0.0;
    const MAX: Self = Self::MAX;
    fn finite(value: f32) -> Self {
        value
    }
}
impl Sample for f64 {
    const TINY: Self = Self::from_bits(1);
    const INVALID: Self = Self::INFINITY;
    const NEGATIVE_ZERO: Self = -0.0;
    const MAX: Self = Self::MAX;
    fn finite(value: f32) -> Self {
        Self::from(value)
    }
}

pub fn low_formats(policy: PcuFloatUnderflowPolicy) {
    super::format::<PcuF16Bits>(policy, true);
    super::format::<PcuBf16Bits>(policy, true);
    super::format::<PcuF8E4M3FnBits>(policy, true);
    super::format::<PcuF8E5M2Bits>(policy, true);
}
