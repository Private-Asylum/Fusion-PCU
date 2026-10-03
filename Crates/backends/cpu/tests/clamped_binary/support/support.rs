//! Manual exact representation constants shared by proof and independent native controls.
#[rustfmt::skip]
use pcu_facade::{
    PcuCheckedFloat,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
pub trait Bits: PcuCheckedFloat + core::fmt::Debug + PartialEq {
    const ONE: u64;
    const SIGN: u64;
    const MAX: u64;
    const MIN_NORMAL: u64;
    fn from_bits(bits: u64) -> Self;
    fn bits(self) -> u64;
}
macro_rules! bits {
    ($ty:ty, $raw:ty, $one:expr, $sign:expr, $max:expr, $min:expr) => {
        impl Bits for $ty {
            const ONE: u64 = $one;
            const SIGN: u64 = $sign;
            const MAX: u64 = $max;
            const MIN_NORMAL: u64 = $min;
            fn from_bits(bits: u64) -> Self {
                Self::from_bits(<$raw>::try_from(bits).unwrap())
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
        }
    };
}
bits!(f32, u32, 0x3f80_0000, 0x8000_0000, 0x7f7f_ffff, 0x80_0000);
bits!(
    f64,
    u64,
    0x3ff0_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x7fef_ffff_ffff_ffff,
    0x10_0000_0000_0000
);
bits!(PcuF16Bits, u16, 0x3c00, 0x8000, 0x7bff, 0x400);
bits!(PcuBf16Bits, u16, 0x3f80, 0x8000, 0x7f7f, 0x80);
bits!(PcuF8E4M3FnBits, u8, 0x38, 0x80, 0x7e, 8);
bits!(PcuF8E5M2Bits, u8, 0x3c, 0x80, 0x7b, 4);
