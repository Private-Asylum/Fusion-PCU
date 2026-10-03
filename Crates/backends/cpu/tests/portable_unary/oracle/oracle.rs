//! Reuses the independent sign/magnitude oracle with explicit low-format encoding constants.
#[path = "../../native_unary/oracle/oracle.rs"]
mod bits;
pub use bits::{bits, native, native_broadcast, Native};
#[rustfmt::skip]
use pcu_facade::{
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
macro_rules! carrier {
    ($ty:ty, $raw:ty, $sign:expr, $max:expr, $fraction:expr) => {
        impl Native for $ty {
            const SIGN: u64 = $sign;
            const MAX: u64 = $max;
            const FRACTION: u32 = $fraction;
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn from_bits(bits: u64) -> Self {
                Self::from_bits(<$raw>::try_from(bits).unwrap())
            }
        }
    };
}
carrier!(PcuF16Bits, u16, 0x8000, 0x7bff, 10);
carrier!(PcuBf16Bits, u16, 0x8000, 0x7f7f, 7);
carrier!(PcuF8E4M3FnBits, u8, 0x80, 0x7e, 3);
carrier!(PcuF8E5M2Bits, u8, 0x80, 0x7b, 2);
