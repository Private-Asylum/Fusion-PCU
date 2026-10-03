//! Independent changed raw encoding banks including every high limb.
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuU256,PcuI256,PcuU512,PcuI512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits};
pub trait Sample: PcuScalar {
    fn pattern(seed: usize) -> Self;
}
macro_rules! samples{($($ty:ty),+)=>{$(impl Sample for $ty{fn pattern(seed:usize)->Self{Self::decode_le(std::array::from_fn(|byte|{if byte<2{u8::try_from((seed>>(byte*8))&255).unwrap()}else{u8::try_from(seed&255).unwrap().wrapping_add(u8::try_from(byte).unwrap().wrapping_mul(37))}}))}})+};}
samples!(
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    u128,
    i128,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    f32,
    f64,
    PcuF128Bits,
    PcuF256Bits
);
