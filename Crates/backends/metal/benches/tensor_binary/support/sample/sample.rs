//! Literal small exact integer/dyadic encodings, independent of checked scalar arithmetic.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuU256,
    PcuI256,
    PcuU512,
    PcuI512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
pub trait Sample: PcuScalar {
    fn literal(value: u8) -> Self;
}
macro_rules! integers{($($ty:ty=>$width:literal;)+)=>{$(
    impl Sample for $ty{fn literal(value:u8)->Self{let mut bytes=[0;$width];bytes[0]=value;Self::decode_le(bytes)}}
)+};}
integers! {u8=>1;i8=>1;u16=>2;i16=>2;u32=>4;i32=>4;u64=>8;i64=>8;u128=>16;i128=>16;PcuU256=>32;PcuI256=>32;PcuU512=>64;PcuI512=>64;}
macro_rules! floats{($($ty:ty,$codes:expr;)+)=>{$(
    impl Sample for $ty{fn literal(value:u8)->Self{let codes=$codes;Self::from_bits(codes[usize::from(value)])}}
)+};}
floats! {PcuF16Bits,[0,0x3c00,0x4000,0x4200,0x4400,0x4500,0x4600];PcuBf16Bits,[0,0x3f80,0x4000,0x4040,0x4080,0x40a0,0x40c0];PcuF8E4M3FnBits,[0,0x38,0x40,0x44,0x48,0x4a,0x4c];PcuF8E5M2Bits,[0,0x3c,0x40,0x42,0x44,0x45,0x46];f32,[0,0x3f80_0000,0x4000_0000,0x4040_0000,0x4080_0000,0x40a0_0000,0x40c0_0000];f64,[0,0x3ff0_0000_0000_0000,0x4000_0000_0000_0000,0x4008_0000_0000_0000,0x4010_0000_0000_0000,0x4014_0000_0000_0000,0x4018_0000_0000_0000];}
