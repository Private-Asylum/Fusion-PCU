//! Raw carrier sample oracle; no floating conversion or arithmetic participates.
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,
    PcuF128Bits,PcuF256Bits,PcuU256,PcuI256,PcuU512,PcuI512};
pub trait Sample: PcuScalar {
    fn sample(seed: u8) -> Self;
}
macro_rules! samples {
    ($($ty:ty => $width:literal;)+) => {$(
        impl Sample for $ty {
            fn sample(seed: u8) -> Self {
                let mut bytes = [0_u8; $width];
                for (index, byte) in bytes.iter_mut().enumerate() {
                    *byte = match seed {
                        0 => u8::MAX,
                        1 => 0,
                        2 => if index + 1 == $width { 0x80 } else { 0 },
                        _ => seed.wrapping_mul(37).wrapping_add(u8::try_from(index).unwrap().wrapping_mul(19)),
                    };
                }
                Self::decode_le(bytes)
            }
        }
    )+};
}
samples! {
    u8 => 1; i8 => 1; u16 => 2; i16 => 2; u32 => 4; i32 => 4; u64 => 8; i64 => 8;
    u128 => 16; i128 => 16; PcuU256 => 32; PcuI256 => 32; PcuU512 => 64; PcuI512 => 64;
    PcuF16Bits => 2; PcuBf16Bits => 2; PcuF8E4M3FnBits => 1; PcuF8E5M2Bits => 1;
    f32 => 4; f64 => 8; PcuF128Bits => 16; PcuF256Bits => 32;
}
pub fn compare<T: Sample>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}
