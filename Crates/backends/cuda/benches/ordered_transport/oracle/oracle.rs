//! Independent raw bytes, including arbitrary float payloads and every high limb.
use pcu_facade::PcuScalar;
pub trait Format: PcuScalar {
    const LABEL: &'static str;
    fn pattern(seed: u8) -> Self;
}
macro_rules! formats {($($ty:ty),+$(,)?)=>{$(
 impl Format for $ty {
 const LABEL: &'static str = stringify!($ty);
 fn pattern(seed:u8)->Self {Self::decode_le(std::array::from_fn(|index|seed.wrapping_add(u8::try_from(index).unwrap().wrapping_mul(37))))}
 }
)+};}
formats!(
    i8,
    u8,
    i16,
    u16,
    i32,
    u32,
    i64,
    u64,
    i128,
    u128,
    pcu_facade::PcuI256,
    pcu_facade::PcuU256,
    pcu_facade::PcuI512,
    pcu_facade::PcuU512,
    pcu_facade::PcuF16Bits,
    pcu_facade::PcuBf16Bits,
    pcu_facade::PcuF8E4M3FnBits,
    pcu_facade::PcuF8E5M2Bits,
    f32,
    f64,
    pcu_facade::PcuF128Bits,
    pcu_facade::PcuF256Bits
);
pub fn verify<T: Format>(input: &[T], actual: &[T]) {
    for (expected, got) in input.iter().zip(actual) {
        assert_eq!(
            expected.encode_le().as_ref(),
            got.encode_le().as_ref(),
            "{}",
            T::LABEL
        );
    }
    for got in &actual[input.len()..] {
        assert_eq!(
            got.encode_le().as_ref(),
            T::pattern(251).encode_le().as_ref()
        );
    }
}
