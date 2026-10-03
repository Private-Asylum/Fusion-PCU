//! Arbitrary carrier bits: complete one/two-byte encodings and every wider bit position.
#[rustfmt::skip]
use pcu_facade::{
    PcuScalar,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
};
pub trait Sample: PcuScalar {
    fn pattern(seed: usize) -> Self;
}
macro_rules! samples {($($ty:ty),+$(,)?)=>{$(impl Sample for $ty {fn pattern(seed:usize)->Self {Self::decode_le(std::array::from_fn(|index| {if Self::HOST_SIZE<=2 {seed.to_le_bytes()[index]}else if seed==0 {0}else if seed==1 {255}else if seed<Self::HOST_SIZE*8+2 {let bit=seed-2;if index==bit/8 {1 << (bit%8)}else{0}}else{let mut state=u64::try_from(seed).unwrap()^u64::try_from(index).unwrap().wrapping_mul(0x9e37_79b9_7f4a_7c15);state^=state<<13;state^=state>>7;state^=state<<17;state.to_le_bytes()[0]}}))}})+};}
samples!(
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
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    f32,
    f64,
    PcuF128Bits,
    PcuF256Bits
);
pub fn same<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
