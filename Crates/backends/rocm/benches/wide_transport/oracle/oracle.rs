//! Independent byte identity oracle, including every high limb and arbitrary float payloads.
use fusion_pcu::PcuScalar;
pub trait Format: PcuScalar {
    const LABEL: &'static str;
    fn pattern(seed: u8) -> Self;
    fn sentinel() -> Self {
        Self::pattern(251)
    }
}
macro_rules! formats {($($ty:ty=>$label:literal),+$(,)?)=>{$(
 impl Format for $ty {
 const LABEL:&'static str=$label;
 fn pattern(seed:u8)->Self {Self::decode_le(std::array::from_fn(|index|seed.wrapping_add(u8::try_from(index).unwrap().wrapping_mul(37))))}
 }
)+};}
formats!(i128=>"i128",u128=>"u128",fusion_pcu::PcuI256=>"i256",fusion_pcu::PcuU256=>"u256",fusion_pcu::PcuI512=>"i512",fusion_pcu::PcuU512=>"u512",fusion_pcu::PcuF128Bits=>"f128",fusion_pcu::PcuF256Bits=>"f256");
pub fn inputs<T: Format>(count: usize, phase: u8) -> (Vec<T>, Vec<T>) {
    let input = (0..count)
        .map(|index| T::pattern(phase.wrapping_add(u8::try_from(index % 256).unwrap())))
        .collect::<Vec<_>>();
    (input.clone(), input)
}
pub fn verify<T: Format>(expected: &[T], output: &[T]) {
    assert_eq!(output.len(), expected.len() + 2);
    for (want, actual) in expected.iter().zip(output) {
        assert_eq!(want.encode_le().as_ref(), actual.encode_le().as_ref());
    }
    for tail in &output[expected.len()..] {
        assert_eq!(
            tail.encode_le().as_ref(),
            T::sentinel().encode_le().as_ref()
        );
    }
}
