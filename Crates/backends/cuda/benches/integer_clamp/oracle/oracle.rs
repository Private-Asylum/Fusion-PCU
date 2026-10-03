//! Arbitrary-precision integer goldens independently generated outside the core implementation.
#[rustfmt::skip]
use fusion_pcu::{PcuScalar,PcuCheckedInteger};
pub trait Format: PcuCheckedInteger + std::fmt::Debug {
    const LABEL: &'static str;
    fn from_bytes(bytes: &[u8]) -> Self;
    fn small(value: u8) -> Self;
    fn sentinel() -> Self {
        Self::small(17)
    }
}
macro_rules! widths {($($ty:ty=>$label:literal),+$(,)?)=>{$(
 impl Format for $ty {
 const LABEL:&'static str=$label;
 fn from_bytes(bytes:&[u8])->Self {Self::decode_le(bytes.try_into().unwrap())}
 fn small(value:u8)->Self {let bytes=Self::ENCODED_SIZE;let mut result=vec![0;bytes];result[0]=value;Self::from_bytes(&result)}
 }
)+};}
widths!(i8=>"i8",u8=>"u8",i16=>"i16",u16=>"u16",i32=>"i32",u32=>"u32",i64=>"i64",u64=>"u64",i128=>"i128",u128=>"u128",fusion_pcu::PcuI256=>"i256",fusion_pcu::PcuU256=>"u256",fusion_pcu::PcuI512=>"i512",fusion_pcu::PcuU512=>"u512");
fn decode<T: Format>(hex: &str) -> T {
    let bytes = hex
        .as_bytes()
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect::<Vec<_>>();
    T::from_bytes(&bytes)
}
pub fn rows<T: Format>(operation: u32) -> Vec<(T, T, T, u32)> {
    include_str!("vectors.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            (fields[0] == T::LABEL && fields[1].parse::<u32>().unwrap() == operation).then(|| {
                (
                    decode(fields[2]),
                    decode(fields[3]),
                    decode(fields[4]),
                    fields[5].parse().unwrap(),
                )
            })
        })
        .collect()
}
pub fn inputs<T: Format>(
    count: usize,
    phase: usize,
    operation: u32,
) -> (Vec<T>, Vec<T>, Vec<T>, u64) {
    let rows = rows::<T>(operation);
    let mut left = Vec::with_capacity(count);
    let mut right = Vec::with_capacity(count);
    let mut expected = Vec::with_capacity(count);
    let mut status = u64::MAX;
    for i in 0..count {
        let (a, b, want, code) = rows[(i + phase) % rows.len()];
        left.push(a);
        right.push(b);
        expected.push(want);
        if status == u64::MAX && code != 0 {
            status = 0x8000_0000_0000_0000 | (u64::try_from(i).unwrap() << 3) | u64::from(code);
        }
    }
    assert_ne!(status, u64::MAX);
    (left, right, expected, status)
}
pub fn result(actual: &Result<(), fusion_pcu::PcuExecutionError>, word: u64) {
    let kind = if word & 7 == 3 {
        fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
    } else {
        fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
    };
    let id = (word & 0x7fff_ffff_ffff_ffff) >> 3;
    assert!(
        matches!(actual,Err(fusion_pcu::PcuExecutionError::ArithmeticFault(f)) if f.recovered&&f.kind==kind&&f.invocation_id==id),
        "{actual:?}; expected {word:#x}"
    );
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
