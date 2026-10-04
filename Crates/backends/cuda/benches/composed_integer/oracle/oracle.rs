//! Independent immutable goldens plus Rust primitive laws; no CUDA emitter is used.
#[rustfmt::skip]
use fusion_pcu::{PcuCheckedInteger,PcuExecutionFaultKind,PcuScalar,PcuRangePolicy};
pub trait Format: PcuCheckedInteger + std::fmt::Debug {
    const LABEL: &'static str;
    const SIGNED: bool;
    fn from_bytes(bytes: &[u8]) -> Self;
}
macro_rules! formats {($($ty:ty => $label:literal,$signed:literal;)+)=>{$(
 impl Format for $ty {
 const LABEL:&'static str=$label;const SIGNED:bool=$signed;
 fn from_bytes(bytes:&[u8])->Self{Self::decode_le(bytes.try_into().unwrap())}
 }
)+};}
formats! {
 u8=>"u8",false;i8=>"i8",true;u16=>"u16",false;i16=>"i16",true;
 u32=>"u32",false;i32=>"i32",true;u64=>"u64",false;i64=>"i64",true;
 u128=>"u128",false;i128=>"i128",true;
 fusion_pcu::PcuU256=>"u256",false;fusion_pcu::PcuI256=>"i256",true;
 fusion_pcu::PcuU512=>"u512",false;fusion_pcu::PcuI512=>"i512",true;
}
pub fn small<T: Format>(v: u8) -> T {
    let mut bytes = [0; 64];
    bytes[0] = v;
    T::from_bytes(&bytes[..T::ENCODED_SIZE])
}
pub fn limit<T: Format>(lower: bool) -> T {
    let mut bytes = [if lower { 0 } else { 255 }; 64];
    if T::SIGNED {
        bytes[T::ENCODED_SIZE - 1] = if lower { 128 } else { 127 };
    }
    T::from_bytes(&bytes[..T::ENCODED_SIZE])
}
pub fn bits<T: PcuScalar>(a: &[T], b: &[T]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
pub fn code(kind: PcuExecutionFaultKind) -> u32 {
    match kind {
        PcuExecutionFaultKind::ArithmeticOverflow => 3,
        PcuExecutionFaultKind::ArithmeticUnderflow => 4,
        _ => panic!("noninteger arithmetic fault"),
    }
}
pub fn law<T: Format>(a: T, b: T, dead: bool) -> (T, T, u32) {
    let mut first = 0;
    let mut apply = |v: Result<T, PcuExecutionFaultKind>| match v {
        Ok(v) => v,
        Err(kind) => {
            if first == 0 {
                first = code(kind);
            }
            limit::<T>(kind == PcuExecutionFaultKind::ArithmeticUnderflow)
        }
    };
    let stage = apply(a.pcu_checked_add(b));
    let output = if dead {
        a
    } else {
        let p = apply(stage.pcu_checked_mul(a));
        apply(p.pcu_checked_sub(a))
    };
    (stage, output, first)
}
pub struct Golden<T> {
    pub input: T,
    pub seed: T,
    pub stage: T,
    pub output: T,
    pub first: u32,
}
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
pub fn goldens<T: Format>(range: PcuRangePolicy, dead: bool) -> Vec<Golden<T>> {
    let all = concat!(
        include_str!("../../../tests/portable_narrow_composed/vectors.txt"),
        "\n",
        include_str!("../../../tests/wide_composed/vectors.txt")
    );
    all.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .filter_map(|line| {
            let f = line.split_whitespace().collect::<Vec<_>>();
            if f[0] != T::LABEL
                || f[1]
                    != if range == PcuRangePolicy::Clamp {
                        "1"
                    } else {
                        "0"
                    }
            {
                return None;
            }
            let a = decode::<T>(f[2]);
            let b = decode::<T>(f[3]);
            let (stage, output, first) = law(a, b, dead);
            if !dead {
                assert_eq!(first, f[6].parse::<u32>().unwrap());
                if range == PcuRangePolicy::Clamp || first == 0 {
                    bits(&[stage], &[decode::<T>(f[4])]);
                    bits(&[output], &[decode::<T>(f[5])]);
                }
            }
            Some(Golden {
                input: a,
                seed: b,
                stage,
                output,
                first,
            })
        })
        .collect()
}
