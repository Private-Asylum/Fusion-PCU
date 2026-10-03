//! Independent exact sign/selection bit banks; no core arithmetic or backend classifier.
#[rustfmt::skip]
use pcu_facade::{
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuDispatchFloatUnaryOp as Op,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
pub trait Sample: PcuCheckedFloat {
    const SIGN: u64;
    const FRACTION: u32;
    fn raw(bits: u64) -> Self;
    fn bits(self) -> u64;
}
macro_rules! small {
    ($ty:ty, $bits:ty, $sign:expr, $fraction:expr) => {
        impl Sample for $ty {
            const SIGN: u64 = $sign;
            const FRACTION: u32 = $fraction;
            fn raw(bits: u64) -> Self {
                Self::from_bits(<$bits>::try_from(bits).unwrap())
            }
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
        }
    };
}
small!(PcuF16Bits, u16, 0x8000, 10);
small!(PcuBf16Bits, u16, 0x8000, 7);
small!(PcuF8E4M3FnBits, u8, 0x80, 3);
small!(PcuF8E5M2Bits, u8, 0x80, 2);
small!(f32, u32, 0x8000_0000, 23);
impl Sample for f64 {
    const SIGN: u64 = 0x8000_0000_0000_0000;
    const FRACTION: u32 = 52;
    fn raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}
pub fn compare<T: Sample>(left: &[T], right: &[T]) {
    assert_eq!(left.len(), right.len());
    for (left, right) in left.iter().zip(right) {
        assert_eq!(left.bits(), right.bits());
    }
}
pub fn bank<T: Sample>(count: usize, full: usize, phase: usize) -> Vec<T> {
    (0..full)
        .map(|lane| {
            let bits = if lane >= count {
                T::SIGN - 1
            } else {
                match (lane + phase) % 4 {
                    0 => 0,
                    1 => T::SIGN,
                    2 => (1 << T::FRACTION) + u64::try_from(phase % 3).unwrap(),
                    _ => T::SIGN | ((1 << T::FRACTION) + u64::try_from(phase % 3).unwrap()),
                }
            };
            T::raw(bits)
        })
        .collect()
}
pub fn expected<T: Sample>(input: &[T], count: usize, op: Op, broadcast: bool) -> Vec<T> {
    (0..count)
        .map(|lane| {
            let bits = input[if broadcast { 0 } else { lane }].bits();
            T::raw(if op == Op::Neg {
                bits ^ T::SIGN
            } else if bits & T::SIGN == 0 && bits != 0 {
                bits
            } else {
                0
            })
        })
        .collect()
}
