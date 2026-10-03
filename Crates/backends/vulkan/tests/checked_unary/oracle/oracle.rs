//! Independent representation-only oracle; no PCU arithmetic or classifier is called.
#[rustfmt::skip]
use pcu_facade::{PcuCheckedFloat,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDispatchFloatUnaryOp,PcuFloatUnderflowPolicy,PcuExecutionFaultKind};
pub trait Float: PcuCheckedFloat + core::fmt::Debug {
    const FORMAT: u32;
    const BITS: u32;
    const SIGN: u64;
    const MAX: u64;
    const FRACTION: u32;
    fn raw(self) -> u64;
    fn from_raw(bits: u64) -> Self;
}
macro_rules! narrow {
    ($ty:ty,$native:ty,$format:expr,$bits:expr,$max:expr,$fraction:expr) => {
        impl Float for $ty {
            const FORMAT: u32 = $format;
            const BITS: u32 = $bits;
            const SIGN: u64 = 1 << ($bits - 1);
            const MAX: u64 = $max;
            const FRACTION: u32 = $fraction;
            fn raw(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn from_raw(bits: u64) -> Self {
                Self::from_bits(<$native>::try_from(bits).unwrap())
            }
        }
    };
}
narrow!(PcuF16Bits, u16, 0, 16, 0x7bff, 10);
narrow!(PcuBf16Bits, u16, 1, 16, 0x7f7f, 7);
narrow!(PcuF8E4M3FnBits, u8, 2, 8, 0x7e, 3);
narrow!(PcuF8E5M2Bits, u8, 3, 8, 0x7b, 2);
narrow!(f32, u32, 4, 32, 0x7f7f_ffff, 23);
impl Float for f64 {
    const FORMAT: u32 = 5;
    const BITS: u32 = 64;
    const SIGN: u64 = 1 << 63;
    const MAX: u64 = 0x7fef_ffff_ffff_ffff;
    const FRACTION: u32 = 52;
    fn raw(self) -> u64 {
        self.to_bits()
    }
    fn from_raw(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
pub fn evaluate<T: Float>(
    bits: u64,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(u64, bool), PcuExecutionFaultKind> {
    let magnitude = bits & (T::SIGN - 1);
    if magnitude > T::MAX {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    let bits = match op {
        PcuDispatchFloatUnaryOp::Neg => bits ^ T::SIGN,
        PcuDispatchFloatUnaryOp::Relu => {
            if bits & T::SIGN == 0 && magnitude != 0 {
                bits
            } else {
                0
            }
        }
    };
    let magnitude = bits & (T::SIGN - 1);
    Ok((
        bits,
        policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
            && magnitude != 0
            && magnitude < 1 << T::FRACTION,
    ))
}
pub fn same<T: Float>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.raw(), b.raw());
    }
}
