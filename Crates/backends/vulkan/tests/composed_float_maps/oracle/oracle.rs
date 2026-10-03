//! Independent representation constants and bounded exact-dyadic native controls.
#[path = "independent/independent.rs"]
pub mod independent;
#[rustfmt::skip]
use pcu_facade::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};
pub trait Format: pcu_facade::PcuCheckedFloat {
    const SIGN: u64;
    const MAX: u64;
    const FRACTION: u32;
    const ONE: u64;
    const MIN_NORMAL: u64 = 1 << Self::FRACTION;
    fn bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;
    fn from(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    fn native_value(self) -> Self;
}
macro_rules! low {
    ($ty:ty, $raw:ty, $sign:expr, $max:expr, $fraction:expr, $one:expr) => {
        impl Format for $ty {
            const SIGN: u64 = $sign;
            const MAX: u64 = $max;
            const FRACTION: u32 = $fraction;
            const ONE: u64 = $one;
            fn bits(self) -> u64 {
                u64::from(self.to_bits())
            }
            fn from_bits(bits: u64) -> Self {
                Self::from_bits(<$raw>::try_from(bits).unwrap())
            }
            fn native_value(self) -> Self {
                independent::expected(
                    self,
                    PcuFloatUnderflowPolicy::IeeeAfterRounding,
                    PcuRangePolicy::Reject,
                )
                .1
                .unwrap()
            }
        }
    };
}
low!(pcu_facade::PcuF16Bits, u16, 0x8000, 0x7bff, 10, 0x3c00);
low!(pcu_facade::PcuBf16Bits, u16, 0x8000, 0x7f7f, 7, 0x3f80);
low!(pcu_facade::PcuF8E4M3FnBits, u8, 0x80, 0x7e, 3, 0x38);
low!(pcu_facade::PcuF8E5M2Bits, u8, 0x80, 0x7b, 2, 0x3c);
impl Format for f32 {
    const SIGN: u64 = 1 << 31;
    const MAX: u64 = 0x7f7f_ffff;
    const FRACTION: u32 = 23;
    const ONE: u64 = 0x3f80_0000;
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
    fn native_value(self) -> Self {
        (self + self) * self
    }
}
impl Format for f64 {
    const SIGN: u64 = 1 << 63;
    const MAX: u64 = 0x7fef_ffff_ffff_ffff;
    const FRACTION: u32 = 52;
    const ONE: u64 = 0x3ff0_0000_0000_0000;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    fn native_value(self) -> Self {
        (self + self) * self
    }
}

pub fn dyadic<T: Format>(phase: usize) -> (T, T) {
    let exponent = phase % 3;
    let input = match exponent {
        0 => T::ONE - T::MIN_NORMAL,
        1 => T::ONE,
        _ => T::ONE + T::MIN_NORMAL,
    };
    let result = match exponent {
        0 => T::ONE - T::MIN_NORMAL,
        1 => T::ONE + T::MIN_NORMAL,
        _ => T::ONE + 3 * T::MIN_NORMAL,
    };
    (
        T::from(input | if phase.is_multiple_of(2) { 0 } else { T::SIGN }),
        T::from(result),
    )
}
pub fn fault(status: u64, lane: usize) -> Option<PcuExecutionFault> {
    if status == u64::MAX {
        return None;
    }
    Some(PcuExecutionFault {
        invocation_id: u64::try_from(lane).unwrap(),
        recovered: status & (1 << 63) != 0,
        kind: match status & 0xff {
            3 => PcuExecutionFaultKind::ArithmeticOverflow,
            4 => PcuExecutionFaultKind::ArithmeticUnderflow,
            5 => PcuExecutionFaultKind::InvalidFloatingOperand,
            other => panic!("unexpected independent model status {other}"),
        },
    })
}
// This peer is deliberately bounded to exact +/-{0.5,1,2}. It executes the same
// two arithmetic steps and validates the independent literal encoding result.
// No general F32/F64 floating-environment conformance follows from these controls.
pub fn dyadic_native<T: Format>(
    input: &[T],
    output: &mut [T],
    count: usize,
) -> Result<(), PcuExecutionFault> {
    for (lane, value) in input[..count].iter().enumerate() {
        if value.bits() & (T::SIGN - 1) > T::MAX {
            return Err(PcuExecutionFault {
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: u64::try_from(lane).unwrap(),
                recovered: false,
            });
        }
        assert!((0..6).any(|phase| dyadic::<T>(phase).0.bits() == value.bits()));
    }
    for (value, target) in input[..count].iter().zip(&mut output[..count]) {
        let phase = (0..6)
            .find(|phase| dyadic::<T>(*phase).0.bits() == value.bits())
            .unwrap();
        *target = value.native_value();
        assert_eq!(target.bits(), dyadic::<T>(phase).1.bits());
    }
    Ok(())
}

pub fn low_expected<T: Format>(
    input: T,
    underflow: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> (Option<PcuExecutionFault>, Option<T>) {
    let (status, output) = independent::expected(input, underflow, range);
    (fault(status, 0), output)
}
