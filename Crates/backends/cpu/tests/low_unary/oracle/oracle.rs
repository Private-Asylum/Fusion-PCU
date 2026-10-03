//! Independent finite encoding oracle: no core arithmetic/classifier/conversion or backend calls.
#[rustfmt::skip]
use pcu_facade::{PcuCheckedFloat,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDispatchFloatUnaryOp,PcuExecutionFault,PcuExecutionFaultKind,PcuRangePolicy,PcuFloatUnderflowPolicy};
pub trait Low: PcuCheckedFloat + Eq + core::fmt::Debug {
    const SIGN: u16;
    const MAX: u16;
    const FRACTION: u32;
    fn bits(self) -> u16;
    fn from_bits(bits: u16) -> Self;
}
macro_rules! half {
    ($ty:ty,$max:expr,$fraction:expr) => {
        impl Low for $ty {
            const SIGN: u16 = 0x8000;
            const MAX: u16 = $max;
            const FRACTION: u32 = $fraction;
            fn bits(self) -> u16 {
                self.to_bits()
            }
            fn from_bits(bits: u16) -> Self {
                Self::from_bits(bits)
            }
        }
    };
}
half!(PcuF16Bits, 0x7bff, 10);
half!(PcuBf16Bits, 0x7f7f, 7);
macro_rules! fp8 {
    ($ty:ty,$max:expr,$fraction:expr) => {
        impl Low for $ty {
            const SIGN: u16 = 0x80;
            const MAX: u16 = $max;
            const FRACTION: u32 = $fraction;
            fn bits(self) -> u16 {
                u16::from(self.to_bits())
            }
            fn from_bits(bits: u16) -> Self {
                Self::from_bits(u8::try_from(bits).unwrap())
            }
        }
    };
}
fp8!(PcuF8E4M3FnBits, 0x7e, 3);
fp8!(PcuF8E5M2Bits, 0x7b, 2);
pub fn evaluate<T: Low>(
    bits: u16,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(u16, bool), PcuExecutionFaultKind> {
    let magnitude = bits & (T::SIGN - 1);
    if magnitude > T::MAX {
        return Err(PcuExecutionFaultKind::InvalidFloatingOperand);
    }
    let result = match op {
        PcuDispatchFloatUnaryOp::Neg => bits ^ T::SIGN,
        PcuDispatchFloatUnaryOp::Relu => {
            if bits & T::SIGN == 0 && magnitude != 0 {
                bits
            } else {
                0
            }
        }
    };
    let magnitude = result & (T::SIGN - 1);
    // Sign manipulation/selection is exact: IEEE-after-rounding and gradual never fault.
    // RejectSubnormalResult tests the result, not an unselected negative input to ReLU.
    Ok((
        result,
        policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
            && magnitude != 0
            && magnitude < (1 << T::FRACTION),
    ))
}
pub fn native<T: Low, const N: usize>(
    input: &[T],
    output: &mut [T],
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    assert!(input.len() >= N && output.len() >= N);
    let mut recovered = None;
    for (index, value) in input[..N].iter().copied().enumerate() {
        let range_fault = match evaluate::<T>(value.bits(), op, policy) {
            Ok((_, range_fault)) => range_fault,
            Err(kind) => {
                return Err(PcuExecutionFault {
                    recovered: false,
                    kind,
                    invocation_id: index as u64,
                });
            }
        };
        if range_fault {
            let fault = PcuExecutionFault {
                recovered: range == PcuRangePolicy::Clamp,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                invocation_id: index as u64,
            };
            if !fault.recovered {
                return Err(fault);
            }
            recovered.get_or_insert(fault);
        }
    }
    for (value, destination) in input[..N].iter().copied().zip(&mut output[..N]) {
        *destination = T::from_bits(evaluate::<T>(value.bits(), op, policy).unwrap().0);
    }
    recovered.map_or(Ok(()), Err)
}
