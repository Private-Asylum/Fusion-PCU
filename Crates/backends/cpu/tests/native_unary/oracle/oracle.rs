//! Independent raw binary32/binary64 sign and selection oracle; no PCU numeric/classifier call.
#[rustfmt::skip]
use pcu_facade::{PcuCheckedFloat,PcuDispatchFloatUnaryOp,PcuExecutionFault,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuRangePolicy};
pub trait Native: PcuCheckedFloat + core::fmt::Debug {
    const SIGN: u64;
    const MAX: u64;
    const FRACTION: u32;
    fn bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;
}
impl Native for f32 {
    const SIGN: u64 = 1 << 31;
    const MAX: u64 = 0x7f7f_ffff;
    const FRACTION: u32 = 23;
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
}
impl Native for f64 {
    const SIGN: u64 = 1 << 63;
    const MAX: u64 = 0x7fef_ffff_ffff_ffff;
    const FRACTION: u32 = 52;
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(bits)
    }
}
pub fn evaluate<T: Native>(
    bits: u64,
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(u64, bool), PcuExecutionFaultKind> {
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
    Ok((
        result,
        policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
            && magnitude != 0
            && magnitude < 1 << T::FRACTION,
    ))
}
pub fn native<T: Native, const N: usize>(
    input: &[T],
    output: &mut [T],
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    native_layout::<T, N, false>(input, output, op, policy, range)
}
pub fn native_broadcast<T: Native, const N: usize>(
    input: &[T],
    output: &mut [T],
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    native_layout::<T, N, true>(input, output, op, policy, range)
}
fn native_layout<T: Native, const N: usize, const BROADCAST: bool>(
    input: &[T],
    output: &mut [T],
    op: PcuDispatchFloatUnaryOp,
    policy: PcuFloatUnderflowPolicy,
    range: PcuRangePolicy,
) -> Result<(), PcuExecutionFault> {
    assert!(input.len() >= if BROADCAST { 1 } else { N } && output.len() >= N);
    let mut recovered = None;
    for index in 0..N {
        let value = input[if BROADCAST { 0 } else { index }];
        let fault = match evaluate::<T>(value.bits(), op, policy) {
            Ok((_, false)) => continue,
            Ok((_, true)) => PcuExecutionFault {
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                recovered: range == PcuRangePolicy::Clamp,
                invocation_id: index as u64,
            },
            Err(kind) => PcuExecutionFault {
                kind,
                recovered: false,
                invocation_id: index as u64,
            },
        };
        if !fault.recovered {
            return Err(fault);
        }
        recovered.get_or_insert(fault);
    }
    for (index, destination) in output[..N].iter_mut().enumerate() {
        *destination = T::from_bits(
            evaluate::<T>(input[if BROADCAST { 0 } else { index }].bits(), op, policy)
                .unwrap()
                .0,
        );
    }
    recovered.map_or(Ok(()), Err)
}
pub fn bits<T: Native>(values: &[T]) -> Vec<u64> {
    values.iter().map(|value| value.bits()).collect()
}
