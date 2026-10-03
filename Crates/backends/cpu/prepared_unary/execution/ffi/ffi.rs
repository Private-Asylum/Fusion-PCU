//! IEEE exact sign manipulation/selection has no rounding or floating environment dependency.
//! PCU rejects nonfinite inputs and `ReLU` canonicalizes nonpositive operands to +0.
//! Named OFP8 carriers retain their OCP encoding; no native low-precision instruction is claimed.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,PcuClampedFloat,PcuClampedError,PcuDispatchFloatUnaryOp,
    PcuRangePolicy,PcuScalarType,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,PcuExecutionFault,
};
use crate::PcuCpuHostError;
pub(in crate::prepared_unary) type Executable =
    fn(&[u8], &mut [u8], usize, PcuFloatUnderflowPolicy) -> Result<(), PcuCpuHostError>;
pub(in crate::prepared_unary) const fn prepare<T: PcuCheckedFloat>(
    op: PcuDispatchFloatUnaryOp,
    range: PcuRangePolicy,
    broadcast: bool,
) -> Result<Executable, PcuCpuHostError> {
    match T::TYPE {
        PcuScalarType::F32 => Ok(select::<f32>(op, range, broadcast)),
        PcuScalarType::F64 => Ok(select::<f64>(op, range, broadcast)),
        PcuScalarType::F16 => Ok(select::<PcuF16Bits>(op, range, broadcast)),
        PcuScalarType::BF16 => Ok(select::<PcuBf16Bits>(op, range, broadcast)),
        PcuScalarType::F8E4M3FN => Ok(select::<PcuF8E4M3FnBits>(op, range, broadcast)),
        PcuScalarType::F8E5M2 => Ok(select::<PcuF8E5M2Bits>(op, range, broadcast)),
        _ => Err(PcuCpuHostError::UnsupportedProfile),
    }
}
const fn select<T: PcuClampedFloat>(
    op: PcuDispatchFloatUnaryOp,
    range: PcuRangePolicy,
    broadcast: bool,
) -> Executable {
    macro_rules! layout {
        ($op:expr) => {
            match (range, broadcast) {
                (PcuRangePolicy::Reject, false) => execute::<T, $op, false, false>,
                (PcuRangePolicy::Reject, true) => execute::<T, $op, false, true>,
                (PcuRangePolicy::Clamp, false) => execute::<T, $op, true, false>,
                (PcuRangePolicy::Clamp, true) => execute::<T, $op, true, true>,
            }
        };
    }
    match op {
        PcuDispatchFloatUnaryOp::Neg => layout!(false),
        PcuDispatchFloatUnaryOp::Relu => layout!(true),
    }
}
fn evaluate<T: PcuClampedFloat, const RELU: bool>(
    value: T,
    policy: PcuFloatUnderflowPolicy,
) -> Result<T, PcuClampedError<T>> {
    if RELU {
        value.pcu_clamped_relu_with_policy(policy)
    } else {
        value.pcu_clamped_neg_with_policy(policy)
    }
}
fn execute<T: PcuClampedFloat, const RELU: bool, const CLAMP: bool, const BROADCAST: bool>(
    input: &[u8],
    output: &mut [u8],
    extent: usize,
    policy: PcuFloatUnderflowPolicy,
) -> Result<(), PcuCpuHostError> {
    // Exact original schemas and checked extent multiplication precede this private entry.
    let input = input[..if BROADCAST {
        T::HOST_SIZE
    } else {
        extent * T::HOST_SIZE
    }]
        .as_ptr()
        .cast::<T>();
    let output = output[..extent * T::HOST_SIZE].as_mut_ptr().cast::<T>();
    let load = |index: usize| {
        // SAFETY: Sealed padding-free carriers admit every initialized bit pattern. The
        // complete input span is checked; broadcast uses zero, other indices are below extent.
        // Unaligned read creates no aligned reference, and no pointer escapes this call.
        unsafe {
            input
                .add(if BROADCAST { 0 } else { index })
                .read_unaligned()
        }
    };
    let mut recovered = None;
    for invocation in 0..extent {
        match evaluate::<T, RELU>(load(invocation), policy) {
            Ok(_) => {}
            Err(error) => {
                let fault = PcuExecutionFault {
                    recovered: CLAMP && matches!(error, PcuClampedError::Range(_)),
                    kind: error.kind(),
                    invocation_id: invocation as u64,
                };
                if !fault.recovered {
                    return Err(PcuCpuHostError::Fault(fault));
                }
                recovered.get_or_insert(fault);
            }
        }
    }
    // Every lane is fatal-free. Borrowed input is immutable and disjoint from the exclusive
    // output for this synchronous call; no earlier recovered lane hides a later fatal fault.
    for invocation in 0..extent {
        let value = match evaluate::<T, RELU>(load(invocation), policy) {
            Ok(value) => value,
            Err(PcuClampedError::Range(fault)) => fault.clamped_value(),
            Err(PcuClampedError::Fatal(_)) => unreachable!("fatal-free scan precedes publication"),
        };
        // SAFETY: Exclusive output has extent complete slots, invocation is strictly in
        // bounds, and unaligned writing preserves carrier bits without an aligned reference.
        unsafe {
            output.add(invocation).write_unaligned(value);
        }
    }
    recovered.map_or(Ok(()), |fault| Err(PcuCpuHostError::Fault(fault)))
}
