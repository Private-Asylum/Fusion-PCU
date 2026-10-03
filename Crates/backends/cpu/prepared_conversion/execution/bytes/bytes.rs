//! Integer-significand rounding, with complete fault validation before native-byte publication.
//! IEEE nearest-even and after-rounding tininess are exact; PCU finite-input rejection and
//! observable range recovery depart from IEEE default exception values and status flags.
#[rustfmt::skip]
use fusion_pcu::{PcuCheckedFloatConversion,PcuCheckedFloatWidening,PcuClampedFloatConversion,PcuClampedError,PcuRangePolicy,PcuDispatchCheckedFloatConversion,PcuFloatUnderflowPolicy,PcuExecutionFault,PcuExecutionFaultKind};
use crate::PcuCpuHostError;
pub(in crate::prepared_conversion) type Executable =
    fn(&[u8], &mut [u8], usize, PcuFloatUnderflowPolicy) -> Result<(), PcuCpuHostError>;
pub(in crate::prepared_conversion) const fn prepare(
    conversion: PcuDispatchCheckedFloatConversion,
    range: PcuRangePolicy,
    broadcast: bool,
) -> Executable {
    macro_rules! layout {
        ($runner:ident) => {
            match (range, broadcast) {
                (PcuRangePolicy::Reject, false) => $runner::<false, false>,
                (PcuRangePolicy::Reject, true) => $runner::<true, false>,
                (PcuRangePolicy::Clamp, false) => $runner::<false, true>,
                (PcuRangePolicy::Clamp, true) => $runner::<true, true>,
            }
        };
    }
    match conversion {
        PcuDispatchCheckedFloatConversion::F32ToF64 => layout!(widen),
        PcuDispatchCheckedFloatConversion::F64ToF32 => layout!(narrow),
    }
}
const fn fault(invocation: usize, kind: PcuExecutionFaultKind, recovered: bool) -> PcuCpuHostError {
    PcuCpuHostError::Fault(PcuExecutionFault {
        kind,
        invocation_id: invocation as u64,
        recovered,
    })
}
macro_rules! convert {
    ($name:ident,$source:ty,$destination:ty,$evaluate:expr) => {
        #[allow(clippy::cast_ptr_alignment)] // Every native access is unaligned; no aligned reference is formed.
        fn $name<const BROADCAST: bool, const CLAMP: bool>(
            input: &[u8],
            output: &mut [u8],
            extent: usize,
            underflow: PcuFloatUnderflowPolicy,
        ) -> Result<(), PcuCpuHostError> {
            let source_elements = if BROADCAST { 1 } else { extent };
            let input = input[..source_elements * size_of::<$source>()]
                .as_ptr()
                .cast::<$source>();
            let output = output[..extent * size_of::<$destination>()]
                .as_mut_ptr()
                .cast::<$destination>();
            let load = |invocation: usize| {
                // SAFETY: Full padding-free primitive spans were validated before entry;
                // the bounded slot is initialized, no aligned reference or pointer escapes.
                unsafe {
                    input
                        .add(if BROADCAST { 0 } else { invocation })
                        .read_unaligned()
                }
            };
            let evaluate = $evaluate;
            let mut first = None;
            for invocation in 0..extent {
                match evaluate(load(invocation), underflow) {
                    Ok(_) => {}
                    Err(PcuClampedError::Range(range)) => {
                        if first.is_none() {
                            first = Some(fault(invocation, range.kind(), true));
                        }
                    }
                    // A later fatal operand always wins over an earlier recoverable range.
                    Err(PcuClampedError::Fatal(kind)) => {
                        return Err(fault(invocation, kind, false));
                    }
                }
            }
            // Safe Rust arguments retain disjoint source/exclusive output borrows. Exact
            // validation proves this repeated evaluation has a value for every output slot.
            for invocation in 0..extent {
                let value = match evaluate(load(invocation), underflow) {
                    Ok(value) => value,
                    Err(PcuClampedError::Range(range)) => range.clamped_value(),
                    Err(PcuClampedError::Fatal(_)) => {
                        unreachable!("immutable checked preflight succeeded")
                    }
                };
                // SAFETY: Unique complete destination span; bounded native slot and no pointer escapes.
                unsafe {
                    output.add(invocation).write_unaligned(value);
                }
            }
            first.map_or(Ok(()), Err)
        }
    };
}
convert!(widen, f32, f64, |value: f32, _underflow| value
    .pcu_checked_to_f64()
    .map_err(PcuClampedError::Fatal));
convert!(narrow, f64, f32, |value: f64, underflow| if CLAMP {
    value.pcu_clamped_to_f32_with_policy(underflow)
} else {
    value
        .pcu_checked_to_f32_with_policy(underflow)
        .map_err(PcuClampedError::Fatal)
});
