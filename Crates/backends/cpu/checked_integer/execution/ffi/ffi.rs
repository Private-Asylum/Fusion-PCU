//! Isolated native carrier access with complete checked span and publication proofs.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFaultKind,
    PcuRangePolicy,
    PcuExecutionFault,
    PcuClampedFault,
};
use super::{Executable, PcuCpuCheckedIntegerError};

pub const fn prepare<T: PcuCheckedInteger>(
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    left_broadcast: bool,
    right_broadcast: bool,
) -> Executable {
    macro_rules! layout {
        ($op:expr,$clamp:expr) => {
            match (left_broadcast, right_broadcast) {
                (false, false) => execute::<T, $op, $clamp, false, false>,
                (false, true) => execute::<T, $op, $clamp, false, true>,
                (true, false) => execute::<T, $op, $clamp, true, false>,
                (true, true) => execute::<T, $op, $clamp, true, true>,
            }
        };
    }
    match (op, range) {
        (PcuDispatchIntegerBinaryOp::Add, PcuRangePolicy::Reject) => layout!(0, false),
        (PcuDispatchIntegerBinaryOp::Sub, PcuRangePolicy::Reject) => layout!(1, false),
        (PcuDispatchIntegerBinaryOp::Mul, PcuRangePolicy::Reject) => layout!(2, false),
        (PcuDispatchIntegerBinaryOp::Add, PcuRangePolicy::Clamp) => layout!(0, true),
        (PcuDispatchIntegerBinaryOp::Sub, PcuRangePolicy::Clamp) => layout!(1, true),
        (PcuDispatchIntegerBinaryOp::Mul, PcuRangePolicy::Clamp) => layout!(2, true),
    }
}

fn execute<
    T: PcuCheckedInteger,
    const OP: u8,
    const CLAMP: bool,
    const LEFT_BROADCAST: bool,
    const RIGHT_BROADCAST: bool,
>(
    left: &[u8],
    right: &[u8],
    output: &mut [u8],
    extent: usize,
) -> Result<(), PcuCpuCheckedIntegerError> {
    // The only entry is a fully admitted, synchronous prepared call. Truncate once before
    // pointer creation to establish each complete access span independently of loop bounds.
    // Extent multiplication has already been checked for every original loaded binding.
    let left_size = if LEFT_BROADCAST {
        T::HOST_SIZE
    } else {
        extent * T::HOST_SIZE
    };
    let right_size = if RIGHT_BROADCAST {
        T::HOST_SIZE
    } else {
        extent * T::HOST_SIZE
    };
    let left = left[..left_size].as_ptr().cast::<T>();
    let right = right[..right_size].as_ptr().cast::<T>();
    let output = output[..extent * T::HOST_SIZE].as_mut_ptr().cast::<T>();
    let load = |invocation: usize| {
        // SAFETY: These spans were checked before entering either loop; broadcast accesses
        // element zero and other accesses are strictly below extent. Cold admission restricts
        // the sealed core trait to ten padding-free primitives and four transparent limb carriers; every initialized bit pattern
        // is valid. read_unaligned creates no aligned T reference and preserves native bits.
        unsafe {
            (
                left.add(if LEFT_BROADCAST { 0 } else { invocation })
                    .read_unaligned(),
                right
                    .add(if RIGHT_BROADCAST { 0 } else { invocation })
                    .read_unaligned(),
            )
        }
    };
    if CLAMP {
        let mut recovered = None;
        for invocation in 0..extent {
            let (lhs, rhs) = load(invocation);
            let value = match evaluate_clamped::<T, OP>(lhs, rhs) {
                Ok(value) => value,
                Err(fault) => {
                    recovered.get_or_insert_with(|| PcuExecutionFault {
                        recovered: true,
                        kind: fault.kind(),
                        invocation_id: u64::try_from(invocation)
                            .expect("admitted u32 logical extent"),
                    });
                    fault.clamped_value()
                }
            };
            // SAFETY: Full schema/span preflight precedes this loop. Sealed Add/Sub/Mul
            // Clamp returns only a complete value or a useful range value, never a fatal
            // arithmetic condition. Exclusive output is disjoint and invocation is in bounds.
            unsafe {
                output.add(invocation).write_unaligned(value);
            }
        }
        return recovered.map_or(Ok(()), |fault| Err(PcuCpuCheckedIntegerError::Fault(fault)));
    }
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        evaluate::<T, OP>(lhs, rhs).map_err(|kind| super::super::fault(invocation, kind))?;
    }
    // Input spans may alias each other (including repeated SSA operands), but safe host
    // argument construction gives output an exclusive Rust borrow disjoint from both.
    // Inputs cannot change during this synchronous call, so exact preflight proves publication.
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        let result = evaluate::<T, OP>(lhs, rhs).expect("checked preflight succeeded");
        // SAFETY: The unique output span has extent complete native T slots, invocation is
        // in bounds, and write_unaligned requires no T alignment. No pointer escapes the call.
        unsafe {
            output.add(invocation).write_unaligned(result);
        }
    }
    Ok(())
}
fn evaluate<T: PcuCheckedInteger, const OP: u8>(
    lhs: T,
    rhs: T,
) -> Result<T, PcuExecutionFaultKind> {
    match OP {
        0 => lhs.pcu_checked_add(rhs),
        1 => lhs.pcu_checked_sub(rhs),
        2 => lhs.pcu_checked_mul(rhs),
        _ => unreachable!("cold constructor selects only checked Add/Sub/Mul"),
    }
}

fn evaluate_clamped<T: PcuCheckedInteger, const OP: u8>(
    lhs: T,
    rhs: T,
) -> Result<T, PcuClampedFault<T>> {
    match OP {
        0 => lhs.pcu_clamped_add(rhs),
        1 => lhs.pcu_clamped_sub(rhs),
        2 => lhs.pcu_clamped_mul(rhs),
        _ => unreachable!("cold selects only integer Add/Sub/Mul"),
    }
}
