//! Cold-frozen binary operations use core's integer-significand IEEE oracle.
//! Rounding is nearest ties-to-even; host rounding/FTZ/DAZ do not participate.
//! PCU rejects nonfinite operands and range faults; IEEE default infinities/status are not published.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedFloat,
    PcuClampedFloat,
    PcuClampedError,
    PcuRangePolicy,
    PcuScalarType,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuExecutionFault,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
};
use super::super::PcuCpuPreparedBinaryError;

pub(in crate::prepared_binary) type Executable = fn(
    &[u8],
    &[u8],
    &mut [u8],
    usize,
    PcuFloatUnderflowPolicy,
) -> Result<(), PcuCpuPreparedBinaryError>;

pub(in crate::prepared_binary) const fn prepare<T: PcuCheckedFloat>(
    op: PcuDispatchFloatBinaryOp,
    range: PcuRangePolicy,
    left_broadcast: bool,
    right_broadcast: bool,
) -> Executable {
    if matches!(range, PcuRangePolicy::Clamp) {
        // The checked-float trait is sealed to these exact representations; cold scalar
        // resolution avoids adding a consumer clamped-trait bound or any warm type switch.
        return match T::TYPE {
            PcuScalarType::F32 => prepare_clamped::<f32>(op, left_broadcast, right_broadcast),
            PcuScalarType::F64 => prepare_clamped::<f64>(op, left_broadcast, right_broadcast),
            PcuScalarType::F16 => {
                prepare_clamped::<PcuF16Bits>(op, left_broadcast, right_broadcast)
            }
            PcuScalarType::BF16 => {
                prepare_clamped::<PcuBf16Bits>(op, left_broadcast, right_broadcast)
            }
            PcuScalarType::F8E4M3FN => {
                prepare_clamped::<PcuF8E4M3FnBits>(op, left_broadcast, right_broadcast)
            }
            PcuScalarType::F8E5M2 => {
                prepare_clamped::<PcuF8E5M2Bits>(op, left_broadcast, right_broadcast)
            }
            _ => panic!("cold admission proved a sealed checked-float representation"),
        };
    }
    macro_rules! layout {
        ($op:expr) => {
            match (left_broadcast, right_broadcast) {
                (false, false) => execute::<T, $op, false, false>,
                (false, true) => execute::<T, $op, false, true>,
                (true, false) => execute::<T, $op, true, false>,
                (true, true) => execute::<T, $op, true, true>,
            }
        };
    }
    match op {
        PcuDispatchFloatBinaryOp::Add => layout!(0),
        PcuDispatchFloatBinaryOp::Sub => layout!(1),
        PcuDispatchFloatBinaryOp::Mul => layout!(2),
        PcuDispatchFloatBinaryOp::Div => layout!(3),
    }
}

fn execute<
    T: PcuCheckedFloat,
    const OP: u8,
    const LEFT_BROADCAST: bool,
    const RIGHT_BROADCAST: bool,
>(
    left: &[u8],
    right: &[u8],
    output: &mut [u8],
    extent: usize,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<(), PcuCpuPreparedBinaryError> {
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
        // element zero and other accesses are strictly below extent. The core trait is
        // sealed to six padding-free scalar representations and every initialized bit pattern
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
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        evaluate::<T, OP>(lhs, rhs, underflow)
            .map_err(|kind| super::super::fault(invocation, kind))?;
    }
    // Input spans may alias each other (including repeated SSA operands), but safe host
    // argument construction gives output an exclusive Rust borrow disjoint from both.
    // Inputs cannot change during this synchronous call, so exact preflight proves publication.
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        let result = evaluate::<T, OP>(lhs, rhs, underflow).expect("checked preflight succeeded");
        // SAFETY: The unique output span has extent complete native T slots, invocation is
        // in bounds, and write_unaligned requires no T alignment. No pointer escapes the call.
        unsafe {
            output.add(invocation).write_unaligned(result);
        }
    }
    Ok(())
}
fn evaluate<T: PcuCheckedFloat, const OP: u8>(
    lhs: T,
    rhs: T,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<T, PcuExecutionFaultKind> {
    match OP {
        0 => lhs.pcu_checked_add_with_policy(rhs, underflow),
        1 => lhs.pcu_checked_sub_with_policy(rhs, underflow),
        2 => lhs.pcu_checked_mul_with_policy(rhs, underflow),
        3 => lhs.pcu_checked_div_with_policy(rhs, underflow),
        _ => unreachable!("cold constructor selects only checked Add/Sub/Mul/Div"),
    }
}

const fn prepare_clamped<T: PcuClampedFloat>(
    op: PcuDispatchFloatBinaryOp,
    left_broadcast: bool,
    right_broadcast: bool,
) -> Executable {
    macro_rules! layout {
        ($op:expr) => {
            match (left_broadcast, right_broadcast) {
                (false, false) => execute_clamped::<T, $op, false, false>,
                (false, true) => execute_clamped::<T, $op, false, true>,
                (true, false) => execute_clamped::<T, $op, true, false>,
                (true, true) => execute_clamped::<T, $op, true, true>,
            }
        };
    }
    match op {
        PcuDispatchFloatBinaryOp::Add => layout!(0),
        PcuDispatchFloatBinaryOp::Sub => layout!(1),
        PcuDispatchFloatBinaryOp::Mul => layout!(2),
        PcuDispatchFloatBinaryOp::Div => layout!(3),
    }
}

fn execute_clamped<
    T: PcuClampedFloat,
    const OP: u8,
    const LEFT_BROADCAST: bool,
    const RIGHT_BROADCAST: bool,
>(
    left: &[u8],
    right: &[u8],
    output: &mut [u8],
    extent: usize,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<(), PcuCpuPreparedBinaryError> {
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
        // element zero and other accesses are strictly below extent. The core trait is
        // sealed to six padding-free scalar representations and every initialized bit pattern
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
    let mut recovered = None;
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        match evaluate_clamped::<T, OP>(lhs, rhs, underflow) {
            Ok(_) => {}
            Err(PcuClampedError::Range(fault)) => {
                if recovered.is_none() {
                    recovered = Some(PcuExecutionFault {
                        recovered: true,
                        kind: fault.kind(),
                        invocation_id: u64::try_from(invocation).expect("admitted u32 extent"),
                    });
                }
            }
            Err(PcuClampedError::Fatal(kind)) => return Err(super::super::fault(invocation, kind)),
        }
    }
    // Input spans may alias each other (including repeated SSA operands), but safe host
    // argument construction gives output an exclusive Rust borrow disjoint from both.
    // Inputs cannot change during this synchronous call, so exact preflight proves publication.
    for invocation in 0..extent {
        let (lhs, rhs) = load(invocation);
        let result = match evaluate_clamped::<T, OP>(lhs, rhs, underflow) {
            Ok(value) => value,
            Err(PcuClampedError::Range(fault)) => fault.clamped_value(),
            Err(PcuClampedError::Fatal(_)) => {
                unreachable!("fatal-free preflight precedes publication")
            }
        };
        // SAFETY: The unique output span has extent complete native T slots, invocation is
        // in bounds, and write_unaligned requires no T alignment. No pointer escapes the call.
        unsafe {
            output.add(invocation).write_unaligned(result);
        }
    }
    recovered.map_or(Ok(()), |fault| Err(PcuCpuPreparedBinaryError::Fault(fault)))
}
fn evaluate_clamped<T: PcuClampedFloat, const OP: u8>(
    lhs: T,
    rhs: T,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<T, PcuClampedError<T>> {
    match OP {
        0 => lhs.pcu_clamped_add_with_policy(rhs, underflow),
        1 => lhs.pcu_clamped_sub_with_policy(rhs, underflow),
        2 => lhs.pcu_clamped_mul_with_policy(rhs, underflow),
        3 => lhs.pcu_clamped_div_with_policy(rhs, underflow),
        _ => unreachable!("cold constructor selects four binary operations"),
    }
}
