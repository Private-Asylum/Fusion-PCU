//! Bounded integer lane blocks with exact scalar fault replay and private publication.
#![allow(unsafe_code)]
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuClampedFault,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFault,
    PcuHostArgument,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    validate_aliases,
    bind_resources,
    initialize,
    bytes,
    publish,
    seed,
    Executable,
    Error,
    Plan,
    STEPS,
    BINDINGS,
    Step,
};

use super::super::program::IntegerInstruction;

pub const fn select(scalar: PcuScalarType) -> Option<Executable> {
    match scalar {
        PcuScalarType::I8 => Some(execute::<i8, 16>),
        PcuScalarType::U8 => Some(execute::<u8, 16>),
        PcuScalarType::I16 => Some(execute::<i16, 16>),
        PcuScalarType::U16 => Some(execute::<u16, 16>),
        PcuScalarType::I32 => Some(execute::<i32, 16>),
        PcuScalarType::U32 => Some(execute::<u32, 16>),
        PcuScalarType::I64 => Some(execute::<i64, 16>),
        PcuScalarType::U64 => Some(execute::<u64, 16>),
        PcuScalarType::I128 => Some(execute::<i128, 8>),
        PcuScalarType::U128 => Some(execute::<u128, 8>),
        PcuScalarType::I256 => Some(execute::<fusion_pcu::PcuI256, 4>),
        PcuScalarType::U256 => Some(execute::<fusion_pcu::PcuU256, 4>),
        PcuScalarType::I512 => Some(execute::<fusion_pcu::PcuI512, 2>),
        PcuScalarType::U512 => Some(execute::<fusion_pcu::PcuU512, 2>),
        _ => None,
    }
}

fn execute<T: PcuCheckedInteger, const LANES: usize>(
    plan: &mut Plan,
    arguments: &mut [PcuHostArgument<'_>],
) -> Result<(), Error> {
    let indices = plan.validate_arguments(arguments)?;
    plan.program.validate_carrier::<T>(plan.scratch.len())?;
    validate_aliases(plan, arguments, &indices)?;
    seed(plan, arguments, &indices);
    let resources = bind_resources(plan, arguments, &indices);
    let zero = bytes::read::<T>(&[0; 64], 0)?;
    let mut registers = [zero; STEPS];
    initialize(plan.program.initializers(), &resources, &mut registers)?;
    let mut recovered = None;
    let instructions = plan
        .program
        .integer_instructions()
        .ok_or(Error::InvalidProgram)?;
    blocks::<T, LANES>(
        instructions,
        plan.program.steps(),
        &resources,
        &registers,
        plan.program.extent(),
        &mut recovered,
    )?;
    publish(plan, arguments, &indices)?;
    recovered.map_or(Ok(()), |fault| Err(Error::Fault(fault)))
}

fn scalar_lanes<T: PcuCheckedInteger>(
    steps: &[Step],
    resources: &[bytes::BoundResource; BINDINGS],
    registers: &mut [T; STEPS],
    start: usize,
    end: usize,
    mut recovered: Option<PcuExecutionFault>,
) -> Result<Option<PcuExecutionFault>, Error> {
    for lane in start..end {
        for step in steps {
            match *step {
                Step::Load {
                    result,
                    resource,
                    zero,
                } => {
                    let element = if zero { 0 } else { lane };
                    // SAFETY: cold resource indexing and the current call preflight prove
                    // the complete carrier span and index < actual resource count <= BINDINGS;
                    // bound bases stay live during execution.
                    let value = unsafe { resources.get_unchecked(resource).read::<T>(element) };
                    // SAFETY: this immutable sealed step defines a slot below STEPS.
                    unsafe { *registers.get_unchecked_mut(result) = value };
                }
                Step::IntegerConstant { .. } => return Err(Error::InvalidProgram),
                Step::IntegerBinary {
                    result,
                    left,
                    right,
                    operation,
                    range,
                } => {
                    // SAFETY: sealed program validation proves both dominated slots
                    // are below STEPS; registers are initialized for every call.
                    let (left, right) = unsafe {
                        (
                            *registers.get_unchecked(left),
                            *registers.get_unchecked(right),
                        )
                    };
                    let evaluated = match operation {
                        PcuDispatchIntegerBinaryOp::Add => left.pcu_clamped_add(right),
                        PcuDispatchIntegerBinaryOp::Sub => left.pcu_clamped_sub(right),
                        PcuDispatchIntegerBinaryOp::Mul => left.pcu_clamped_mul(right),
                    };
                    let evaluated = value(evaluated, range, lane, &mut recovered)?;
                    // SAFETY: the sealed result index is below STEPS.
                    unsafe { *registers.get_unchecked_mut(result) = evaluated };
                }
                Step::Store { resource, value } => {
                    // SAFETY: Store has a private writable scratch span covering every
                    // logical lane. Sealed indices are below BINDINGS/STEPS, and
                    // the register definition dominates this use. No public destination
                    // is mutated until full success.
                    unsafe {
                        resources
                            .get_unchecked(resource)
                            .write(lane, *registers.get_unchecked(value));
                    }
                }
                Step::Constant { .. } | Step::Binary { .. } | Step::Unary { .. } => {
                    return Err(Error::InvalidProgram);
                }
            }
        }
    }
    Ok(recovered)
}

pub const fn select_scalar(scalar: PcuScalarType) -> Option<Executable> {
    match scalar {
        PcuScalarType::I8 => Some(execute_scalar::<i8>),
        PcuScalarType::U8 => Some(execute_scalar::<u8>),
        PcuScalarType::I16 => Some(execute_scalar::<i16>),
        PcuScalarType::U16 => Some(execute_scalar::<u16>),
        PcuScalarType::I32 => Some(execute_scalar::<i32>),
        PcuScalarType::U32 => Some(execute_scalar::<u32>),
        PcuScalarType::I64 => Some(execute_scalar::<i64>),
        PcuScalarType::U64 => Some(execute_scalar::<u64>),
        PcuScalarType::I128 => Some(execute_scalar::<i128>),
        PcuScalarType::U128 => Some(execute_scalar::<u128>),
        PcuScalarType::I256 => Some(execute_scalar::<fusion_pcu::PcuI256>),
        PcuScalarType::U256 => Some(execute_scalar::<fusion_pcu::PcuU256>),
        PcuScalarType::I512 => Some(execute_scalar::<fusion_pcu::PcuI512>),
        PcuScalarType::U512 => Some(execute_scalar::<fusion_pcu::PcuU512>),
        _ => None,
    }
}

fn execute_scalar<T: PcuCheckedInteger>(
    plan: &mut Plan,
    arguments: &mut [PcuHostArgument<'_>],
) -> Result<(), Error> {
    let indices = plan.validate_arguments(arguments)?;
    plan.program.validate_carrier::<T>(plan.scratch.len())?;
    validate_aliases(plan, arguments, &indices)?;
    seed(plan, arguments, &indices);
    let resources = bind_resources(plan, arguments, &indices);
    let zero = bytes::read::<T>(&[0; 64], 0)?;
    let mut registers = [zero; STEPS];
    initialize(plan.program.initializers(), &resources, &mut registers)?;
    let recovered = scalar_lanes(
        plan.program.steps(),
        &resources,
        &mut registers,
        0,
        plan.program.extent(),
        None,
    )?;
    publish(plan, arguments, &indices)?;
    recovered.map_or(Ok(()), |fault| Err(Error::Fault(fault)))
}

fn blocks<T: PcuCheckedInteger, const LANES: usize>(
    instructions: &[IntegerInstruction],
    steps: &[Step],
    resources: &[bytes::BoundResource; BINDINGS],
    initial: &[T; STEPS],
    extent: usize,
    recovered: &mut Option<PcuExecutionFault>,
) -> Result<(), Error> {
    // At most 8 KiB for every selected carrier, including 512-bit integers.
    // Hot definitions are overwritten in each block; readonly initializers persist.
    let mut bank = [[initial[0]; LANES]; STEPS];
    for (slot, value) in bank.iter_mut().zip(initial) {
        slot.fill(*value);
    }
    for start in (0..extent).step_by(LANES) {
        let count = (extent - start).min(LANES);
        if !block(instructions, resources, &mut bank, start, count) {
            // No admitted load observes speculative writes. Replay overwrites all
            // store-only output cells, or returns fatal before any publication.
            // Fresh initializers preserve lane-major lexical fault ordering.
            let mut registers = *initial;
            *recovered = scalar_lanes(
                steps,
                resources,
                &mut registers,
                start,
                start + count,
                *recovered,
            )?;
        }
    }
    Ok(())
}

fn block<T: PcuCheckedInteger, const LANES: usize>(
    instructions: &[IntegerInstruction],
    resources: &[bytes::BoundResource; BINDINGS],
    bank: &mut [[T; LANES]; STEPS],
    start: usize,
    count: usize,
) -> bool {
    for instruction in instructions {
        match *instruction {
            IntegerInstruction::Load {
                result,
                resource,
                zero,
            } => {
                for lane in 0..count {
                    // SAFETY: immutable cold validation proves the byte-sized
                    // resource/slot indices and complete readonly carrier spans.
                    unsafe {
                        bank.get_unchecked_mut(usize::from(result))[lane] = resources
                            .get_unchecked(usize::from(resource))
                            .read::<T>(if zero { 0 } else { start + lane });
                    }
                }
            }
            IntegerInstruction::Add {
                result,
                left,
                right,
            } => {
                if !binary_block(bank, result, left, right, count, T::pcu_clamped_add) {
                    return false;
                }
            }
            IntegerInstruction::Sub {
                result,
                left,
                right,
            } => {
                if !binary_block(bank, result, left, right, count, T::pcu_clamped_sub) {
                    return false;
                }
            }
            IntegerInstruction::Mul {
                result,
                left,
                right,
            } => {
                if !binary_block(bank, result, left, right, count, T::pcu_clamped_mul) {
                    return false;
                }
            }
            IntegerInstruction::Store { resource, value } => {
                for lane in 0..count {
                    // SAFETY: sealed store-only spans cover this block and are
                    // private until full-call success; the value slot is dominated.
                    unsafe {
                        resources
                            .get_unchecked(usize::from(resource))
                            .write(start + lane, bank.get_unchecked(usize::from(value))[lane]);
                    }
                }
            }
        }
    }
    true
}

#[inline]
fn binary_block<T: PcuCheckedInteger, F, const LANES: usize>(
    bank: &mut [[T; LANES]; STEPS],
    result: u8,
    left: u8,
    right: u8,
    count: usize,
    operation: F,
) -> bool
where
    F: Fn(T, T) -> Result<T, PcuClampedFault<T>>,
{
    for lane in 0..count {
        // SAFETY: indices were packed only after SSA dominance and bounds proof.
        // Each operation is monomorphized independently outside the lane loop.
        unsafe {
            let evaluated = operation(
                bank.get_unchecked(usize::from(left))[lane],
                bank.get_unchecked(usize::from(right))[lane],
            );
            let Ok(value) = evaluated else {
                return false;
            };
            bank.get_unchecked_mut(usize::from(result))[lane] = value;
        }
    }
    true
}

fn value<T: PcuCheckedInteger>(
    result: Result<T, PcuClampedFault<T>>,
    range: PcuRangePolicy,
    lane: usize,
    recovered: &mut Option<PcuExecutionFault>,
) -> Result<T, Error> {
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            let fault = PcuExecutionFault {
                invocation_id: lane as u64,
                kind: error.kind(),
                recovered: range == PcuRangePolicy::Clamp,
            };
            if fault.recovered {
                recovered.get_or_insert(fault);
                Ok(error.clamped_value())
            } else {
                Err(Error::Fault(fault))
            }
        }
    }
}
