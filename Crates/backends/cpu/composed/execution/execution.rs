//! Execute detached scalar steps against private mutable state; commit after complete success.
#[path = "bytes/bytes.rs"]
mod bytes;
#[path = "integer/integer.rs"]
pub(super) mod integer;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuClampedError,
    PcuClampedFloat,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuExecutionFault,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    BINDINGS,
    REGISTERS,
    Executable,
    PcuCpuComposedMapError as Error,
    PcuCpuPreparedComposedMap as Plan,
    Step,
};
pub(super) const fn select(scalar: PcuScalarType) -> Option<Executable> {
    match scalar {
        PcuScalarType::F16 => Some(execute::<PcuF16Bits>),
        PcuScalarType::BF16 => Some(execute::<PcuBf16Bits>),
        PcuScalarType::F8E4M3FN => Some(execute::<PcuF8E4M3FnBits>),
        PcuScalarType::F8E5M2 => Some(execute::<PcuF8E5M2Bits>),
        PcuScalarType::F32 => Some(execute::<f32>),
        PcuScalarType::F64 => Some(execute::<f64>),
        _ => None,
    }
}
pub(super) fn argument_indices(
    plan: &Plan,
    arguments: &[PcuHostArgument<'_>],
) -> Result<[usize; BINDINGS], Error> {
    let mut indices = [0; BINDINGS];
    for (declaration, &(binding, _, _)) in plan.schema.iter().enumerate() {
        indices[declaration] = arguments
            .iter()
            .position(|argument| argument.target() == binding)
            .ok_or(Error::InvalidProgram)?;
    }
    for (index, resource) in plan.resources.iter().enumerate() {
        let argument = &arguments[indices[resource.declaration]];
        let length = resource.read_bytes.max(resource.write_bytes);
        if resource.write_bytes != 0 && argument.access() != PcuBindingAccess::ReadWrite {
            return Err(Error::InvalidProgram);
        }
        if length == 0 {
            continue;
        }
        let start = argument.bytes().as_ptr() as usize;
        let end = start.checked_add(length).ok_or(Error::ExtentOverflow)?;
        for other in &plan.resources[..index] {
            if resource.write_bytes == 0 && other.write_bytes == 0 {
                continue;
            }
            let other_length = other.read_bytes.max(other.write_bytes);
            if other_length == 0 {
                continue;
            }
            let other_start = arguments[indices[other.declaration]].bytes().as_ptr() as usize;
            let other_end = other_start
                .checked_add(other_length)
                .ok_or(Error::ExtentOverflow)?;
            if start < other_end && other_start < end {
                return Err(Error::PhysicalAlias);
            }
        }
    }
    Ok(indices)
}
pub(super) fn seed(
    plan: &mut Plan,
    arguments: &[PcuHostArgument<'_>],
    indices: &[usize; BINDINGS],
) {
    for resource in &plan.resources {
        if let Some(start) = resource.shadow {
            // Only originally loaded bytes need initialization. Every written cell is
            // overwritten by the straight-line store on every fatal-free logical lane.
            plan.scratch[start..start + resource.read_bytes].copy_from_slice(
                &arguments[indices[resource.declaration]].bytes()[..resource.read_bytes],
            );
        }
    }
}
fn execute<T: PcuClampedFloat>(
    plan: &mut Plan,
    arguments: &mut [PcuHostArgument<'_>],
) -> Result<(), Error> {
    let indices = argument_indices(plan, arguments)?;
    seed(plan, arguments, &indices);
    let mut registers = [None; REGISTERS];
    let mut recovered = None;
    for lane in 0..plan.extent {
        for step in &plan.steps {
            let assigned = match *step {
                Step::Load {
                    result,
                    resource,
                    zero,
                } => {
                    let resource = plan.resources[resource];
                    let element = if zero { 0 } else { lane };
                    let value = if let Some(start) = resource.shadow {
                        bytes::read::<T>(&plan.scratch, start + element * plan.size)?
                    } else {
                        bytes::read::<T>(
                            arguments[indices[resource.declaration]].bytes(),
                            element * plan.size,
                        )?
                    };
                    Some((result, Ok(value), PcuRangePolicy::Reject))
                }
                Step::Constant {
                    result,
                    bytes: constant,
                } => Some((
                    result,
                    Ok(bytes::read::<T>(&constant, 0)?),
                    PcuRangePolicy::Reject,
                )),
                Step::Binary {
                    result,
                    left,
                    right,
                    operation,
                    range,
                    underflow,
                } => Some((
                    result,
                    binary(
                        operation,
                        registers[left].ok_or(Error::InvalidProgram)?,
                        registers[right].ok_or(Error::InvalidProgram)?,
                        underflow,
                    ),
                    range,
                )),
                Step::Unary {
                    result,
                    value,
                    operation,
                    range,
                    underflow,
                } => Some((
                    result,
                    unary(
                        operation,
                        registers[value].ok_or(Error::InvalidProgram)?,
                        underflow,
                    ),
                    range,
                )),
                Step::Store { resource, value } => {
                    let resource = plan.resources[resource];
                    let start = resource.shadow.ok_or(Error::InvalidProgram)?;
                    bytes::write(
                        &mut plan.scratch,
                        start + lane * plan.size,
                        registers[value].ok_or(Error::InvalidProgram)?,
                    )?;
                    None
                }
                Step::IntegerBinary { .. } | Step::IntegerConstant { .. } => {
                    return Err(Error::InvalidProgram);
                }
            };
            if let Some((result, value, range)) = assigned {
                registers[result] = Some(checked_value(value, range, lane, &mut recovered)?);
            }
        }
    }
    publish(plan, arguments, &indices)?;
    recovered.map_or(Ok(()), |fault| Err(Error::Fault(fault)))
}
fn publish(
    plan: &Plan,
    arguments: &mut [PcuHostArgument<'_>],
    indices: &[usize; BINDINGS],
) -> Result<(), Error> {
    // Complete preflight checked all exclusive destinations. Nothing altered the arguments;
    // every destination is still mutable and sized. No public store occurred above.
    for resource in &plan.resources {
        if resource.write_bytes == 0 {
            continue;
        }
        let start = resource.shadow.ok_or(Error::InvalidProgram)?;
        let output = arguments[indices[resource.declaration]]
            .bytes_mut()
            .ok_or(Error::InvalidProgram)?;
        output[..resource.write_bytes]
            .copy_from_slice(&plan.scratch[start..start + resource.write_bytes]);
    }
    Ok(())
}
fn checked_value<T>(
    value: Result<T, PcuClampedError<T>>,
    range: PcuRangePolicy,
    lane: usize,
    recovered: &mut Option<PcuExecutionFault>,
) -> Result<T, Error> {
    match value {
        Ok(value) => Ok(value),
        Err(PcuClampedError::Range(fault)) if range == PcuRangePolicy::Clamp => {
            recovered.get_or_insert(PcuExecutionFault {
                kind: fault.kind(),
                invocation_id: lane as u64,
                recovered: true,
            });
            Ok(fault.clamped_value())
        }
        Err(error) => Err(Error::Fault(PcuExecutionFault {
            kind: error.kind(),
            invocation_id: lane as u64,
            recovered: false,
        })),
    }
}
fn binary<T: PcuClampedFloat>(
    operation: PcuDispatchFloatBinaryOp,
    left: T,
    right: T,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<T, PcuClampedError<T>> {
    match operation {
        PcuDispatchFloatBinaryOp::Add => left.pcu_clamped_add_with_policy(right, underflow),
        PcuDispatchFloatBinaryOp::Sub => left.pcu_clamped_sub_with_policy(right, underflow),
        PcuDispatchFloatBinaryOp::Mul => left.pcu_clamped_mul_with_policy(right, underflow),
        PcuDispatchFloatBinaryOp::Div => left.pcu_clamped_div_with_policy(right, underflow),
    }
}
fn unary<T: PcuClampedFloat>(
    operation: PcuDispatchFloatUnaryOp,
    value: T,
    underflow: PcuFloatUnderflowPolicy,
) -> Result<T, PcuClampedError<T>> {
    match operation {
        PcuDispatchFloatUnaryOp::Neg => value.pcu_clamped_neg_with_policy(underflow),
        PcuDispatchFloatUnaryOp::Relu => value.pcu_clamped_relu_with_policy(underflow),
    }
}
