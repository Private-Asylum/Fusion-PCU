//! Execute detached scalar steps against private mutable state; commit after complete success.
#![allow(unsafe_code)]
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
    STEPS,
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
pub(super) fn validate_aliases(
    plan: &Plan,
    arguments: &[PcuHostArgument<'_>],
    indices: &[usize; BINDINGS],
) -> Result<(), Error> {
    for (index, resource) in plan.program.resources().iter().enumerate() {
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
        for other in &plan.program.resources()[..index] {
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
    Ok(())
}
pub(super) fn seed(
    plan: &mut Plan,
    arguments: &[PcuHostArgument<'_>],
    indices: &[usize; BINDINGS],
) {
    for resource in plan.program.resources() {
        if let Some(start) = resource.shadow {
            // Only originally loaded bytes need initialization. Every written cell is
            // overwritten by the straight-line store on every fatal-free logical lane.
            plan.scratch[start..start + resource.read_bytes].copy_from_slice(
                &arguments[indices[resource.declaration]].bytes()[..resource.read_bytes],
            );
        }
    }
}
// A new set of bases is established on every call. Arguments were type/length
// checked before entry; validate_aliases additionally checked physical overlap.
// The resource assessor proved every load/store's index and minimum extent.
// Scratch does not resize, and arguments are not published/mutated until execution ends.
fn bind_resources(
    plan: &mut Plan,
    arguments: &[PcuHostArgument<'_>],
    indices: &[usize; BINDINGS],
) -> [bytes::BoundResource; BINDINGS] {
    let mut bound = [bytes::BoundResource::EMPTY; BINDINGS];
    for (index, resource) in plan.program.resources().iter().enumerate() {
        bound[index] = if let Some(start) = resource.shadow {
            let pointer = plan.scratch.as_mut_ptr().wrapping_add(start);
            bytes::BoundResource {
                read: pointer,
                write: pointer,
            }
        } else {
            bytes::BoundResource {
                read: arguments[indices[resource.declaration]].bytes().as_ptr(),
                write: core::ptr::null_mut(),
            }
        };
    }
    bound
}
fn initialize<T: fusion_pcu::PcuScalar>(
    initializers: &[Step],
    resources: &[bytes::BoundResource; BINDINGS],
    registers: &mut [T; STEPS],
) -> Result<(), Error> {
    for initializer in initializers {
        let (result, value) = match *initializer {
            Step::Constant {
                result,
                bytes: constant,
            } => (result, bytes::read::<T>(&constant, 0)?),
            Step::IntegerConstant {
                result,
                bytes: constant,
            } => (result, bytes::read::<T>(&constant, 0)?),
            Step::Load {
                result,
                resource,
                zero: true,
            } => {
                // SAFETY: the cold partition admits only readonly element-zero loads.
                // The current call validated a complete carrier and built fresh bases;
                // physical alias preflight excludes every potentially mutable resource.
                (result, unsafe { resources[resource].read::<T>(0) })
            }
            _ => return Err(Error::InvalidProgram),
        };
        registers[result] = value;
    }
    Ok(())
}
fn execute<T: PcuClampedFloat>(
    plan: &mut Plan,
    arguments: &mut [PcuHostArgument<'_>],
) -> Result<(), Error> {
    let indices = plan.validate_arguments(arguments)?;
    plan.program.validate_carrier::<T>(plan.scratch.len())?;
    validate_aliases(plan, arguments, &indices)?;
    seed(plan, arguments, &indices);
    let resources = bind_resources(plan, arguments, &indices);
    // Every operand is dominated by its definition, verified during preparation.
    // Initialized storage keeps the interpreter safe without per-operand Option checks.
    let zero = bytes::read::<T>(&[0; 64], 0)?;
    let mut registers = [zero; STEPS];
    initialize(plan.program.initializers(), &resources, &mut registers)?;
    let mut recovered = None;
    for lane in 0..plan.program.extent() {
        for step in plan.program.steps() {
            let assigned = match *step {
                Step::Load {
                    result,
                    resource,
                    zero,
                } => {
                    let element = if zero { 0 } else { lane };
                    // SAFETY: cold indexing and call preflight prove this span covers
                    // the element and the sealed resource index is below BINDINGS.
                    // Bound bases remain live throughout this loop.
                    let value = unsafe { resources.get_unchecked(resource).read::<T>(element) };
                    Some((result, Ok(value), PcuRangePolicy::Reject))
                }
                Step::Constant { .. } => return Err(Error::InvalidProgram),
                Step::Binary {
                    result,
                    left,
                    right,
                    operation,
                    range,
                    underflow,
                } => Some((
                    result,
                    // SAFETY: the sealed program proves both dominated slots < STEPS.
                    unsafe {
                        binary(
                            operation,
                            *registers.get_unchecked(left),
                            *registers.get_unchecked(right),
                            underflow,
                        )
                    },
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
                    // SAFETY: the sealed program proves this dominated slot < STEPS.
                    unsafe { unary(operation, *registers.get_unchecked(value), underflow) },
                    range,
                )),
                Step::Store { resource, value } => {
                    // SAFETY: a Store only targets an assessor-proven writable
                    // resource with a private scratch span covering every logical lane.
                    // Sealed indices are below BINDINGS/STEPS; the value is dominated.
                    unsafe {
                        resources
                            .get_unchecked(resource)
                            .write(lane, *registers.get_unchecked(value));
                    }
                    None
                }
                Step::IntegerBinary { .. } | Step::IntegerConstant { .. } => {
                    return Err(Error::InvalidProgram);
                }
            };
            if let Some((result, value, range)) = assigned {
                let value = checked_value(value, range, lane, &mut recovered)?;
                // SAFETY: the sealed program proves this unique result slot < STEPS.
                unsafe { *registers.get_unchecked_mut(result) = value };
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
    for resource in plan.program.resources() {
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
