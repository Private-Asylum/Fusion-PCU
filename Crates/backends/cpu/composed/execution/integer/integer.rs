//! Monomorphized scalar integer SSA execution with private all-output publication.
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
    argument_indices,
    bytes,
    publish,
    seed,
    Executable,
    Error,
    Plan,
    REGISTERS,
    Step,
};

pub const fn select(scalar: PcuScalarType) -> Option<Executable> {
    match scalar {
        PcuScalarType::I8 => Some(execute::<i8>),
        PcuScalarType::U8 => Some(execute::<u8>),
        PcuScalarType::I16 => Some(execute::<i16>),
        PcuScalarType::U16 => Some(execute::<u16>),
        PcuScalarType::I32 => Some(execute::<i32>),
        PcuScalarType::U32 => Some(execute::<u32>),
        PcuScalarType::I64 => Some(execute::<i64>),
        PcuScalarType::U64 => Some(execute::<u64>),
        PcuScalarType::I128 => Some(execute::<i128>),
        PcuScalarType::U128 => Some(execute::<u128>),
        PcuScalarType::I256 => Some(execute::<fusion_pcu::PcuI256>),
        PcuScalarType::U256 => Some(execute::<fusion_pcu::PcuU256>),
        PcuScalarType::I512 => Some(execute::<fusion_pcu::PcuI512>),
        PcuScalarType::U512 => Some(execute::<fusion_pcu::PcuU512>),
        _ => None,
    }
}

fn execute<T: PcuCheckedInteger>(
    plan: &mut Plan,
    arguments: &mut [PcuHostArgument<'_>],
) -> Result<(), Error> {
    let indices = argument_indices(plan, arguments)?;
    seed(plan, arguments, &indices);
    let mut registers = [None; REGISTERS];
    let mut recovered = None;
    for lane in 0..plan.extent {
        for ordinal in 0..plan.steps.len() {
            if let Some((result, value)) = step::<T>(
                plan.steps[ordinal],
                plan,
                arguments,
                &indices,
                &registers,
                lane,
                &mut recovered,
            )? {
                registers[result] = Some(value);
            }
        }
    }
    publish(plan, arguments, &indices)?;
    recovered.map_or(Ok(()), |fault| Err(Error::Fault(fault)))
}

fn step<T: PcuCheckedInteger>(
    step: Step,
    plan: &mut Plan,
    arguments: &[PcuHostArgument<'_>],
    indices: &[usize; super::BINDINGS],
    registers: &[Option<T>; REGISTERS],
    lane: usize,
    recovered: &mut Option<PcuExecutionFault>,
) -> Result<Option<(usize, T)>, Error> {
    match step {
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
            Ok(Some((result, value)))
        }
        Step::IntegerConstant {
            result,
            bytes: constant,
        } => Ok(Some((result, bytes::read::<T>(&constant, 0)?))),
        Step::IntegerBinary {
            result,
            left,
            right,
            operation,
            range,
        } => {
            let left = registers[left].ok_or(Error::InvalidProgram)?;
            let right = registers[right].ok_or(Error::InvalidProgram)?;
            let evaluated = match operation {
                PcuDispatchIntegerBinaryOp::Add => left.pcu_clamped_add(right),
                PcuDispatchIntegerBinaryOp::Sub => left.pcu_clamped_sub(right),
                PcuDispatchIntegerBinaryOp::Mul => left.pcu_clamped_mul(right),
            };
            Ok(Some((result, value(evaluated, range, lane, recovered)?)))
        }
        Step::Store { resource, value } => {
            let resource = plan.resources[resource];
            bytes::write(
                &mut plan.scratch,
                resource.shadow.ok_or(Error::InvalidProgram)? + lane * plan.size,
                registers[value].ok_or(Error::InvalidProgram)?,
            )?;
            Ok(None)
        }
        Step::Constant { .. } | Step::Binary { .. } | Step::Unary { .. } => {
            Err(Error::InvalidProgram)
        }
    }
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
