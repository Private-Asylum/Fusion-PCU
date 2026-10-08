//! Compact instructions admitted only after the enclosing program's complete proof.
use alloc::vec::Vec;
use fusion_pcu::PcuDispatchIntegerBinaryOp;
#[rustfmt::skip]
use super::{Error, Resource, Step};

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    Load {
        result: u8,
        resource: u8,
        zero: bool,
    },
    Add {
        result: u8,
        left: u8,
        right: u8,
    },
    Sub {
        result: u8,
        left: u8,
        right: u8,
    },
    Mul {
        result: u8,
        left: u8,
        right: u8,
    },
    Store {
        resource: u8,
        value: u8,
    },
}

pub(super) fn lower(
    extent: usize,
    resources: &[Resource],
    steps: &[Step],
) -> Result<Option<Vec<Instruction>>, Error> {
    // Small calls retain the scalar path to avoid the bounded bank setup cost.
    if extent < 64 {
        return Ok(None);
    }
    // A speculative store may survive in private whole-call scratch after a fault.
    // Reject every program that could read it, including mutable broadcasts. The
    // enclosing validator also proves every output cell is overwritten on replay.
    if steps.iter().any(|step| {
        matches!(*step, Step::Load { resource, .. }
            if resources.get(resource).is_none_or(|resource| resource.write_bytes != 0))
    }) {
        return Ok(None);
    }
    let mut instructions = Vec::new();
    instructions
        .try_reserve_exact(steps.len())
        .map_err(|_| Error::AllocationFailed)?;
    for step in steps {
        instructions.push(instruction(*step).ok_or(Error::InvalidProgram)?);
    }
    Ok(Some(instructions))
}

fn instruction(step: Step) -> Option<Instruction> {
    Some(match step {
        Step::Load {
            result,
            resource,
            zero,
        } => Instruction::Load {
            result: u8::try_from(result).ok()?,
            resource: u8::try_from(resource).ok()?,
            zero,
        },
        Step::IntegerBinary {
            result,
            left,
            right,
            operation,
            ..
        } => {
            let result = u8::try_from(result).ok()?;
            let left = u8::try_from(left).ok()?;
            let right = u8::try_from(right).ok()?;
            match operation {
                PcuDispatchIntegerBinaryOp::Add => Instruction::Add {
                    result,
                    left,
                    right,
                },
                PcuDispatchIntegerBinaryOp::Sub => Instruction::Sub {
                    result,
                    left,
                    right,
                },
                PcuDispatchIntegerBinaryOp::Mul => Instruction::Mul {
                    result,
                    left,
                    right,
                },
            }
        }
        Step::Store { resource, value } => Instruction::Store {
            resource: u8::try_from(resource).ok()?,
            value: u8::try_from(value).ok()?,
        },
        _ => return None,
    })
}
