use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchSubmission,
    PcuDispatchValueId,
    PcuHostScalarBinding,
    PcuHostScalarSlice,
    PcuParameterValue,
};

use crate::{
    PcuF32ReferenceError,
    VALUE_SLOTS,
};

pub fn validate_program(
    submission: PcuDispatchSubmission<'_>,
    bindings: &[PcuHostScalarBinding<'_, f32>],
) -> Result<(), PcuF32ReferenceError> {
    if let Some((position, extent, body)) =
        submission
            .kernel
            .ops
            .iter()
            .enumerate()
            .find_map(|(position, op)| match op {
                PcuDispatchOp::GridStrideLoop { extent, body } => Some((position, extent, body)),
                _ => None,
            })
    {
        if *extent == 0 || submission.kernel.ops.len() != 2 || position != 0 {
            return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
        }
        if !matches!(
            submission.kernel.ops[1],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return)
        ) {
            return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
        }
        validate_grid_stride_body(submission.kernel, body, bindings)?;
        return Ok(());
    }
    let mut defined = [false; VALUE_SLOTS];
    let mut saw_store = false;
    let mut saw_return = false;
    for (position, op) in submission.kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_load_index(index, position)?;
                check_binding(submission.kernel, bindings, binding, false)?;
                define(&mut defined, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                define(&mut defined, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                        | PcuDispatchAluOp::Min
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                require(&defined, lhs)?;
                require(&defined, rhs)?;
                define(&mut defined, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                check_store_index(index, position)?;
                check_binding(submission.kernel, bindings, binding, true)?;
                require(&defined, value)?;
                saw_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(position)),
        }
    }
    if !saw_store {
        return Err(PcuF32ReferenceError::MissingStore);
    }
    if !saw_return {
        return Err(PcuF32ReferenceError::MissingReturn);
    }
    Ok(())
}

pub fn f32_min(left: f32, right: f32) -> f32 {
    if left == 0.0 && right == 0.0 {
        return f32::from_bits(1 << 31);
    }
    left.min(right)
}

pub fn f32_max(left: f32, right: f32) -> f32 {
    if left == 0.0 && right == 0.0 {
        return 0.0;
    }
    left.max(right)
}

fn validate_grid_stride_body(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
    bindings: &[PcuHostScalarBinding<'_, f32>],
) -> Result<(), PcuF32ReferenceError> {
    let mut saw_store = false;
    for (position, op) in body.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                check_grid_load_index(index, position)?;
                check_binding(kernel, bindings, binding, false)?;
                grid_define(body, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                grid_define(body, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                        | PcuDispatchAluOp::Min
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(PcuF32ReferenceError::UnsupportedInstruction(position));
                }
                grid_require(body, position, lhs)?;
                grid_require(body, position, rhs)?;
                grid_define(body, position, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                check_grid_store_index(index, position)?;
                check_binding(kernel, bindings, binding, true)?;
                grid_require(body, position, value)?;
                saw_store = true;
            }
            _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(position)),
        }
    }
    if saw_store {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::MissingStore)
    }
}

pub fn execute_grid_stride_body(
    body: &[PcuDispatchOp<'_>],
    logical: usize,
    bindings: &mut [PcuHostScalarBinding<'_, f32>],
) -> Result<(), PcuF32ReferenceError> {
    let mut values = [None; VALUE_SLOTS];
    for op in body {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                let source = bindings
                    .iter()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                let element = if *index == PcuDispatchIndex::BindingElementZero {
                    0
                } else {
                    logical
                };
                let value = match &source.slice {
                    PcuHostScalarSlice::Read(slice) => slice[element],
                    PcuHostScalarSlice::ReadWrite(slice) => slice[element],
                };
                values[usize::from(result.0)] = Some(value);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result,
                value: PcuParameterValue::F32(bits),
            }) => {
                values[usize::from(result.0)] = Some(f32::from_bits(*bits));
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                let left =
                    values[usize::from(lhs.0)].ok_or(PcuF32ReferenceError::InvalidValue(*lhs))?;
                let right =
                    values[usize::from(rhs.0)].ok_or(PcuF32ReferenceError::InvalidValue(*rhs))?;
                values[usize::from(result.0)] = Some(match op {
                    PcuDispatchAluOp::Add => left + right,
                    PcuDispatchAluOp::Sub => left - right,
                    PcuDispatchAluOp::Mul => left * right,
                    PcuDispatchAluOp::Div => left / right,
                    PcuDispatchAluOp::Min => f32_min(left, right),
                    PcuDispatchAluOp::Max => f32_max(left, right),
                    _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(0)),
                });
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, value, .. }) => {
                let destination = bindings
                    .iter_mut()
                    .find(|candidate| candidate.target == *binding)
                    .ok_or(PcuF32ReferenceError::MissingBinding(*binding))?;
                let PcuHostScalarSlice::ReadWrite(slice) = &mut destination.slice else {
                    return Err(PcuF32ReferenceError::AccessMismatch(*binding));
                };
                slice[logical] = values[usize::from(value.0)]
                    .ok_or(PcuF32ReferenceError::InvalidValue(*value))?;
            }
            _ => return Err(PcuF32ReferenceError::UnsupportedInstruction(0)),
        }
    }
    Ok(())
}

const fn check_grid_load_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32ReferenceError> {
    if matches!(
        index,
        PcuDispatchIndex::GridStrideId | PcuDispatchIndex::BindingElementZero
    ) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::UnsupportedInstruction(position))
    }
}

const fn check_grid_store_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32ReferenceError> {
    if matches!(index, PcuDispatchIndex::GridStrideId) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::UnsupportedInstruction(position))
    }
}

fn grid_define(
    body: &[PcuDispatchOp<'_>],
    position: usize,
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    if usize::from(id.0) >= VALUE_SLOTS
        || id.0 == 0
        || body[..position].iter().any(|op| match op {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { result, .. }
                | PcuDispatchDataOp::Constant { result, .. }
                | PcuDispatchDataOp::Alu { result, .. },
            ) => *result == id,
            _ => false,
        })
    {
        Err(PcuF32ReferenceError::InvalidValue(id))
    } else {
        Ok(())
    }
}

fn grid_require(
    body: &[PcuDispatchOp<'_>],
    position: usize,
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    if usize::from(id.0) < VALUE_SLOTS
        && body[..position].iter().any(|op| match op {
            PcuDispatchOp::Data(
                PcuDispatchDataOp::BindingLoad { result, .. }
                | PcuDispatchDataOp::Constant { result, .. }
                | PcuDispatchDataOp::Alu { result, .. },
            ) => *result == id,
            _ => false,
        })
    {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::InvalidValue(id))
    }
}

const fn check_load_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32ReferenceError> {
    if matches!(
        index,
        PcuDispatchIndex::InvocationId | PcuDispatchIndex::BindingElementZero
    ) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::UnsupportedInstruction(position))
    }
}

const fn check_store_index(
    index: PcuDispatchIndex,
    position: usize,
) -> Result<(), PcuF32ReferenceError> {
    if matches!(index, PcuDispatchIndex::InvocationId) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::UnsupportedInstruction(position))
    }
}

fn check_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    bindings: &[PcuHostScalarBinding<'_, f32>],
    target: PcuBindingRef,
    write: bool,
) -> Result<(), PcuF32ReferenceError> {
    let Some(declared) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuF32ReferenceError::MissingBinding(target));
    };
    let Some(binding) = bindings.iter().find(|binding| binding.target == target) else {
        return Err(PcuF32ReferenceError::MissingBinding(target));
    };
    if (write
        && (declared.access == PcuBindingAccess::ReadOnly
            || binding.slice.access() != PcuBindingAccess::ReadWrite))
        || (!write && declared.access == PcuBindingAccess::WriteOnly)
    {
        return Err(PcuF32ReferenceError::AccessMismatch(target));
    }
    Ok(())
}

fn define(
    defined: &mut [bool; VALUE_SLOTS],
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    let Some(slot) = defined.get_mut(usize::from(id.0)) else {
        return Err(PcuF32ReferenceError::InvalidValue(id));
    };
    if id.0 == 0 || *slot {
        return Err(PcuF32ReferenceError::InvalidValue(id));
    }
    *slot = true;
    Ok(())
}

fn require(
    defined: &[bool; VALUE_SLOTS],
    id: PcuDispatchValueId,
) -> Result<(), PcuF32ReferenceError> {
    if defined.get(usize::from(id.0)) == Some(&true) {
        Ok(())
    } else {
        Err(PcuF32ReferenceError::InvalidValue(id))
    }
}
