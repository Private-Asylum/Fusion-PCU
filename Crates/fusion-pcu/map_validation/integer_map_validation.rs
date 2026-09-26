//! Shared structural admission for exact-width signed and unsigned integer map profiles.

use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuValueType,
    PcuValueTypeCaps,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerMapValidationError {
    UnsupportedInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    MissingReturn,
}

/// Shared structural admission for checked exact-width quotient/remainder maps.
#[allow(clippy::too_many_lines)]
pub fn validate_integer_checked_div_rem_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), IntegerMapValidationError> {
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 4 {
        return Err(IntegerMapValidationError::UnsupportedInterface);
    }
    let allowed_types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(IntegerMapValidationError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(IntegerMapValidationError::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(IntegerMapValidationError::InvalidBinding(reference));
        }
    }
    let refs = [
        kernel.bindings[0].reference(),
        kernel.bindings[1].reference(),
        kernel.bindings[2].reference(),
        kernel.bindings[3].reference(),
    ];
    for input in &kernel.bindings[..2] {
        if input.access != PcuBindingAccess::ReadOnly {
            return Err(IntegerMapValidationError::InvalidBinding(input.reference()));
        }
    }
    for output in &kernel.bindings[2..] {
        if output.access == PcuBindingAccess::ReadOnly {
            return Err(IntegerMapValidationError::InvalidBinding(
                output.reference(),
            ));
        }
    }
    let (body, index, direct) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(IntegerMapValidationError::UnsupportedOperation(0));
            }
            (*body, PcuDispatchIndex::GridStrideId, false)
        }
        ops if matches!(
            ops.last(),
            Some(PcuDispatchOp::Control(PcuDispatchControlOp::Return))
        ) =>
        {
            (&ops[..ops.len() - 1], PcuDispatchIndex::InvocationId, true)
        }
        _ => return Err(IntegerMapValidationError::MissingReturn),
    };
    if body.len() != 5 {
        return Err(IntegerMapValidationError::UnsupportedOperation(body.len()));
    }
    let mut loaded = [false; 2];
    let mut values = [false; 256];
    for (slot, operation) in body[..2].iter().copied().enumerate() {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index: got,
        }) = operation
        else {
            return Err(IntegerMapValidationError::UnsupportedOperation(slot));
        };
        if got != index {
            return Err(IntegerMapValidationError::InvalidIndex(slot));
        }
        let input_slot = if binding == refs[0] {
            0
        } else if binding == refs[1] {
            1
        } else {
            return Err(IntegerMapValidationError::InvalidBinding(binding));
        };
        if loaded[input_slot] {
            return Err(IntegerMapValidationError::UnsupportedOperation(slot));
        }
        define_integer_div_value(&mut values, result)?;
        loaded[input_slot] = true;
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
        value_type: actual_type,
        flags,
        quotient,
        remainder,
        lhs,
        rhs,
    }) = body[2]
    else {
        return Err(IntegerMapValidationError::UnsupportedOperation(2));
    };
    if actual_type != value_type || flags.bits() != 0 {
        return Err(IntegerMapValidationError::UnsupportedOperation(2));
    }
    require_integer_div_value(&values, lhs)?;
    require_integer_div_value(&values, rhs)?;
    if quotient == remainder {
        return Err(IntegerMapValidationError::InvalidValue(remainder));
    }
    define_integer_div_value(&mut values, quotient)?;
    define_integer_div_value(&mut values, remainder)?;
    for (slot, (expected_binding, expected_value)) in [(refs[2], quotient), (refs[3], remainder)]
        .into_iter()
        .enumerate()
    {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding,
            index: got,
            value,
        }) = body[slot + 3]
        else {
            return Err(IntegerMapValidationError::UnsupportedOperation(slot + 3));
        };
        if got != index {
            return Err(IntegerMapValidationError::InvalidIndex(slot + 3));
        }
        if binding != expected_binding {
            return Err(IntegerMapValidationError::InvalidBinding(binding));
        }
        if value != expected_value {
            return Err(IntegerMapValidationError::InvalidValue(value));
        }
        require_integer_div_value(&values, value)?;
    }
    if !loaded.into_iter().all(core::convert::identity) {
        return Err(IntegerMapValidationError::UnsupportedOperation(2));
    }
    if direct && kernel.ops.len() != body.len() + 1 {
        return Err(IntegerMapValidationError::UnsupportedOperation(body.len()));
    }
    Ok(())
}

fn define_integer_div_value(
    values: &mut [bool; 256],
    id: PcuDispatchValueId,
) -> Result<(), IntegerMapValidationError> {
    let Some(defined) = values.get_mut(usize::from(id.0)) else {
        return Err(IntegerMapValidationError::InvalidValue(id));
    };
    if id.0 == 0 || *defined {
        return Err(IntegerMapValidationError::InvalidValue(id));
    }
    *defined = true;
    Ok(())
}

fn require_integer_div_value(
    values: &[bool; 256],
    id: PcuDispatchValueId,
) -> Result<(), IntegerMapValidationError> {
    if values.get(usize::from(id.0)).copied().unwrap_or(false) {
        Ok(())
    } else {
        Err(IntegerMapValidationError::InvalidValue(id))
    }
}

/// Validates the exact scalar type supplied by the public typed wrapper.
#[allow(clippy::too_many_lines)]
pub fn validate_integer_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), IntegerMapValidationError> {
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 3 {
        return Err(IntegerMapValidationError::UnsupportedInterface);
    }
    let types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !types.bits() != 0
        || kernel.required_feature_support().bits() & !features.bits() != 0
    {
        return Err(IntegerMapValidationError::UnsupportedRequirements);
    }
    for (i, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..i]
            .iter()
            .any(|b| b.reference() == reference)
        {
            return Err(IntegerMapValidationError::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(IntegerMapValidationError::InvalidBinding(reference));
        }
    }
    let input0 = kernel.bindings[0].reference();
    let input1 = kernel.bindings[1].reference();
    let output = kernel.bindings[2].reference();
    if kernel.bindings[0].access != PcuBindingAccess::ReadOnly
        || kernel.bindings[1].access != PcuBindingAccess::ReadOnly
        || kernel.bindings[2].access == PcuBindingAccess::ReadOnly
    {
        return Err(IntegerMapValidationError::InvalidBinding(output));
    }

    let (ops, index, direct) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(IntegerMapValidationError::UnsupportedOperation(0));
            }
            (*body, PcuDispatchIndex::GridStrideId, false)
        }
        ops if matches!(
            ops.last(),
            Some(PcuDispatchOp::Control(PcuDispatchControlOp::Return))
        ) =>
        {
            (&ops[..ops.len() - 1], PcuDispatchIndex::InvocationId, true)
        }
        _ => return Err(IntegerMapValidationError::MissingReturn),
    };
    if ops.len() < 4 {
        return Err(IntegerMapValidationError::UnsupportedOperation(ops.len()));
    }
    let mut defined = [false; 256];
    let mut loaded_inputs = [false; 2];
    let mut alu_result = None;
    for (position, op) in ops.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: got,
            }) if position + 1 < ops.len() => {
                if got != index {
                    return Err(IntegerMapValidationError::InvalidIndex(position));
                }
                let input_slot = if binding == input0 {
                    0
                } else if binding == input1 {
                    1
                } else {
                    return Err(IntegerMapValidationError::InvalidBinding(binding));
                };
                define(&mut defined, result)?;
                loaded_inputs[input_slot] = true;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: got_type,
                result,
                op,
                lhs,
                rhs,
            }) if position + 1 < ops.len() => {
                if got_type != value_type
                    || !matches!(
                        op,
                        PcuDispatchAluOp::Add | PcuDispatchAluOp::Sub | PcuDispatchAluOp::Mul
                    )
                {
                    return Err(IntegerMapValidationError::UnsupportedOperation(position));
                }
                require(&defined, lhs)?;
                require(&defined, rhs)?;
                define(&mut defined, result)?;
                alu_result = Some(result);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: got,
                value,
            }) if position + 1 == ops.len() => {
                if got != index {
                    return Err(IntegerMapValidationError::InvalidIndex(position));
                }
                if binding != output {
                    return Err(IntegerMapValidationError::InvalidBinding(binding));
                }
                if Some(value) != alu_result {
                    return Err(IntegerMapValidationError::InvalidValue(value));
                }
                require(&defined, value)?;
            }
            _ => return Err(IntegerMapValidationError::UnsupportedOperation(position)),
        }
    }
    if !loaded_inputs.into_iter().all(core::convert::identity) || alu_result.is_none() {
        return Err(IntegerMapValidationError::UnsupportedOperation(ops.len()));
    }
    if direct && kernel.ops.len() != ops.len() + 1 {
        return Err(IntegerMapValidationError::UnsupportedOperation(ops.len()));
    }
    Ok(())
}

fn define(
    defined: &mut [bool; 256],
    id: PcuDispatchValueId,
) -> Result<(), IntegerMapValidationError> {
    let Some(slot) = defined.get_mut(usize::from(id.0)) else {
        return Err(IntegerMapValidationError::InvalidValue(id));
    };
    if id.0 == 0 || *slot {
        return Err(IntegerMapValidationError::InvalidValue(id));
    }
    *slot = true;
    Ok(())
}

fn require(defined: &[bool; 256], id: PcuDispatchValueId) -> Result<(), IntegerMapValidationError> {
    if defined.get(usize::from(id.0)) == Some(&true) {
        Ok(())
    } else {
        Err(IntegerMapValidationError::InvalidValue(id))
    }
}
