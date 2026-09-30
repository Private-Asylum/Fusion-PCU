//! Structural and typed admission for checked integer binary maps.

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use super::{
    IntegerMapValidationError,
    define_integer_div_value,
    require_integer_div_value,
};

/// Validates the bounded checked integer binary map profile.
///
/// Inputs may use the common invocation index or `BindingElementZero` for broadcast; output
/// always uses the common invocation index. The body is two loads, one checked binary op, and
/// one store, followed by Return (or wrapped in one grid-stride region followed by Return).
///
/// # Errors
///
/// Returns the first unsupported interface, capability requirement, binding, operation,
/// index, SSA value, or missing return.
pub fn validate_integer_checked_binary_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    op: crate::model::PcuDispatchIntegerBinaryOp,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), IntegerMapValidationError> {
    let refs = validate_checked_binary_interface(kernel, value_type, scalar_caps)?;
    validate_checked_binary_body(kernel, value_type, op, refs)
}

fn validate_checked_binary_interface(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<[PcuBindingRef; 3], IntegerMapValidationError> {
    if !matches!(
        value_type,
        PcuValueType::Scalar(
            crate::PcuScalarType::I8
                | crate::PcuScalarType::U8
                | crate::PcuScalarType::I16
                | crate::PcuScalarType::U16
                | crate::PcuScalarType::I32
                | crate::PcuScalarType::U32
                | crate::PcuScalarType::I64
                | crate::PcuScalarType::U64
        )
    ) {
        return Err(IntegerMapValidationError::UnsupportedInterface);
    }
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 3 {
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
    ];
    if kernel.bindings[0].access != PcuBindingAccess::ReadOnly
        || kernel.bindings[1].access != PcuBindingAccess::ReadOnly
        || kernel.bindings[2].access == PcuBindingAccess::ReadOnly
    {
        let bad = kernel
            .bindings
            .iter()
            .find(|binding| {
                (binding.reference() == refs[0] || binding.reference() == refs[1])
                    != (binding.access == PcuBindingAccess::ReadOnly)
            })
            .map_or(refs[2], |binding| binding.reference());
        return Err(IntegerMapValidationError::InvalidBinding(bad));
    }
    Ok(refs)
}

fn validate_checked_binary_body(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    op: crate::model::PcuDispatchIntegerBinaryOp,
    refs: [PcuBindingRef; 3],
) -> Result<(), IntegerMapValidationError> {
    let (body, index) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(IntegerMapValidationError::UnsupportedOperation(0));
            }
            (*body, PcuDispatchIndex::GridStrideId)
        }
        ops if matches!(
            ops.last(),
            Some(PcuDispatchOp::Control(PcuDispatchControlOp::Return))
        ) =>
        {
            (&ops[..ops.len() - 1], PcuDispatchIndex::InvocationId)
        }
        _ => return Err(IntegerMapValidationError::MissingReturn),
    };
    if body.len() != 4 {
        return Err(IntegerMapValidationError::UnsupportedOperation(body.len()));
    }
    let mut values = [false; 256];
    let mut loaded = [false; 2];
    for (slot, instruction) in body[..2].iter().copied().enumerate() {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index: got,
        }) = instruction
        else {
            return Err(IntegerMapValidationError::UnsupportedOperation(slot));
        };
        if got != index && got != PcuDispatchIndex::BindingElementZero {
            return Err(IntegerMapValidationError::InvalidIndex(slot));
        }
        let input = if binding == refs[0] {
            0
        } else if binding == refs[1] {
            1
        } else {
            return Err(IntegerMapValidationError::InvalidBinding(binding));
        };
        if loaded[input] {
            return Err(IntegerMapValidationError::UnsupportedOperation(slot));
        }
        define_integer_div_value(&mut values, result)?;
        loaded[input] = true;
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
        value_type: actual,
        op: actual_op,
        result,
        lhs,
        rhs,
    }) = body[2]
    else {
        return Err(IntegerMapValidationError::UnsupportedOperation(2));
    };
    if actual != value_type || actual_op != op {
        return Err(IntegerMapValidationError::UnsupportedOperation(2));
    }
    require_integer_div_value(&values, lhs)?;
    require_integer_div_value(&values, rhs)?;
    define_integer_div_value(&mut values, result)?;
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding,
        index: got,
        value,
    }) = body[3]
    else {
        return Err(IntegerMapValidationError::UnsupportedOperation(3));
    };
    if got != index {
        return Err(IntegerMapValidationError::InvalidIndex(3));
    }
    if binding != refs[2] {
        return Err(IntegerMapValidationError::InvalidBinding(binding));
    }
    if value != result {
        return Err(IntegerMapValidationError::InvalidValue(value));
    }
    require_integer_div_value(&values, value)?;
    if !loaded.into_iter().all(core::convert::identity) {
        return Err(IntegerMapValidationError::UnsupportedOperation(2));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
