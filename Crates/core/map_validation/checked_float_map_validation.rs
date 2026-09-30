//! Admission for homogeneous scalar maps whose floating arithmetic is explicitly checked.

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
    PcuDispatchFloatUnaryOp,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural failure while validating a checked floating scalar-map kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedFloatMapValidationError {
    UnsupportedType,
    UnsupportedInterface,
    InvalidLogicalShape,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidSsa,
    MissingCheckedOperation,
    MissingStoreOrReturn,
}

/// Validate a homogeneous F32 or F64 map in which every arithmetic operation is checked.
///
/// The accepted body consists only of scalar bindings, matching-width float constants,
/// loads, stores, `CheckedFloatBinary`, `CheckedFloatUnary`, and a terminal return. It may be a direct invocation
/// body or exactly one grid-stride body followed by Return. A checked operation carries its
/// own underflow policy; policies may differ between operations. Loads may use the canonical
/// invocation index (or grid-stride index inside the loop) or broadcast element zero. Stores
/// must use the canonical index. Unchecked ALU, conversions, mixed types, and other control
/// flow are rejected.
///
/// # Errors
///
/// Returns the first structural, interface, indexing, or SSA error.
#[allow(clippy::too_many_lines)] // Keep the allowlisted execution profile auditable in one pass.
pub fn validate_checked_float_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), CheckedFloatMapValidationError> {
    let cap = if value_type == PcuValueType::f32() {
        PcuValueTypeCaps::FLOAT32
    } else if value_type == PcuValueType::f64() {
        PcuValueTypeCaps::FLOAT64
    } else {
        return Err(CheckedFloatMapValidationError::UnsupportedType);
    };
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.is_empty() {
        return Err(CheckedFloatMapValidationError::UnsupportedInterface);
    }
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(CheckedFloatMapValidationError::InvalidLogicalShape);
    }
    let allowed_types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
        | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        | PcuDispatchFeatureCaps::RANGE_CLAMP;
    if !scalar_caps.contains(cap)
        || kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(CheckedFloatMapValidationError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(CheckedFloatMapValidationError::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(CheckedFloatMapValidationError::InvalidBinding(reference));
        }
    }

    let (body, index) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(CheckedFloatMapValidationError::UnsupportedOperation(0));
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
        _ => return Err(CheckedFloatMapValidationError::MissingStoreOrReturn),
    };
    if body.is_empty() {
        return Err(CheckedFloatMapValidationError::MissingCheckedOperation);
    }

    let mut has_checked = false;
    let mut has_store = false;
    for (position, instruction) in body.iter().copied().enumerate() {
        match instruction {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding,
                index: got,
                ..
            }) => {
                if got != index && got != PcuDispatchIndex::BindingElementZero {
                    return Err(CheckedFloatMapValidationError::InvalidIndex(position));
                }
                let binding_info = find_binding(kernel, binding)?;
                if binding_info.access == PcuBindingAccess::WriteOnly {
                    return Err(CheckedFloatMapValidationError::InvalidBinding(binding));
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { value, .. }) => {
                if !constant_matches(value, value_type) {
                    return Err(CheckedFloatMapValidationError::UnsupportedOperation(
                        position,
                    ));
                }
            }
            PcuDispatchOp::Data(
                PcuDispatchDataOp::CheckedFloatBinary {
                    value_type: actual, ..
                }
                | PcuDispatchDataOp::CheckedFloatUnary {
                    value_type: actual,
                    op: PcuDispatchFloatUnaryOp::Relu | PcuDispatchFloatUnaryOp::Neg,
                    ..
                },
            ) => {
                if actual != value_type {
                    return Err(CheckedFloatMapValidationError::UnsupportedOperation(
                        position,
                    ));
                }
                has_checked = true;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: got,
                ..
            }) => {
                if got != index {
                    return Err(CheckedFloatMapValidationError::InvalidIndex(position));
                }
                let binding_info = find_binding(kernel, binding)?;
                if binding_info.access == PcuBindingAccess::ReadOnly {
                    return Err(CheckedFloatMapValidationError::InvalidBinding(binding));
                }
                has_store = true;
            }
            _ => {
                return Err(CheckedFloatMapValidationError::UnsupportedOperation(
                    position,
                ));
            }
        }
    }
    if !has_checked {
        return Err(CheckedFloatMapValidationError::MissingCheckedOperation);
    }
    if !has_store {
        return Err(CheckedFloatMapValidationError::MissingStoreOrReturn);
    }
    crate::validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| CheckedFloatMapValidationError::InvalidSsa)?;
    Ok(())
}

fn find_binding<'kernel, 'dispatch>(
    kernel: &'kernel PcuDispatchKernelIr<'dispatch>,
    reference: PcuBindingRef,
) -> Result<&'kernel crate::PcuBinding<'dispatch>, CheckedFloatMapValidationError> {
    kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == reference)
        .ok_or(CheckedFloatMapValidationError::InvalidBinding(reference))
}

const fn constant_matches(value: PcuParameterValue, value_type: PcuValueType) -> bool {
    matches!(
        (value, value_type),
        (
            PcuParameterValue::F32(_),
            PcuValueType::Scalar(crate::PcuScalarType::F32)
        ) | (
            PcuParameterValue::F64(_),
            PcuValueType::Scalar(crate::PcuScalarType::F64)
        )
    )
}

#[cfg(test)]
mod tests;
