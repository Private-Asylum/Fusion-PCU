//! Admission for mixed-width checked floating scalar maps.

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchCheckedFloatConversion,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural or SSA failure in the mixed-width checked float conversion map profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedFloatConversionMapValidationError {
    UnsupportedInterface,
    InvalidLogicalShape,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidSsa,
    MissingConversion,
    MissingStore,
    MissingReturn,
}

/// Validates a mixed F32/F64 map containing checked F64-to-F32 conversion.
///
/// Bodies may contain any bounded sequence of F32/F64 constants, loads, stores, checked
/// F32/F64 arithmetic, and checked F64-to-F32 conversions. There must be at least one checked
/// conversion and one store; loads are optional for literal-only conversions. Inputs are
/// readable F32/F64 bindings, and outputs are writable F32/F64 bindings. Read/write bindings may be
/// both loaded and stored. Loads use the invocation/grid-stride index or element-zero broadcast;
/// stores use the invocation/grid-stride index. SSA and type flow are checked by the
/// shared typed verifier. This structural validator does not claim that any provider executes
/// the profile; providers must advertise the corresponding operation and type capabilities.
///
/// # Errors
///
/// Returns the first interface, capability, binding, operation, indexing, or SSA failure.
#[allow(clippy::too_many_lines)]
pub fn validate_checked_float_conversion_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), CheckedFloatConversionMapValidationError> {
    let both_float_widths = PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64;
    if !scalar_caps.contains(both_float_widths) {
        return Err(CheckedFloatConversionMapValidationError::UnsupportedRequirements);
    }
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.is_empty() {
        return Err(CheckedFloatConversionMapValidationError::UnsupportedInterface);
    }
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(CheckedFloatConversionMapValidationError::InvalidLogicalShape);
    }
    let allowed_types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
        | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        | PcuDispatchFeatureCaps::RANGE_CLAMP;
    if kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(CheckedFloatConversionMapValidationError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(CheckedFloatConversionMapValidationError::DuplicateBinding(
                reference,
            ));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || !matches!(
                binding.binding_type,
                PcuBindingType::Value(PcuValueType::Scalar(
                    crate::PcuScalarType::F32 | crate::PcuScalarType::F64
                ))
            )
        {
            return Err(CheckedFloatConversionMapValidationError::InvalidBinding(
                reference,
            ));
        }
    }

    let (body, index) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(CheckedFloatConversionMapValidationError::UnsupportedOperation(0));
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
        _ => return Err(CheckedFloatConversionMapValidationError::MissingReturn),
    };
    let mut has_store = false;
    let mut has_conversion = false;
    for (position, instruction) in body.iter().copied().enumerate() {
        match instruction {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                binding,
                index: got,
                ..
            }) => {
                if got != index && got != PcuDispatchIndex::BindingElementZero {
                    return Err(CheckedFloatConversionMapValidationError::InvalidIndex(
                        position,
                    ));
                }
                let binding_info = find_binding(kernel, binding)?;
                if binding_info.access == PcuBindingAccess::WriteOnly {
                    return Err(CheckedFloatConversionMapValidationError::InvalidBinding(
                        binding,
                    ));
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { value, .. }) => {
                if !matches!(value, PcuParameterValue::F32(_) | PcuParameterValue::F64(_)) {
                    return Err(
                        CheckedFloatConversionMapValidationError::UnsupportedOperation(position),
                    );
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }) => {
                if value_type != PcuValueType::f32() && value_type != PcuValueType::f64() {
                    return Err(
                        CheckedFloatConversionMapValidationError::UnsupportedOperation(position),
                    );
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert { conversion, .. }) => {
                if !matches!(
                    conversion,
                    PcuDispatchCheckedFloatConversion::F64ToF32
                        | PcuDispatchCheckedFloatConversion::F32ToF64
                ) {
                    return Err(
                        CheckedFloatConversionMapValidationError::UnsupportedOperation(position),
                    );
                }
                has_conversion = true;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: got,
                ..
            }) => {
                if got != index {
                    return Err(CheckedFloatConversionMapValidationError::InvalidIndex(
                        position,
                    ));
                }
                let binding_info = find_binding(kernel, binding)?;
                if binding_info.access == PcuBindingAccess::ReadOnly
                    || !matches!(
                        binding_info.binding_type,
                        PcuBindingType::Value(PcuValueType::Scalar(
                            crate::PcuScalarType::F32 | crate::PcuScalarType::F64
                        ))
                    )
                {
                    return Err(CheckedFloatConversionMapValidationError::InvalidBinding(
                        binding,
                    ));
                }
                has_store = true;
            }
            _ => {
                return Err(
                    CheckedFloatConversionMapValidationError::UnsupportedOperation(position),
                );
            }
        }
    }
    if !has_conversion {
        return Err(CheckedFloatConversionMapValidationError::MissingConversion);
    }
    if !has_store {
        return Err(CheckedFloatConversionMapValidationError::MissingStore);
    }
    crate::validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| CheckedFloatConversionMapValidationError::InvalidSsa)?;
    Ok(())
}

fn find_binding<'kernel, 'dispatch>(
    kernel: &'kernel PcuDispatchKernelIr<'dispatch>,
    reference: PcuBindingRef,
) -> Result<&'kernel crate::PcuBinding<'dispatch>, CheckedFloatConversionMapValidationError> {
    kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == reference)
        .ok_or(CheckedFloatConversionMapValidationError::InvalidBinding(
            reference,
        ))
}

#[cfg(test)]
mod tests;
