//! Structural admission for bit-preserving f16/bf16 identity copies.

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
    PcuDispatchValueId,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuHalfIdentityValidationError {
    UnsupportedInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    MissingStore,
    MissingReturn,
}

/// Admit only bit-preserving identity transport for IEEE binary16 storage.
///
/// # Errors
///
/// Returns the first interface, type, index, value-flow, or access violation.
pub fn validate_f16_identity_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuHalfIdentityValidationError> {
    validate_half_identity_kernel(kernel, PcuScalarType::F16)
}

/// Admit only bit-preserving identity transport for bfloat16 storage.
///
/// # Errors
///
/// Returns the first interface, type, index, value-flow, or access violation.
pub fn validate_bf16_identity_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuHalfIdentityValidationError> {
    validate_half_identity_kernel(kernel, PcuScalarType::BF16)
}

fn validate_half_identity_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
) -> Result<(), PcuHalfIdentityValidationError> {
    validate_half_interface(kernel, scalar)?;
    if let [
        PcuDispatchOp::GridStrideLoop { extent, body },
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ] = kernel.ops
    {
        return validate_grid_copy(kernel, *extent, body);
    }
    validate_direct_copy(kernel)
}

fn validate_half_interface(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
) -> Result<(), PcuHalfIdentityValidationError> {
    use PcuHalfIdentityValidationError as E;
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 2 {
        return Err(E::UnsupportedInterface);
    }
    let supported_type = match scalar {
        PcuScalarType::F16 => PcuValueTypeCaps::FLOAT16,
        PcuScalarType::BF16 => PcuValueTypeCaps::BFLOAT16,
        _ => unreachable!(),
    } | PcuValueTypeCaps::SCALAR_VALUES;
    let supported_features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !supported_type.bits() != 0
        || kernel.required_feature_support().bits() & !supported_features.bits() != 0
    {
        return Err(E::UnsupportedRequirements);
    }
    let value_type = PcuValueType::Scalar(scalar);
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(E::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(E::InvalidBinding(reference));
        }
    }

    Ok(())
}

fn validate_grid_copy(
    kernel: &PcuDispatchKernelIr<'_>,
    extent: u32,
    body: &[PcuDispatchOp<'_>],
) -> Result<(), PcuHalfIdentityValidationError> {
    use PcuHalfIdentityValidationError as E;
    if extent == 0 {
        return Err(E::UnsupportedOperation(0));
    }
    let [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding: input,
            index: PcuDispatchIndex::GridStrideId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: output,
            index: PcuDispatchIndex::GridStrideId,
            value,
        }),
    ] = body
    else {
        return Err(E::UnsupportedOperation(0));
    };
    if result.0 == 0 || value != result {
        return Err(E::InvalidValue(*value));
    }
    if input == output {
        return Err(E::InvalidBinding(*output));
    }
    check_binding(kernel, *input, false)?;
    check_binding(kernel, *output, true)
}

fn validate_direct_copy(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuHalfIdentityValidationError> {
    use PcuHalfIdentityValidationError as E;
    if kernel.ops.len() < 3 {
        return Err(match kernel.ops.len() {
            0 => E::UnsupportedOperation(0),
            1 => E::MissingStore,
            _ => E::MissingReturn,
        });
    }
    if kernel.ops.len() > 3 {
        return Err(E::UnsupportedOperation(3));
    }
    let (loaded, input) = match kernel.ops[0] {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result,
            binding,
            index: PcuDispatchIndex::InvocationId,
        }) => (result, binding),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { .. }) => {
            return Err(E::InvalidIndex(0));
        }
        _ => return Err(E::UnsupportedOperation(0)),
    };
    if loaded.0 == 0 {
        return Err(E::InvalidValue(loaded));
    }
    check_binding(kernel, input, false)?;
    match kernel.ops[1] {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding,
            index: PcuDispatchIndex::InvocationId,
            value,
        }) => {
            if value != loaded {
                return Err(E::InvalidValue(value));
            }
            if binding == input {
                return Err(E::InvalidBinding(binding));
            }
            check_binding(kernel, binding, true)?;
        }
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { .. }) => {
            return Err(E::InvalidIndex(1));
        }
        _ => return Err(E::MissingStore),
    }
    if !matches!(
        kernel.ops[2],
        PcuDispatchOp::Control(PcuDispatchControlOp::Return)
    ) {
        return Err(E::MissingReturn);
    }
    Ok(())
}

fn check_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    target: PcuBindingRef,
    write: bool,
) -> Result<(), PcuHalfIdentityValidationError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == target)
    else {
        return Err(PcuHalfIdentityValidationError::InvalidBinding(target));
    };
    if (write && binding.access == PcuBindingAccess::ReadOnly)
        || (!write && binding.access == PcuBindingAccess::WriteOnly)
    {
        Err(PcuHalfIdentityValidationError::InvalidBinding(target))
    } else {
        Ok(())
    }
}
