//! Bounded admission for checked binary32 and binary64 maps.

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuFloatUnderflowPolicy,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural or SSA failure within the checked scalar floating-point binary profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedFloatBinaryMapValidationError {
    UnsupportedInterface,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    InvalidValue(PcuDispatchValueId),
    InvalidSsa,
    MissingReturn,
}

/// Validates a checked binary32 or binary64 Add/Sub/Mul indexed map and its frozen underflow policy.
///
/// Two read-only value bindings feed one writable value binding. The body consists of two
/// homogeneous scalar loads, exactly one checked binary operation and one matching store, followed by Return.
/// It may execute directly using `InvocationId` or inside one `GridStrideLoop` using
/// `GridStrideId`. Input loads may use `BindingElementZero` for broadcast. `scalar_caps` is the backend's
/// direct value-type floor; this profile never admits half, vector, or matrix arithmetic.
///
/// # Errors
///
/// Returns the first interface, capability, binding, operation, indexing, or SSA failure.
#[allow(clippy::too_many_lines)]
pub fn validate_checked_float_binary_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    op: PcuDispatchFloatBinaryOp,
    underflow_policy: PcuFloatUnderflowPolicy,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), CheckedFloatBinaryMapValidationError> {
    let required_scalar_cap = if value_type == PcuValueType::f32() {
        PcuValueTypeCaps::FLOAT32
    } else if value_type == PcuValueType::f64() {
        PcuValueTypeCaps::FLOAT64
    } else {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedRequirements);
    };
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.len() != 3 {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedInterface);
    }
    let allowed_types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
        | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        | PcuDispatchFeatureCaps::RANGE_CLAMP;
    if kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
        || !scalar_caps.contains(required_scalar_cap)
    {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(CheckedFloatBinaryMapValidationError::DuplicateBinding(
                reference,
            ));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(CheckedFloatBinaryMapValidationError::InvalidBinding(
                reference,
            ));
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
        return Err(CheckedFloatBinaryMapValidationError::InvalidBinding(bad));
    }

    let (body, index) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
                    0,
                ));
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
        _ => return Err(CheckedFloatBinaryMapValidationError::MissingReturn),
    };
    if body.len() != 4 {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            body.len(),
        ));
    }
    let mut loaded = [false; 2];
    for (position, instruction) in body[..2].iter().copied().enumerate() {
        let PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            binding,
            index: got,
            ..
        }) = instruction
        else {
            return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
                position,
            ));
        };
        if got != index && got != PcuDispatchIndex::BindingElementZero {
            return Err(CheckedFloatBinaryMapValidationError::InvalidIndex(position));
        }
        let input = if binding == refs[0] {
            0
        } else if binding == refs[1] {
            1
        } else {
            return Err(CheckedFloatBinaryMapValidationError::InvalidBinding(
                binding,
            ));
        };
        if loaded[input] {
            return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
                position,
            ));
        }
        loaded[input] = true;
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
        value_type: actual_type,
        op: actual_op,
        underflow_policy: actual_policy,
        result,
        ..
    }) = body[2]
    else {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            2,
        ));
    };
    if actual_type != value_type || actual_op != op || actual_policy != underflow_policy {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            2,
        ));
    }
    let PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding,
        index: got,
        value,
    }) = body[3]
    else {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            3,
        ));
    };
    if got != index {
        return Err(CheckedFloatBinaryMapValidationError::InvalidIndex(3));
    }
    if binding != refs[2] {
        return Err(CheckedFloatBinaryMapValidationError::InvalidBinding(
            binding,
        ));
    }
    if value != result {
        return Err(CheckedFloatBinaryMapValidationError::InvalidValue(value));
    }
    if !loaded.into_iter().all(core::convert::identity) {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            2,
        ));
    }
    crate::validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| CheckedFloatBinaryMapValidationError::InvalidSsa)?;
    Ok(())
}

#[cfg(test)]
mod tests;
