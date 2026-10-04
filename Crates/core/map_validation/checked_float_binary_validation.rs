//! Bounded structural admission for checked scalar floating-point maps.

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

/// Validates a checked Add/Sub/Mul/Div indexed map and its frozen underflow policy.
///
/// Two read-only value bindings feed one writable value binding. The body consists of two
/// homogeneous scalar loads, exactly one checked binary operation and one matching store, followed by Return.
/// It may execute directly using `InvocationId` or inside one `GridStrideLoop` using
/// `GridStrideId`. Input loads may use `BindingElementZero` for broadcast. `scalar_caps` is the backend's
/// direct value-type floor. Binary16/32/64 and named BF16/OFP8 formats have a checked
/// scalar contract; vector/matrix and wide-float arithmetic are outside this profile.
/// Structural validation does not establish an executable backend implementation.
///
/// # Errors
///
/// Returns the first interface, capability, binding, operation, indexing, or SSA failure.
pub fn validate_checked_float_binary_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    op: PcuDispatchFloatBinaryOp,
    underflow_policy: PcuFloatUnderflowPolicy,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), CheckedFloatBinaryMapValidationError> {
    validate_canonical_binary_kernel(kernel, value_type, op, underflow_policy, scalar_caps, false)
}

// The legacy distinct-input profile stays narrow until providers explicitly adopt the
// detached operand schema. Broadening structural admission alone is not executable proof.
#[allow(clippy::too_many_lines)] // One cold profile validates the complete interface, body and SSA.
fn validate_canonical_binary_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    op: PcuDispatchFloatBinaryOp,
    underflow_policy: PcuFloatUnderflowPolicy,
    scalar_caps: PcuValueTypeCaps,
    repeated_loads: bool,
) -> Result<(), CheckedFloatBinaryMapValidationError> {
    let required_scalar_cap = match value_type {
        PcuValueType::Scalar(
            scalar @ (crate::PcuScalarType::F16
            | crate::PcuScalarType::BF16
            | crate::PcuScalarType::F32
            | crate::PcuScalarType::F64
            | crate::PcuScalarType::F8E4M3FN
            | crate::PcuScalarType::F8E5M2),
        ) => PcuValueTypeCaps::for_scalar(scalar),
        _ => return Err(CheckedFloatBinaryMapValidationError::UnsupportedRequirements),
    };
    if !kernel.ports.is_empty()
        || !kernel.parameters.is_empty()
        || !(kernel.bindings.len() == 3 || repeated_loads && kernel.bindings.len() == 2)
    {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedInterface);
    }
    let allowed_types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
        | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        | PcuDispatchFeatureCaps::RANGE_CLAMP;
    // Guard flat-map admission before recursive capability scanners.
    if kernel.has_nested_grid_stride_loop() {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            0,
        ));
    }
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
        kernel.bindings[kernel.bindings.len() - 2].reference(),
        kernel.bindings[kernel.bindings.len() - 1].reference(),
    ];
    if kernel.bindings[0].access != PcuBindingAccess::ReadOnly
        || kernel.bindings[kernel.bindings.len() - 2].access != PcuBindingAccess::ReadOnly
        || kernel.bindings[kernel.bindings.len() - 1].access == PcuBindingAccess::ReadOnly
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
        if loaded[input] && !repeated_loads {
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
    if !repeated_loads && !loaded.into_iter().all(core::convert::identity) {
        return Err(CheckedFloatBinaryMapValidationError::UnsupportedOperation(
            2,
        ));
    }
    crate::validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| CheckedFloatBinaryMapValidationError::InvalidSsa)?;
    Ok(())
}

mod operand_schema;
pub use operand_schema::*;

#[cfg(test)]
mod tests;
