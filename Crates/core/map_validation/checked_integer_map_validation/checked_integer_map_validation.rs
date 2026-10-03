//! Checked integer composition shares cold resource/SSA laws with scalar maps.
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
    PcuParameterValue,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural failure while validating a composed checked integer scalar map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckedIntegerMapValidationError {
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

/// Validate a homogeneous checked Add/Sub/Mul map over all fourteen sealed integer widths.
///
/// Supports direct or one canonical grid-stride body, ordinary indexed stores,
/// indexed/element-zero loads and matching narrow constants. Full declarations
/// and typed SSA remain validated. Each checked instruction retains its own
/// range policy independently of the complete original header. This grants no
/// provider execution, Portable conformance, alias or publication permission.
/// Division, unchecked ALU, conversions and arbitrary control flow are refused.
///
/// # Errors
/// Returns the first structural, interface, indexing or typed SSA failure.
#[allow(clippy::too_many_lines)] // Keep the allowlisted execution profile auditable in one pass.
pub fn validate_checked_integer_map_kernel(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<(), CheckedIntegerMapValidationError> {
    // Structural eligibility follows the sealed integer reference contract.
    // Provider execution and numerical reproducibility are separate admission.
    let cap = match value_type {
        PcuValueType::Scalar(scalar)
            if super::typed_dispatch::is_supported_checked_integer(value_type) =>
        {
            PcuValueTypeCaps::for_scalar(scalar)
        }
        _ => return Err(CheckedIntegerMapValidationError::UnsupportedType),
    };
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.is_empty() {
        return Err(CheckedIntegerMapValidationError::UnsupportedInterface);
    }
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(CheckedIntegerMapValidationError::InvalidLogicalShape);
    }
    let allowed_types = scalar_caps | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features = PcuDispatchFeatureCaps::MUTABLE_RESOURCES
        | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
        | PcuDispatchFeatureCaps::RANGE_CLAMP;
    if !scalar_caps.contains(cap)
        || kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(CheckedIntegerMapValidationError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == reference)
        {
            return Err(CheckedIntegerMapValidationError::DuplicateBinding(
                reference,
            ));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(value_type)
        {
            return Err(CheckedIntegerMapValidationError::InvalidBinding(reference));
        }
    }

    let (body, index) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => {
            if *extent == 0 {
                return Err(CheckedIntegerMapValidationError::UnsupportedOperation(0));
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
        _ => return Err(CheckedIntegerMapValidationError::MissingStoreOrReturn),
    };
    if body.is_empty() {
        return Err(CheckedIntegerMapValidationError::MissingCheckedOperation);
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
                    return Err(CheckedIntegerMapValidationError::InvalidIndex(position));
                }
                let binding_info = find_binding(kernel, binding)?;
                if binding_info.access == PcuBindingAccess::WriteOnly {
                    return Err(CheckedIntegerMapValidationError::InvalidBinding(binding));
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { value, .. }) => {
                if !constant_matches(value, value_type) {
                    return Err(CheckedIntegerMapValidationError::UnsupportedOperation(
                        position,
                    ));
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                value_type: actual,
                ..
            }) => {
                if actual != value_type {
                    return Err(CheckedIntegerMapValidationError::UnsupportedOperation(
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
                    return Err(CheckedIntegerMapValidationError::InvalidIndex(position));
                }
                let binding_info = find_binding(kernel, binding)?;
                if binding_info.access == PcuBindingAccess::ReadOnly {
                    return Err(CheckedIntegerMapValidationError::InvalidBinding(binding));
                }
                has_store = true;
            }
            _ => {
                return Err(CheckedIntegerMapValidationError::UnsupportedOperation(
                    position,
                ));
            }
        }
    }
    if !has_checked {
        return Err(CheckedIntegerMapValidationError::MissingCheckedOperation);
    }
    if !has_store {
        return Err(CheckedIntegerMapValidationError::MissingStoreOrReturn);
    }
    crate::validate_typed_dispatch_value_flow(kernel)
        .map_err(|_| CheckedIntegerMapValidationError::InvalidSsa)?;
    Ok(())
}

fn find_binding<'kernel, 'dispatch>(
    kernel: &'kernel PcuDispatchKernelIr<'dispatch>,
    reference: PcuBindingRef,
) -> Result<&'kernel crate::PcuBinding<'dispatch>, CheckedIntegerMapValidationError> {
    kernel
        .bindings
        .iter()
        .find(|binding| binding.reference() == reference)
        .ok_or(CheckedIntegerMapValidationError::InvalidBinding(reference))
}

const fn constant_matches(value: PcuParameterValue, value_type: PcuValueType) -> bool {
    matches!(
        (value, value_type),
        (
            PcuParameterValue::U8(_),
            PcuValueType::Scalar(PcuScalarType::U8)
        ) | (
            PcuParameterValue::U16(_),
            PcuValueType::Scalar(PcuScalarType::U16)
        ) | (
            PcuParameterValue::U32(_),
            PcuValueType::Scalar(PcuScalarType::U32)
        ) | (
            PcuParameterValue::U64(_),
            PcuValueType::Scalar(PcuScalarType::U64)
        ) | (
            PcuParameterValue::I8(_),
            PcuValueType::Scalar(PcuScalarType::I8)
        ) | (
            PcuParameterValue::I16(_),
            PcuValueType::Scalar(PcuScalarType::I16)
        ) | (
            PcuParameterValue::I32(_),
            PcuValueType::Scalar(PcuScalarType::I32)
        ) | (
            PcuParameterValue::I64(_),
            PcuValueType::Scalar(PcuScalarType::I64)
        )
    )
}

/// Project actual integer resources after complete checked-map validation.
///
/// Shares first-access ordering, view versus initial-content spans, same-lane
/// store/load dependencies and cross-index hazards with floating composition.
/// Caller-chosen capacity creates no universal PCU binding limit. All declared
/// bindings are validated before unused declarations disappear from projection.
/// The result retains the exact numerical header, not an executable capability.
///
/// # Errors
/// Returns structural rejection or insufficient caller-chosen resource capacity.
pub fn assess_checked_integer_map_resources<const CAPACITY: usize>(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<
    crate::CheckedScalarMapResourceSchema<CAPACITY>,
    crate::CheckedScalarMapResourceError<CheckedIntegerMapValidationError>,
> {
    validate_checked_integer_map_kernel(kernel, value_type, scalar_caps)
        .map_err(crate::CheckedScalarMapResourceError::InvalidMap)?;
    super::checked_scalar_resources::project(
        kernel,
        value_type,
        CheckedIntegerMapValidationError::InvalidBinding,
    )
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
