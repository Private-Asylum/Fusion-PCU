//! Structural admission for the bounded HIP Dispatch lowering profiles.

use super::{
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
    PcuF32MapValidationError,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
    RocmLowerError,
    validate_f32_map_kernel,
    validate_f64_map_kernel,
    validate_i16_map_kernel,
    validate_i32_checked_div_rem_kernel,
    validate_i32_map_kernel,
    validate_i64_map_kernel,
    validate_i8_map_kernel,
    validate_u16_map_kernel,
    validate_u32_checked_div_rem_kernel,
    validate_u32_identity_kernel,
    validate_u32_map_kernel,
    validate_u64_checked_div_rem_kernel,
    validate_u64_identity_kernel,
    validate_u64_map_kernel,
    validate_u8_map_kernel,
};

#[allow(clippy::too_many_lines)] // One pass keeps the supported IR subset's validation rules together.
pub(super) fn validate_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), RocmLowerError> {
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(RocmLowerError::InvalidKernelShape);
    }
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() {
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8))
    }) {
        return validate_i8_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::U8))
    }) {
        return validate_u8_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16))
    }) {
        return validate_i16_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::U16))
    }) {
        return validate_u16_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::u32()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u32_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_u32_identity_kernel(kernel).is_ok() || validate_u32_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::i32()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i32_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_i32_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::u64()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u64_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_u64_identity_kernel(kernel).is_ok() || validate_u64_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::i64()))
    {
        if validate_i64_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::f64()))
    {
        if validate_f64_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(RocmLowerError::UnsupportedKernelInterface);
    }
    let allowed_types = PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(RocmLowerError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().copied().enumerate() {
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.binding_type != PcuBindingType::Value(PcuValueType::f32())
            || binding.builtin.is_some()
        {
            return Err(RocmLowerError::InvalidBinding(PcuBindingRef::new(
                binding.set,
                binding.binding,
            )));
        }
        if kernel.bindings[..index]
            .iter()
            .any(|other| other.set == binding.set && other.binding == binding.binding)
        {
            return Err(RocmLowerError::InvalidBinding(PcuBindingRef::new(
                binding.set,
                binding.binding,
            )));
        }
    }

    let mut definitions = Vec::new();
    let mut has_store = false;
    let mut has_grid_stride_loop = false;
    let mut saw_return = false;
    for (index, op) in kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(RocmLowerError::OperationAfterReturn);
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index,
            }) => {
                require_load_index(index)?;
                require_binding(kernel, binding, false)?;
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(RocmLowerError::UnsupportedConstant);
                }
                define(&mut definitions, result)?;
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
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(RocmLowerError::UnsupportedAlu(op));
                }
                require_defined(&definitions, lhs)?;
                require_defined(&definitions, rhs)?;
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index,
                value,
            }) => {
                require_store_index(index)?;
                require_binding(kernel, binding, true)?;
                require_defined(&definitions, value)?;
                has_store = true;
            }
            PcuDispatchOp::GridStrideLoop { extent, body } => {
                if has_grid_stride_loop || extent == 0 {
                    return Err(RocmLowerError::UnsupportedOperation {
                        index,
                        support: op.support_flag(),
                    });
                }
                validate_grid_stride_body(kernel, body)?;
                has_grid_stride_loop = true;
                has_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => {
                return Err(RocmLowerError::UnsupportedOperation {
                    index,
                    support: op.support_flag(),
                });
            }
        }
    }
    if has_store && saw_return && !has_grid_stride_loop {
        validate_f32_map_kernel(kernel).map_err(|error| map_common_validation_error(error, kernel))
    } else if has_store && saw_return {
        if matches!(
            kernel.ops,
            [
                PcuDispatchOp::GridStrideLoop { .. },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return)
            ]
        ) {
            Ok(())
        } else {
            Err(RocmLowerError::UnsupportedKernelInterface)
        }
    } else {
        Err(RocmLowerError::MissingStoreOrReturn)
    }
}

pub(super) fn kernel_uses_checked_div_rem(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. }) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => body.iter().any(|body_op| {
            matches!(
                body_op,
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. })
            )
        }),
        _ => false,
    })
}

fn validate_grid_stride_body(
    kernel: &PcuDispatchKernelIr<'_>,
    body: &[PcuDispatchOp<'_>],
) -> Result<(), RocmLowerError> {
    let mut definitions = Vec::new();
    let mut has_store = false;
    for (index, op) in body.iter().copied().enumerate() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: PcuDispatchIndex::GridStrideId,
            }) => {
                require_binding(kernel, binding, false)?;
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: PcuDispatchIndex::BindingElementZero,
            }) => {
                require_binding(kernel, binding, false)?;
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: PcuDispatchIndex::GridStrideId,
                value,
            }) => {
                require_binding(kernel, binding, true)?;
                require_defined(&definitions, value)?;
                has_store = true;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(RocmLowerError::UnsupportedConstant);
                }
                define(&mut definitions, result)?;
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
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(RocmLowerError::UnsupportedAlu(op));
                }
                require_defined(&definitions, lhs)?;
                require_defined(&definitions, rhs)?;
                define(&mut definitions, result)?;
            }
            _ => {
                return Err(RocmLowerError::UnsupportedOperation {
                    index,
                    support: op.support_flag(),
                });
            }
        }
    }
    if has_store {
        Ok(())
    } else {
        Err(RocmLowerError::MissingStoreOrReturn)
    }
}

fn map_common_validation_error(
    error: PcuF32MapValidationError,
    kernel: &PcuDispatchKernelIr<'_>,
) -> RocmLowerError {
    match error {
        PcuF32MapValidationError::UnsupportedInterface => {
            RocmLowerError::UnsupportedKernelInterface
        }
        PcuF32MapValidationError::InvalidBinding(binding)
        | PcuF32MapValidationError::DuplicateBinding(binding) => {
            RocmLowerError::InvalidBinding(binding)
        }
        PcuF32MapValidationError::UnsupportedOperation(index) => {
            RocmLowerError::UnsupportedOperation {
                index,
                support: kernel.ops[index].support_flag(),
            }
        }
        PcuF32MapValidationError::InvalidIndex(_) => RocmLowerError::UnsupportedIndex,
        PcuF32MapValidationError::InvalidValue(value) => RocmLowerError::InvalidValue(value),
        PcuF32MapValidationError::DuplicateValue(value) => RocmLowerError::DuplicateValue(value),
        PcuF32MapValidationError::MissingStore | PcuF32MapValidationError::MissingReturn => {
            RocmLowerError::MissingStoreOrReturn
        }
    }
}

fn define(
    definitions: &mut Vec<PcuDispatchValueId>,
    result: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    if definitions.contains(&result) {
        return Err(RocmLowerError::DuplicateValue(result));
    }
    definitions.push(result);
    Ok(())
}

fn require_defined(
    definitions: &[PcuDispatchValueId],
    value: PcuDispatchValueId,
) -> Result<(), RocmLowerError> {
    if definitions.contains(&value) {
        Ok(())
    } else {
        Err(RocmLowerError::UndefinedValue(value))
    }
}

const fn require_load_index(index: PcuDispatchIndex) -> Result<(), RocmLowerError> {
    if matches!(
        index,
        PcuDispatchIndex::InvocationId | PcuDispatchIndex::BindingElementZero
    ) {
        Ok(())
    } else {
        Err(RocmLowerError::UnsupportedIndex)
    }
}

fn require_store_index(index: PcuDispatchIndex) -> Result<(), RocmLowerError> {
    if index == PcuDispatchIndex::InvocationId {
        Ok(())
    } else {
        Err(RocmLowerError::UnsupportedIndex)
    }
}

fn require_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    reference: PcuBindingRef,
    write: bool,
) -> Result<(), RocmLowerError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.set == reference.set && binding.binding == reference.binding)
    else {
        return Err(RocmLowerError::InvalidBinding(reference));
    };
    let allowed = if write {
        matches!(
            binding.access,
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        )
    } else {
        matches!(
            binding.access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
        )
    };
    if allowed {
        Ok(())
    } else {
        Err(RocmLowerError::InvalidBindingAccess(reference))
    }
}
