//! Structural admission for the bounded HIP Dispatch lowering profiles.

#[rustfmt::skip]
use super::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuF32MapValidationError,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
    RocmLowerError,
    validate_f32_map_kernel,
    validate_checked_float_map_kernel,
    validate_checked_float_conversion_map_kernel,
    validate_f64_map_kernel,
    validate_i16_map_kernel,
    validate_i16_checked_div_rem_kernel,
    validate_i32_checked_div_rem_kernel,
    validate_i64_checked_div_rem_kernel,
    validate_i32_map_kernel,
    validate_i64_map_kernel,
    validate_i8_map_kernel,
    validate_i8_checked_div_rem_kernel,
    validate_u16_checked_div_rem_kernel,
    validate_u16_map_kernel,
    validate_u32_checked_div_rem_kernel,
    validate_u32_identity_kernel,
    validate_u32_map_kernel,
    validate_u64_checked_div_rem_kernel,
    validate_u64_identity_kernel,
    validate_u64_map_kernel,
    validate_u8_checked_div_rem_kernel,
    validate_integer_checked_binary_kernel,
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
    if contains_raw_value_alu(kernel.ops) {
        let op = kernel
            .ops
            .iter()
            .find_map(|op| match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::Alu { op, .. }) => Some(*op),
                PcuDispatchOp::GridStrideLoop { body, .. } => {
                    body.iter().find_map(|body_op| match body_op {
                        PcuDispatchOp::Data(PcuDispatchDataOp::Alu { op, .. }) => Some(*op),
                        _ => None,
                    })
                }
                _ => None,
            })
            .expect("raw ALU presence was checked");
        return Err(RocmLowerError::UnsupportedAlu(op));
    }
    if has_checked_float_conversion(kernel) {
        return validate_checked_float_conversion_map_kernel(
            kernel,
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        )
        .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if let Some(value_type) = checked_float_binary_profile(kernel) {
        let scalar_caps = if value_type == PcuValueType::f32() {
            PcuValueTypeCaps::FLOAT32
        } else if value_type == PcuValueType::f64() {
            PcuValueTypeCaps::FLOAT64
        } else {
            return Err(RocmLowerError::UnsupportedKernelInterface);
        };
        return validate_checked_float_map_kernel(kernel, value_type, scalar_caps)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if let Some(value_type) = checked_float_unary_profile(kernel) {
        let scalar_caps = if value_type == PcuValueType::f32() {
            PcuValueTypeCaps::FLOAT32
        } else if value_type == PcuValueType::f64() {
            PcuValueTypeCaps::FLOAT64
        } else {
            return Err(RocmLowerError::UnsupportedKernelInterface);
        };
        return validate_checked_float_map_kernel(kernel, value_type, scalar_caps)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if let Some((value_type, op)) = checked_integer_binary_profile(kernel) {
        let scalar = value_type.scalar_type();
        return validate_integer_checked_binary_kernel(
            kernel,
            value_type,
            op,
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
        kernel.bindings.first().map(|binding| binding.binding_type)
        && matches!(
            scalar,
            fusion_pcu::PcuScalarType::I8
                | fusion_pcu::PcuScalarType::U8
                | fusion_pcu::PcuScalarType::I16
                | fusion_pcu::PcuScalarType::U16
                | fusion_pcu::PcuScalarType::I32
                | fusion_pcu::PcuScalarType::U32
                | fusion_pcu::PcuScalarType::I64
                | fusion_pcu::PcuScalarType::U64
                | fusion_pcu::PcuScalarType::F16
                | fusion_pcu::PcuScalarType::BF16
                | fusion_pcu::PcuScalarType::F32
                | fusion_pcu::PcuScalarType::F64
        )
        && fusion_pcu::validate_scalar_identity_kernel(kernel, scalar).is_ok()
    {
        return Ok(());
    }
    if super::mixed_widening_profile(kernel).is_some() {
        return Ok(());
    }
    if super::half_conversion_profile(kernel).is_some() {
        return Ok(());
    }
    if super::is_exact_f32_f64_profile(kernel) {
        return Ok(());
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::F16))
    }) {
        return fusion_pcu::validate_f16_identity_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16))
    }) {
        return fusion_pcu::validate_bf16_identity_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i8_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_i8_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::U8))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u8_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_u8_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i16_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_i16_map_kernel(kernel)
            .map_err(|_| RocmLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::U16))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u16_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
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
        if kernel_uses_checked_div_rem(kernel) {
            validate_i64_checked_div_rem_kernel(kernel)
                .map_err(|_| RocmLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
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
        return validate_f64_map_kernel(kernel).map_err(RocmLowerError::InvalidF64Map);
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
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu { op, .. }) => {
                return Err(RocmLowerError::UnsupportedAlu(op));
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

fn contains_raw_value_alu(ops: &[PcuDispatchOp<'_>]) -> bool {
    ops.iter().any(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::Alu { .. }) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => contains_raw_value_alu(body),
        _ => false,
    })
}

pub(super) fn kernel_uses_checked_fault(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::CheckedDivRem { .. }
            | PcuDispatchDataOp::CheckedIntegerBinary { .. }
            | PcuDispatchDataOp::CheckedFloatBinary { .. }
            | PcuDispatchDataOp::CheckedFloatUnary { .. }
            | PcuDispatchDataOp::CheckedFloatConvert { .. },
        ) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => body.iter().any(|body_op| {
            matches!(
                body_op,
                PcuDispatchOp::Data(
                    PcuDispatchDataOp::CheckedDivRem { .. }
                        | PcuDispatchDataOp::CheckedIntegerBinary { .. }
                        | PcuDispatchDataOp::CheckedFloatBinary { .. }
                        | PcuDispatchDataOp::CheckedFloatUnary { .. }
                        | PcuDispatchDataOp::CheckedFloatConvert { .. }
                )
            )
        }),
        _ => false,
    })
}

fn has_checked_float_conversion(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    fn contains(ops: &[PcuDispatchOp<'_>]) -> bool {
        ops.iter().any(|op| match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert { .. }) => true,
            PcuDispatchOp::GridStrideLoop { body, .. } => contains(body),
            _ => false,
        })
    }
    contains(kernel.ops)
}

fn checked_float_binary_profile(kernel: &PcuDispatchKernelIr<'_>) -> Option<PcuValueType> {
    kernel.ops.iter().find_map(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }) => {
            Some(*value_type)
        }
        PcuDispatchOp::GridStrideLoop { body, .. } => {
            body.iter().find_map(|body_op| match body_op {
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                    value_type, ..
                }) => Some(*value_type),
                _ => None,
            })
        }
        _ => None,
    })
}

fn checked_float_unary_profile(kernel: &PcuDispatchKernelIr<'_>) -> Option<PcuValueType> {
    kernel.ops.iter().find_map(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type,
            op: PcuDispatchFloatUnaryOp::Relu | PcuDispatchFloatUnaryOp::Neg,
            ..
        }) => Some(*value_type),
        PcuDispatchOp::GridStrideLoop { body, .. } => {
            body.iter().find_map(|body_op| match body_op {
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                    value_type,
                    op: PcuDispatchFloatUnaryOp::Relu | PcuDispatchFloatUnaryOp::Neg,
                    ..
                }) => Some(*value_type),
                _ => None,
            })
        }
        _ => None,
    })
}

fn kernel_uses_checked_div_rem(kernel: &PcuDispatchKernelIr<'_>) -> bool {
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

fn checked_integer_binary_profile(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Option<(PcuValueType, PcuDispatchIntegerBinaryOp)> {
    kernel.ops.iter().find_map(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { value_type, op, .. }) => {
            Some((*value_type, *op))
        }
        PcuDispatchOp::GridStrideLoop { body, .. } => {
            body.iter().find_map(|body_op| match body_op {
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary {
                    value_type,
                    op,
                    ..
                }) => Some((*value_type, *op)),
                _ => None,
            })
        }
        _ => None,
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
                index: PcuDispatchIndex::GridStrideId | PcuDispatchIndex::BindingElementZero,
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
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu { op, .. }) => {
                return Err(RocmLowerError::UnsupportedAlu(op));
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

const fn map_common_validation_error(
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
