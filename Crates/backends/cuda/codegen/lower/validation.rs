//! Structural admission for the bounded CUDA Dispatch lowering profiles.

#[rustfmt::skip]
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
    PcuDispatchIntegerBinaryOp,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuF32MapValidationError,
    PcuParameterValue,
    PcuValueType,
    PcuValueTypeCaps,
    CudaLowerError,
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
pub(super) fn validate_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), CudaLowerError> {
    // Only independently qualified synthesized low-float and exact integer maps opt in.
    // The neutral descriptor supplies structural eligibility, never broad backend conformance.
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
        && fusion_pcu::describe_portable_v1_map(kernel).is_err()
        && fusion_pcu::describe_portable_v1_integer_map(kernel).is_err()
        && fusion_pcu::describe_portable_v1_integer_div_rem_map(kernel).is_err()
        && fusion_pcu::describe_portable_v1_unary_map(kernel).is_err()
    {
        return Err(CudaLowerError::UnsupportedRequirements);
    }
    // Checked division has only fatal zero/overflow domains; no quotient/remainder Clamp law.
    // Reject incompatible headers before compilation without changing valid emitter bytes/IDs.
    if kernel_uses_checked_div_rem(kernel)
        && kernel.numerical_requirements.range_policy != fusion_pcu::PcuRangePolicy::Reject
    {
        return Err(CudaLowerError::UnsupportedRequirements);
    }
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(CudaLowerError::InvalidKernelShape);
    }
    if super::composed::has_broadcast_write_hazard(kernel) {
        return Err(CudaLowerError::UnsupportedRequirements);
    }
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() {
        return Err(CudaLowerError::UnsupportedKernelInterface);
    }
    if has_checked_float_conversion(kernel) {
        return validate_checked_float_conversion_map_kernel(
            kernel,
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        )
        .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if low_unary_profile(kernel.ops)
        && let Some(value_type) = checked_float_binary_profile(kernel)
        && matches!(
            value_type,
            PcuValueType::Scalar(
                fusion_pcu::PcuScalarType::F16
                    | fusion_pcu::PcuScalarType::BF16
                    | fusion_pcu::PcuScalarType::F8E4M3FN
                    | fusion_pcu::PcuScalarType::F8E5M2
            )
        )
    {
        return validate_checked_float_map_kernel(
            kernel,
            value_type,
            PcuValueTypeCaps::for_scalar(value_type.scalar_type()),
        )
        .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    // Independent operand-schema admission preserves actual loaded bindings and SSA roles.
    // Existing multi-operation F32/F64 profiles keep their established validator below.
    if super::map_binding_projection(kernel).is_some() {
        return Ok(());
    }
    if let Some(value_type) = checked_float_binary_profile(kernel) {
        if matches!(
            value_type,
            PcuValueType::Scalar(
                fusion_pcu::PcuScalarType::F16
                    | fusion_pcu::PcuScalarType::BF16
                    | fusion_pcu::PcuScalarType::F8E4M3FN
                    | fusion_pcu::PcuScalarType::F8E5M2
            )
        ) {
            let (op, underflow) =
                low_binary_profile(kernel.ops).ok_or(CudaLowerError::UnsupportedKernelInterface)?;
            return fusion_pcu::validate_checked_float_binary_kernel(
                kernel,
                value_type,
                op,
                underflow,
                PcuValueTypeCaps::for_scalar(value_type.scalar_type()),
            )
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
        }
        let scalar_caps = if value_type == PcuValueType::f32() {
            PcuValueTypeCaps::FLOAT32
        } else if value_type == PcuValueType::f64() {
            PcuValueTypeCaps::FLOAT64
        } else {
            return Err(CudaLowerError::UnsupportedKernelInterface);
        };
        return validate_checked_float_map_kernel(kernel, value_type, scalar_caps)
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if let Some((value_type, op)) = checked_integer_binary_profile(kernel) {
        let scalar = value_type.scalar_type();
        return validate_integer_checked_binary_kernel(
            kernel,
            value_type,
            op,
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel_uses_checked_div_rem(kernel)
        && let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        && matches!(
            scalar,
            fusion_pcu::PcuScalarType::I128
                | fusion_pcu::PcuScalarType::U128
                | fusion_pcu::PcuScalarType::I256
                | fusion_pcu::PcuScalarType::U256
                | fusion_pcu::PcuScalarType::I512
                | fusion_pcu::PcuScalarType::U512
        )
    {
        return fusion_pcu::validate_integer_checked_div_rem_kernel(
            kernel,
            PcuValueType::Scalar(scalar),
            PcuValueTypeCaps::for_scalar(scalar),
        )
        .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
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
                | fusion_pcu::PcuScalarType::F8E4M3FN
                | fusion_pcu::PcuScalarType::F8E5M2
                | fusion_pcu::PcuScalarType::F32
                | fusion_pcu::PcuScalarType::F64
                | fusion_pcu::PcuScalarType::I128
                | fusion_pcu::PcuScalarType::U128
                | fusion_pcu::PcuScalarType::I256
                | fusion_pcu::PcuScalarType::U256
                | fusion_pcu::PcuScalarType::I512
                | fusion_pcu::PcuScalarType::U512
                | fusion_pcu::PcuScalarType::F128
                | fusion_pcu::PcuScalarType::F256
        )
        && (fusion_pcu::validate_scalar_identity_kernel(kernel, scalar).is_ok()
            || fusion_pcu::validate_scalar_broadcast_kernel(kernel, scalar).is_ok())
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
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::BF16))
    }) {
        return fusion_pcu::validate_bf16_identity_kernel(kernel)
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i8_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_i8_map_kernel(kernel)
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::U8))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u8_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_u8_map_kernel(kernel)
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::I16))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i16_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_i16_map_kernel(kernel)
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.binding_type
            == PcuBindingType::Value(PcuValueType::Scalar(fusion_pcu::PcuScalarType::U16))
    }) {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u16_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        return validate_u16_map_kernel(kernel)
            .map_err(|_| CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::u32()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u32_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_u32_identity_kernel(kernel).is_ok() || validate_u32_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::i32()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i32_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_i32_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::u64()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_u64_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_u64_identity_kernel(kernel).is_ok() || validate_u64_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::i64()))
    {
        if kernel_uses_checked_div_rem(kernel) {
            validate_i64_checked_div_rem_kernel(kernel)
                .map_err(|_| CudaLowerError::UnsupportedKernelInterface)?;
            return Ok(());
        }
        if validate_i64_map_kernel(kernel).is_ok() {
            return Ok(());
        }
        return Err(CudaLowerError::UnsupportedKernelInterface);
    }
    if kernel
        .bindings
        .iter()
        .any(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::f64()))
    {
        return validate_f64_map_kernel(kernel).map_err(CudaLowerError::InvalidF64Map);
    }
    let allowed_types = PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::SCALAR_VALUES;
    let allowed_features =
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES | PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
    if kernel.required_type_support().bits() & !allowed_types.bits() != 0
        || kernel.required_feature_support().bits() & !allowed_features.bits() != 0
    {
        return Err(CudaLowerError::UnsupportedRequirements);
    }
    for (index, binding) in kernel.bindings.iter().copied().enumerate() {
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.binding_type != PcuBindingType::Value(PcuValueType::f32())
            || binding.builtin.is_some()
        {
            return Err(CudaLowerError::InvalidBinding(PcuBindingRef::new(
                binding.set,
                binding.binding,
            )));
        }
        if kernel.bindings[..index]
            .iter()
            .any(|other| other.set == binding.set && other.binding == binding.binding)
        {
            return Err(CudaLowerError::InvalidBinding(PcuBindingRef::new(
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
            return Err(CudaLowerError::OperationAfterReturn);
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
                    return Err(CudaLowerError::UnsupportedConstant);
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
                    return Err(CudaLowerError::UnsupportedAlu(op));
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
                    return Err(CudaLowerError::UnsupportedOperation {
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
                return Err(CudaLowerError::UnsupportedOperation {
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
            Err(CudaLowerError::UnsupportedKernelInterface)
        }
    } else {
        Err(CudaLowerError::MissingStoreOrReturn)
    }
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

pub(super) fn checked_float_binary_profile(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Option<PcuValueType> {
    kernel.ops.iter().find_map(|op| match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }
            | PcuDispatchDataOp::CheckedFloatUnary { value_type, .. },
        ) => Some(*value_type),
        PcuDispatchOp::GridStrideLoop { body, .. } => {
            body.iter().find_map(|body_op| match body_op {
                PcuDispatchOp::Data(
                    PcuDispatchDataOp::CheckedFloatBinary { value_type, .. }
                    | PcuDispatchDataOp::CheckedFloatUnary { value_type, .. },
                ) => Some(*value_type),
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

pub(super) fn checked_integer_div_rem_profile(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Option<PcuValueType> {
    kernel.ops.iter().find_map(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type, .. }) => {
            Some(*value_type)
        }
        PcuDispatchOp::GridStrideLoop { body, .. } => body.iter().find_map(|op| match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { value_type, .. }) => {
                Some(*value_type)
            }
            _ => None,
        }),
        _ => None,
    })
}

pub(super) fn checked_integer_binary_profile(
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
) -> Result<(), CudaLowerError> {
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
                    return Err(CudaLowerError::UnsupportedConstant);
                }
                define(&mut definitions, result)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                value_type,
            }) => {
                if value_type != PcuValueType::f32() {
                    return Err(CudaLowerError::UnsupportedOperation {
                        index,
                        support: op.support_flag(),
                    });
                }
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                        | PcuDispatchAluOp::Max
                ) {
                    return Err(CudaLowerError::UnsupportedAlu(op));
                }
                require_defined(&definitions, lhs)?;
                require_defined(&definitions, rhs)?;
                define(&mut definitions, result)?;
            }
            _ => {
                return Err(CudaLowerError::UnsupportedOperation {
                    index,
                    support: op.support_flag(),
                });
            }
        }
    }
    if has_store {
        Ok(())
    } else {
        Err(CudaLowerError::MissingStoreOrReturn)
    }
}

const fn map_common_validation_error(
    error: PcuF32MapValidationError,
    kernel: &PcuDispatchKernelIr<'_>,
) -> CudaLowerError {
    match error {
        PcuF32MapValidationError::UnsupportedInterface => {
            CudaLowerError::UnsupportedKernelInterface
        }
        PcuF32MapValidationError::InvalidBinding(binding)
        | PcuF32MapValidationError::DuplicateBinding(binding) => {
            CudaLowerError::InvalidBinding(binding)
        }
        PcuF32MapValidationError::UnsupportedOperation(index) => {
            CudaLowerError::UnsupportedOperation {
                index,
                support: kernel.ops[index].support_flag(),
            }
        }
        PcuF32MapValidationError::InvalidIndex(_) => CudaLowerError::UnsupportedIndex,
        PcuF32MapValidationError::InvalidValue(value) => CudaLowerError::InvalidValue(value),
        PcuF32MapValidationError::DuplicateValue(value) => CudaLowerError::DuplicateValue(value),
        PcuF32MapValidationError::MissingStore | PcuF32MapValidationError::MissingReturn => {
            CudaLowerError::MissingStoreOrReturn
        }
    }
}

fn define(
    definitions: &mut Vec<PcuDispatchValueId>,
    result: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    if definitions.contains(&result) {
        return Err(CudaLowerError::DuplicateValue(result));
    }
    definitions.push(result);
    Ok(())
}

fn require_defined(
    definitions: &[PcuDispatchValueId],
    value: PcuDispatchValueId,
) -> Result<(), CudaLowerError> {
    if definitions.contains(&value) {
        Ok(())
    } else {
        Err(CudaLowerError::UndefinedValue(value))
    }
}

const fn require_load_index(index: PcuDispatchIndex) -> Result<(), CudaLowerError> {
    if matches!(
        index,
        PcuDispatchIndex::InvocationId | PcuDispatchIndex::BindingElementZero
    ) {
        Ok(())
    } else {
        Err(CudaLowerError::UnsupportedIndex)
    }
}

fn require_store_index(index: PcuDispatchIndex) -> Result<(), CudaLowerError> {
    if index == PcuDispatchIndex::InvocationId {
        Ok(())
    } else {
        Err(CudaLowerError::UnsupportedIndex)
    }
}

fn require_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    reference: PcuBindingRef,
    write: bool,
) -> Result<(), CudaLowerError> {
    let Some(binding) = kernel
        .bindings
        .iter()
        .find(|binding| binding.set == reference.set && binding.binding == reference.binding)
    else {
        return Err(CudaLowerError::InvalidBinding(reference));
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
        Err(CudaLowerError::InvalidBindingAccess(reference))
    }
}

pub(super) fn low_binary_profile(
    ops: &[PcuDispatchOp<'_>],
) -> Option<(
    fusion_pcu::PcuDispatchFloatBinaryOp,
    fusion_pcu::PcuFloatUnderflowPolicy,
)> {
    ops.iter().find_map(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
            op,
            underflow_policy,
            ..
        }) => Some((*op, *underflow_policy)),
        PcuDispatchOp::GridStrideLoop { body, .. } => low_binary_profile(body),
        _ => None,
    })
}

fn low_unary_profile(ops: &[PcuDispatchOp<'_>]) -> bool {
    fn count(ops: &[PcuDispatchOp<'_>], unary: &mut usize, other: &mut bool) {
        for op in ops {
            match op {
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { .. }) => *unary += 1,
                PcuDispatchOp::Data(
                    PcuDispatchDataOp::CheckedFloatBinary { .. }
                    | PcuDispatchDataOp::CheckedFloatConvert { .. },
                ) => *other = true,
                PcuDispatchOp::GridStrideLoop { body, .. } => count(body, unary, other),
                _ => (),
            }
        }
    }
    let mut unary = 0;
    let mut other = false;
    count(ops, &mut unary, &mut other);
    unary == 1 && !other
}
