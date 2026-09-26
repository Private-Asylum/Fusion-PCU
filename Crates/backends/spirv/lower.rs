//! PCU dispatch to SPIR-V lowering entry points.

use alloc::vec::Vec;

use fusion_pcu::{
    PcuF32MapValidationError,
    PcuBindingRef,
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchRayTraceOp,
    PcuDispatchResourceOp,
    PcuDispatchValueOp,
    PcuDispatchValueId,
    PcuParameterValue,
    PcuScalarType,
    PcuValueType,
    validate_f32_map_kernel,
};

use super::{
    PcuSpirvCapability,
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
    PcuSpirvWriter,
};

/// Lowers one PCU dispatch kernel into a SPIR-V module.
///
/// This first cut intentionally emits only an empty compute entry point for kernels whose op
/// stream is empty or explicitly returns. Richer operations are rejected with precise structured
/// errors before any words are written.
///
/// # Errors
///
/// Returns a validation, unsupported capability, unsupported instruction, or sink failure.
pub fn lower_dispatch_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<PcuSpirvModuleInfo, PcuSpirvError> {
    validate_dispatch_for_spirv(kernel, options)?;

    let mut writer = PcuSpirvWriter::new(sink);
    if has_f32_dataflow_ops(kernel) {
        return writer.emit_f32_dataflow_map_module(kernel, options);
    }
    if is_legacy_parallel_float_map_kernel(kernel) {
        return writer.emit_parallel_float_map_module(kernel.entry.name, options);
    }
    writer.emit_minimal_compute_module(kernel.entry.name, options)
}

/// Validates that one dispatch kernel is admissible for the current SPIR-V lowering subset.
///
/// # Errors
///
/// Returns the first precise reason this kernel cannot be lowered.
pub fn validate_dispatch_for_spirv(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
) -> Result<(), PcuSpirvError> {
    require_capability(options, PcuSpirvCapability::Shader)?;
    validate_spirv_version(options)?;
    validate_kernel_signature(kernel, options)?;
    validate_dispatch_features(kernel.required_feature_support())?;

    if has_f32_dataflow_ops(kernel) {
        return validate_f32_dataflow_kernel(kernel);
    }

    if is_legacy_parallel_float_map_kernel(kernel) {
        return Ok(());
    }

    for op in kernel.ops.iter().copied() {
        validate_op_for_spirv(op, kernel, options)?;
    }

    Ok(())
}

const fn validate_spirv_version(options: PcuSpirvLoweringOptions) -> Result<(), PcuSpirvError> {
    let version = options.version.0;
    // SPIR-V's header encodes major/minor/revision in the low 24 bits. This backend
    // advertises the core 1.0 through 1.3 versions only. Newer SPIR-V requires
    // Block/StorageBuffer emission instead of the BufferBlock representation here.
    let major = (version >> 16) & 0xff;
    let minor = (version >> 8) & 0xff;
    let revision = version & 0xff;
    if version & 0xff00_0000 == 0 && major == 1 && minor <= 3 && revision == 0 {
        Ok(())
    } else {
        Err(PcuSpirvError::UnsupportedVersion(options.version))
    }
}

fn validate_kernel_signature(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
) -> Result<(), PcuSpirvError> {
    if kernel.entry.name.is_empty() || kernel.entry.logical_shape.contains(&0) {
        return Err(PcuSpirvError::InvalidKernelSignature);
    }

    for binding in kernel.bindings.iter().copied() {
        if !binding.is_well_formed() {
            return Err(PcuSpirvError::InvalidBinding);
        }
        match binding.binding_type {
            PcuBindingType::Value(value_type) => validate_value_type(value_type, options)?,
            PcuBindingType::Image(image_type) => {
                let required = match binding.access {
                    PcuBindingAccess::ReadOnly => PcuSpirvCapability::Image,
                    PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite => {
                        PcuSpirvCapability::StorageImage
                    }
                };
                require_capability(options, required)?;
                validate_value_type(image_type.texel_type, options)?;
            }
            PcuBindingType::Sampler(_) => require_capability(options, PcuSpirvCapability::Image)?,
            PcuBindingType::AccelerationStructure(_) => {
                if !options
                    .capabilities
                    .supports(PcuSpirvCapability::RayTracing)
                    && !options.capabilities.supports(PcuSpirvCapability::RayQuery)
                {
                    return Err(PcuSpirvError::UnsupportedCapability(
                        PcuSpirvCapability::RayTracing,
                    ));
                }
            }
        }
    }

    for port in kernel.ports.iter().copied() {
        validate_value_type(port.value_type, options)?;
    }

    for parameter in kernel.parameters.iter().copied() {
        validate_value_type(parameter.value_type, options)?;
    }

    Ok(())
}

const fn validate_dispatch_features(features: PcuDispatchFeatureCaps) -> Result<(), PcuSpirvError> {
    if features.contains(PcuDispatchFeatureCaps::COOPERATIVE_SCRATCHPAD) {
        return Err(PcuSpirvError::UnsupportedInstruction(
            PcuDispatchOpCaps::SYNC_BARRIER,
        ));
    }
    Ok(())
}

fn validate_op_for_spirv(
    op: PcuDispatchOp<'_>,
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
) -> Result<(), PcuSpirvError> {
    match op {
        PcuDispatchOp::Control(PcuDispatchControlOp::Return) => Ok(()),
        PcuDispatchOp::Control(_)
        | PcuDispatchOp::Value(_)
        | PcuDispatchOp::Arithmetic(_)
        | PcuDispatchOp::Port(_)
        | PcuDispatchOp::Sync(_)
        | PcuDispatchOp::Intrinsic { .. }
        | PcuDispatchOp::GridStrideLoop { .. } => unsupported(op),
        PcuDispatchOp::Resource(resource) => validate_resource_op(resource, options),
        PcuDispatchOp::Data(data) => {
            Err(PcuSpirvError::UnsupportedInstruction(data.support_flag()))
        }
        PcuDispatchOp::Coordinate(coordinate) => {
            let flag = coordinate.support_flag();
            if flag.contains(PcuDispatchOpCaps::DERIVATIVE_X)
                || flag.contains(PcuDispatchOpCaps::DERIVATIVE_Y)
            {
                require_capability(options, PcuSpirvCapability::CoordinateDerivative)?;
            }
            Err(PcuSpirvError::UnsupportedInstruction(flag))
        }
        PcuDispatchOp::RayTrace(ray) => validate_ray_op(ray, kernel, options),
    }
}

fn validate_resource_op(
    resource: PcuDispatchResourceOp,
    options: PcuSpirvLoweringOptions,
) -> Result<(), PcuSpirvError> {
    match resource {
        PcuDispatchResourceOp::Sample(_) => {
            require_capability(options, PcuSpirvCapability::Image)?;
            Err(PcuSpirvError::UnsupportedInstruction(
                resource.support_flag(),
            ))
        }
        PcuDispatchResourceOp::Store | PcuDispatchResourceOp::Atomic => {
            require_capability(options, PcuSpirvCapability::StorageImage)?;
            Err(PcuSpirvError::UnsupportedInstruction(
                resource.support_flag(),
            ))
        }
        PcuDispatchResourceOp::Load => Err(PcuSpirvError::UnsupportedInstruction(
            resource.support_flag(),
        )),
    }
}

fn validate_ray_op(
    ray: PcuDispatchRayTraceOp,
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
) -> Result<(), PcuSpirvError> {
    match ray {
        PcuDispatchRayTraceOp::TraceRay(trace) => {
            require_capability(options, PcuSpirvCapability::RayTracing)?;
            trace
                .validate(kernel.bindings)
                .map_err(|_| PcuSpirvError::InvalidBinding)?;
        }
        PcuDispatchRayTraceOp::TraceRayInline(trace) => {
            require_capability(options, PcuSpirvCapability::RayQuery)?;
            trace
                .validate(kernel.bindings)
                .map_err(|_| PcuSpirvError::InvalidBinding)?;
        }
        PcuDispatchRayTraceOp::RayQueryProceed
        | PcuDispatchRayTraceOp::RayQueryCommittedStatus
        | PcuDispatchRayTraceOp::RayQueryCommittedDistance
        | PcuDispatchRayTraceOp::RayQueryCommittedInstance
        | PcuDispatchRayTraceOp::RayQueryCommittedPrimitive => {
            require_capability(options, PcuSpirvCapability::RayQuery)?;
        }
        PcuDispatchRayTraceOp::ReportHit { .. }
        | PcuDispatchRayTraceOp::IgnoreHit
        | PcuDispatchRayTraceOp::AcceptHitAndEndSearch
        | PcuDispatchRayTraceOp::PayloadRead { .. }
        | PcuDispatchRayTraceOp::PayloadWrite { .. } => {
            require_capability(options, PcuSpirvCapability::RayTracing)?;
        }
    }

    Err(PcuSpirvError::UnsupportedInstruction(ray.support_flag()))
}

fn validate_value_type(
    value_type: PcuValueType,
    options: PcuSpirvLoweringOptions,
) -> Result<(), PcuSpirvError> {
    match value_type {
        PcuValueType::Scalar(scalar) => validate_scalar_type(scalar),
        PcuValueType::Vector { scalar, lanes } => {
            validate_scalar_type(scalar)?;
            if lanes == 2 || lanes == 3 || lanes == 4 {
                Ok(())
            } else {
                Err(PcuSpirvError::UnsupportedValueType(value_type))
            }
        }
        PcuValueType::Matrix { scalar, rows, cols } => {
            require_capability(options, PcuSpirvCapability::Matrix)?;
            validate_scalar_type(scalar)?;
            if (rows == 2 || rows == 3 || rows == 4) && (cols == 2 || cols == 3 || cols == 4) {
                Ok(())
            } else {
                Err(PcuSpirvError::UnsupportedValueType(value_type))
            }
        }
    }
}

const fn validate_scalar_type(scalar: PcuScalarType) -> Result<(), PcuSpirvError> {
    match scalar {
        PcuScalarType::Bool | PcuScalarType::I32 | PcuScalarType::U32 | PcuScalarType::F32 => {
            Ok(())
        }
        PcuScalarType::I4
        | PcuScalarType::U4
        | PcuScalarType::I8
        | PcuScalarType::U8
        | PcuScalarType::I16
        | PcuScalarType::U16
        | PcuScalarType::I64
        | PcuScalarType::U64
        | PcuScalarType::F16
        | PcuScalarType::BF16
        | PcuScalarType::F64 => Err(PcuSpirvError::UnsupportedValueType(PcuValueType::Scalar(
            scalar,
        ))),
    }
}

const fn require_capability(
    options: PcuSpirvLoweringOptions,
    capability: PcuSpirvCapability,
) -> Result<(), PcuSpirvError> {
    if options.capabilities.supports(capability) {
        Ok(())
    } else {
        Err(PcuSpirvError::UnsupportedCapability(capability))
    }
}

const fn unsupported(op: PcuDispatchOp<'_>) -> Result<(), PcuSpirvError> {
    Err(PcuSpirvError::UnsupportedInstruction(op.support_flag()))
}

fn is_legacy_parallel_float_map_kernel(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    has_parallel_float_bindings(kernel) && has_parallel_float_ops(kernel)
}

fn has_parallel_float_bindings(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    let [input_a, input_b, output] = kernel.bindings else {
        return false;
    };

    is_storage_f32_binding(*input_a, 0, 0, PcuBindingAccess::ReadOnly)
        && is_storage_f32_binding(*input_b, 0, 1, PcuBindingAccess::ReadOnly)
        && matches!(
            output.access,
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        )
        && is_storage_f32_binding_type(*output, 0, 2)
}

fn is_storage_f32_binding(
    binding: fusion_pcu::PcuBinding<'_>,
    set: u32,
    slot: u32,
    access: PcuBindingAccess,
) -> bool {
    binding.access == access && is_storage_f32_binding_type(binding, set, slot)
}

fn is_storage_f32_binding_type(binding: fusion_pcu::PcuBinding<'_>, set: u32, slot: u32) -> bool {
    binding.set == set
        && binding.binding == slot
        && binding.storage == PcuBindingStorageClass::Storage
        && binding.binding_type == PcuBindingType::Value(PcuValueType::f32())
}

const fn has_parallel_float_ops(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    matches!(
        kernel.ops,
        [
            PcuDispatchOp::Resource(PcuDispatchResourceOp::Load),
            PcuDispatchOp::Resource(PcuDispatchResourceOp::Load),
            PcuDispatchOp::Arithmetic(PcuDispatchAluOp::Mul),
            PcuDispatchOp::Arithmetic(PcuDispatchAluOp::Add),
            PcuDispatchOp::Value(PcuDispatchValueOp::Constant),
            PcuDispatchOp::Arithmetic(PcuDispatchAluOp::Add),
            PcuDispatchOp::Resource(PcuDispatchResourceOp::Store),
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ]
    )
}

fn has_f32_dataflow_ops(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    kernel.ops.iter().any(|op| {
        matches!(
            op,
            PcuDispatchOp::Data(_) | PcuDispatchOp::GridStrideLoop { .. }
        )
    })
}

fn validate_f32_dataflow_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), PcuSpirvError> {
    if kernel
        .ops
        .iter()
        .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { .. }))
    {
        return validate_grid_stride_kernel(kernel);
    }
    if !kernel
        .bindings
        .iter()
        .copied()
        .all(is_storage_f32_dataflow_binding)
    {
        return Err(PcuSpirvError::InvalidBinding);
    }

    let mut saw_return = false;
    let mut saw_store = false;
    for (op_index, op) in kernel.ops.iter().copied().enumerate() {
        if saw_return {
            return Err(PcuSpirvError::InvalidKernelSignature);
        }
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: data_index,
            }) => {
                validate_result_slot(kernel, op_index, result)?;
                validate_dataflow_binding(kernel, binding, PcuBindingAccess::ReadOnly)?;
                validate_invocation_index(data_index)?;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant { result, value }) => {
                validate_result_slot(kernel, op_index, result)?;
                if !matches!(value, PcuParameterValue::F32(_)) {
                    return Err(PcuSpirvError::UnsupportedValueType(value.value_type()));
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                validate_result_slot(kernel, op_index, result)?;
                validate_defined_before(kernel, op_index, lhs)?;
                validate_defined_before(kernel, op_index, rhs)?;
                match op {
                    PcuDispatchAluOp::Add
                    | PcuDispatchAluOp::Sub
                    | PcuDispatchAluOp::Mul
                    | PcuDispatchAluOp::Div => {}
                    _ => return Err(PcuSpirvError::UnsupportedInstruction(op.support_flag())),
                }
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: data_index,
                value,
            }) => {
                validate_dataflow_binding(kernel, binding, PcuBindingAccess::WriteOnly)?;
                validate_invocation_index(data_index)?;
                validate_defined_before(kernel, op_index, value)?;
                saw_store = true;
            }
            PcuDispatchOp::Control(PcuDispatchControlOp::Return) => saw_return = true,
            _ => return Err(PcuSpirvError::UnsupportedInstruction(op.support_flag())),
        }
    }

    if saw_return && saw_store {
        validate_f32_map_kernel(kernel).map_err(|error| match error {
            PcuF32MapValidationError::InvalidBinding(_)
            | PcuF32MapValidationError::DuplicateBinding(_) => PcuSpirvError::InvalidBinding,
            PcuF32MapValidationError::InvalidIndex(_) => PcuSpirvError::UnsupportedInstruction(
                PcuDispatchOpCaps::BINDING_LOAD | PcuDispatchOpCaps::BINDING_STORE,
            ),
            PcuF32MapValidationError::UnsupportedOperation(position) => {
                PcuSpirvError::UnsupportedInstruction(kernel.ops[position].support_flag())
            }
            PcuF32MapValidationError::UnsupportedInterface
            | PcuF32MapValidationError::InvalidValue(_)
            | PcuF32MapValidationError::DuplicateValue(_)
            | PcuF32MapValidationError::MissingStore
            | PcuF32MapValidationError::MissingReturn => PcuSpirvError::InvalidKernelSignature,
        })
    } else {
        Err(PcuSpirvError::InvalidKernelSignature)
    }
}

fn validate_grid_stride_kernel(kernel: &PcuDispatchKernelIr<'_>) -> Result<(), PcuSpirvError> {
    let mut loops = kernel.ops.iter().filter_map(|op| match op {
        PcuDispatchOp::GridStrideLoop { extent, body } => Some((*extent, *body)),
        _ => None,
    });
    let Some((extent, body)) = loops.next() else {
        return Err(PcuSpirvError::InvalidKernelSignature);
    };
    if extent == 0 || loops.next().is_some() || kernel.entry.logical_shape[0] == 0 {
        return Err(PcuSpirvError::InvalidKernelSignature);
    }
    if !matches!(
        kernel.ops,
        [
            PcuDispatchOp::GridStrideLoop { .. },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return)
        ]
    ) {
        return Err(PcuSpirvError::InvalidKernelSignature);
    }
    if !kernel
        .bindings
        .iter()
        .copied()
        .all(is_storage_f32_dataflow_binding)
    {
        return Err(PcuSpirvError::InvalidBinding);
    }
    let mut definitions = Vec::new();
    let mut has_store = false;
    for op in body.iter().copied() {
        match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result,
                binding,
                index: PcuDispatchIndex::GridStrideId,
            }) => {
                validate_value_result(result)?;
                validate_dataflow_binding(kernel, binding, PcuBindingAccess::ReadOnly)?;
                if definitions.contains(&result) {
                    return Err(PcuSpirvError::InvalidKernelSignature);
                }
                definitions.push(result);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding,
                index: PcuDispatchIndex::GridStrideId,
                value,
            }) => {
                validate_dataflow_binding(kernel, binding, PcuBindingAccess::WriteOnly)?;
                if !definitions.contains(&value) {
                    return Err(PcuSpirvError::InvalidKernelSignature);
                }
                has_store = true;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result,
                value: PcuParameterValue::F32(_),
            }) => {
                validate_value_result(result)?;
                if definitions.contains(&result) {
                    return Err(PcuSpirvError::InvalidKernelSignature);
                }
                definitions.push(result);
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                result,
                op,
                lhs,
                rhs,
                ..
            }) => {
                validate_value_result(result)?;
                if !matches!(
                    op,
                    PcuDispatchAluOp::Add
                        | PcuDispatchAluOp::Sub
                        | PcuDispatchAluOp::Mul
                        | PcuDispatchAluOp::Div
                ) {
                    return Err(PcuSpirvError::UnsupportedInstruction(op.support_flag()));
                }
                if definitions.contains(&result)
                    || !definitions.contains(&lhs)
                    || !definitions.contains(&rhs)
                {
                    return Err(PcuSpirvError::InvalidKernelSignature);
                }
                definitions.push(result);
            }
            _ => return Err(PcuSpirvError::UnsupportedInstruction(op.support_flag())),
        }
    }
    if has_store {
        Ok(())
    } else {
        Err(PcuSpirvError::InvalidKernelSignature)
    }
}

fn is_storage_f32_dataflow_binding(binding: fusion_pcu::PcuBinding<'_>) -> bool {
    binding.storage == PcuBindingStorageClass::Storage
        && matches!(
            binding.access,
            PcuBindingAccess::ReadOnly | PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
        )
        && binding.binding_type == PcuBindingType::Value(PcuValueType::f32())
}

fn validate_dataflow_binding(
    kernel: &PcuDispatchKernelIr<'_>,
    binding: PcuBindingRef,
    required: PcuBindingAccess,
) -> Result<(), PcuSpirvError> {
    let Some(candidate) = kernel
        .bindings
        .iter()
        .copied()
        .find(|candidate| candidate.set == binding.set && candidate.binding == binding.binding)
    else {
        return Err(PcuSpirvError::InvalidBinding);
    };

    let access_allowed = matches!(
        (candidate.access, required),
        (
            PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite,
            PcuBindingAccess::ReadOnly,
        ) | (
            PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite,
            PcuBindingAccess::WriteOnly,
        )
    );
    if access_allowed && is_storage_f32_dataflow_binding(candidate) {
        Ok(())
    } else {
        Err(PcuSpirvError::InvalidBinding)
    }
}

fn validate_invocation_index(index: PcuDispatchIndex) -> Result<(), PcuSpirvError> {
    match index {
        PcuDispatchIndex::InvocationId => Ok(()),
        PcuDispatchIndex::BindingElementZero => Err(PcuSpirvError::UnsupportedInstruction(
            PcuDispatchOpCaps::BINDING_LOAD | PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO,
        )),
        PcuDispatchIndex::GridStrideId | PcuDispatchIndex::Value(_) => {
            Err(PcuSpirvError::UnsupportedInstruction(
                PcuDispatchOpCaps::BINDING_LOAD | PcuDispatchOpCaps::BINDING_STORE,
            ))
        }
    }
}

const fn validate_value_result(value: PcuDispatchValueId) -> Result<(), PcuSpirvError> {
    if value.0 == 0 {
        Err(PcuSpirvError::InvalidKernelSignature)
    } else {
        Ok(())
    }
}

fn validate_result_slot(
    kernel: &PcuDispatchKernelIr<'_>,
    op_index: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuSpirvError> {
    validate_value_result(value)?;
    if kernel
        .ops
        .iter()
        .take(op_index)
        .copied()
        .any(|op| dataflow_result(op) == Some(value))
    {
        Err(PcuSpirvError::InvalidKernelSignature)
    } else {
        Ok(())
    }
}

fn validate_defined_before(
    kernel: &PcuDispatchKernelIr<'_>,
    op_index: usize,
    value: PcuDispatchValueId,
) -> Result<(), PcuSpirvError> {
    validate_value_result(value)?;
    if kernel
        .ops
        .iter()
        .take(op_index)
        .copied()
        .any(|op| dataflow_result(op) == Some(value))
    {
        Ok(())
    } else {
        Err(PcuSpirvError::InvalidKernelSignature)
    }
}

const fn dataflow_result(op: PcuDispatchOp<'_>) -> Option<PcuDispatchValueId> {
    match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::BindingLoad { result, .. }
            | PcuDispatchDataOp::Constant { result, .. }
            | PcuDispatchDataOp::Alu { result, .. },
        ) => Some(result),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        lower_dispatch_to_spirv,
        validate_invocation_index,
        validate_dispatch_for_spirv,
    };
    use super::super::{
        PcuSpirvCapability,
        PcuSpirvCapabilityCaps,
        PcuSpirvError,
        PcuSpirvFixedSink,
        PcuSpirvLoweringOptions,
        SPIRV_MAGIC,
    };
    use fusion_pcu::{
        PcuAccelerationStructureBindingType,
        PcuAccelerationStructureLevel,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchCoordinateOp,
        PcuDispatchDataOp,
        PcuDispatchIndex,
        PcuDispatchOp,
        PcuDispatchOpCaps,
        PcuDispatchResourceOp,
        PcuDispatchRayTraceOp,
        PcuDispatchValueId,
        PcuDispatchValueOp,
        PcuParameterValue,
        PcuTraceRayOp,
        PcuValueType,
    };
    use fusion_pcu::model::PcuDispatchKernelBuilder;

    #[test]
    fn element_zero_load_reports_its_specific_unsupported_capability() {
        assert_eq!(
            validate_invocation_index(PcuDispatchIndex::BindingElementZero),
            Err(PcuSpirvError::UnsupportedInstruction(
                PcuDispatchOpCaps::BINDING_LOAD.union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
            ))
        );
    }

    #[test]
    fn minimal_dispatch_lowers_to_spirv_header_and_compute_entry() {
        let builder = PcuDispatchKernelBuilder::<1>::new(1, "main", [1, 1, 1])
            .with_control_op(fusion_pcu::PcuDispatchControlOp::Return)
            .expect("test builder should accept return");
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<64>::new();

        let info = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect("minimal return-only dispatch should lower");

        assert_eq!(sink.as_slice()[0], SPIRV_MAGIC);
        assert_eq!(info.bound, 5);
        assert_eq!(info.word_count, sink.len());
        assert!(sink.as_slice().contains(&0x6e69_616d));
        assert_eq!(execution_local_size(sink.as_slice()), [1, 1, 1]);
    }

    #[test]
    fn unsupported_spirv_versions_fail_before_writing_words() {
        let builder = PcuDispatchKernelBuilder::<1>::new(1, "main", [64, 1, 1]);
        let kernel = builder.ir();
        for version in [
            super::super::PcuSpirvVersion(0),
            super::super::PcuSpirvVersion(0x0002_0000),
            super::super::PcuSpirvVersion(0x0001_0700),
            super::super::PcuSpirvVersion(0x0001_0400),
            super::super::PcuSpirvVersion(0x0001_0500),
            super::super::PcuSpirvVersion(0x0001_0600),
            super::super::PcuSpirvVersion(0x0001_0001),
            super::super::PcuSpirvVersion(0x0101_0000),
        ] {
            let mut sink = PcuSpirvFixedSink::<64>::new();
            let result = lower_dispatch_to_spirv(
                &kernel,
                PcuSpirvLoweringOptions::minimal_shader().with_version(version),
                &mut sink,
            );
            assert_eq!(result, Err(PcuSpirvError::UnsupportedVersion(version)));
            assert!(sink.is_empty());
        }
    }

    #[test]
    fn common_f32_profile_rejects_duplicate_binding_addresses() {
        let bindings = [
            PcuBinding::scalar::<f32>(
                Some("first"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
            ),
            PcuBinding::scalar::<f32>(
                Some("duplicate"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadWrite,
            ),
        ];
        let builder = PcuDispatchKernelBuilder::<3>::new(7, "main", [4, 1, 1])
            .with_bindings(&bindings)
            .with_data_op(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(1),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            })
            .expect("constant")
            .with_data_op(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            })
            .expect("store")
            .with_control_op(fusion_pcu::PcuDispatchControlOp::Return)
            .expect("return");
        let kernel = builder.ir();
        assert_eq!(
            validate_dispatch_for_spirv(&kernel, PcuSpirvLoweringOptions::minimal_shader()),
            Err(PcuSpirvError::InvalidBinding)
        );
    }

    #[test]
    fn parallel_float_map_lowers_to_storage_buffer_compute() {
        let bindings = [
            PcuBinding::value(
                Some("input_a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("input_b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let builder = PcuDispatchKernelBuilder::<8>::new(8, "main", [64, 1, 1])
            .with_bindings(&bindings)
            .with_resource_op(PcuDispatchResourceOp::Load)
            .expect("test builder should accept input load")
            .with_resource_op(PcuDispatchResourceOp::Load)
            .expect("test builder should accept input load")
            .with_arithmetic_op(PcuDispatchAluOp::Mul)
            .expect("test builder should accept multiply")
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("test builder should accept add")
            .with_value_op(PcuDispatchValueOp::Constant)
            .expect("test builder should accept constant")
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("test builder should accept add")
            .with_resource_op(PcuDispatchResourceOp::Store)
            .expect("test builder should accept output store")
            .with_control_op(fusion_pcu::PcuDispatchControlOp::Return)
            .expect("test builder should accept return");
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<256>::new();

        let info = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect("parallel float map should lower");

        assert_eq!(sink.as_slice()[0], SPIRV_MAGIC);
        assert_eq!(info.bound, super::super::PARALLEL_FLOAT_BOUND);
        assert_eq!(info.word_count, sink.len());
        assert_eq!(execution_local_size(sink.as_slice()), [1, 1, 1]);
        assert!(
            sink.as_slice()
                .iter()
                .any(|word| (*word & 0xffff) == u32::from(super::super::OP_F_MUL))
        );
        assert!(
            sink.as_slice()
                .iter()
                .any(|word| (*word & 0xffff) == u32::from(super::super::OP_STORE))
        );
    }

    #[test]
    fn operandful_parallel_float_map_lowers_to_storage_buffer_compute() {
        let bindings = [
            PcuBinding::value(
                Some("input_a"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("input_b"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let builder = PcuDispatchKernelBuilder::<9>::new(8, "main", [64, 1, 1])
            .with_bindings(&bindings)
            .with_data_op(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::InvocationId,
            })
            .expect("test builder should accept input load")
            .with_data_op(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(2),
                value: PcuParameterValue::from_f32_bits(0x4000_0000),
            })
            .expect("test builder should accept constant")
            .with_data_op(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Mul,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            })
            .expect("test builder should accept multiply")
            .with_data_op(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(4),
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
            })
            .expect("test builder should accept input load")
            .with_data_op(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(5),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(3),
                rhs: PcuDispatchValueId(4),
            })
            .expect("test builder should accept add")
            .with_data_op(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(6),
                value: PcuParameterValue::from_f32_bits(0x3f80_0000),
            })
            .expect("test builder should accept constant")
            .with_data_op(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(7),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(5),
                rhs: PcuDispatchValueId(6),
            })
            .expect("test builder should accept add")
            .with_data_op(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(7),
            })
            .expect("test builder should accept output store")
            .with_control_op(fusion_pcu::PcuDispatchControlOp::Return)
            .expect("test builder should accept return");
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<256>::new();

        let info = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect("operandful parallel float map should lower");

        assert_eq!(sink.as_slice()[0], SPIRV_MAGIC);
        assert_eq!(info.bound, 38);
        assert_eq!(info.word_count, sink.len());
        assert_eq!(execution_local_size(sink.as_slice()), [1, 1, 1]);
        assert!(
            sink.as_slice()
                .iter()
                .any(|word| (*word & 0xffff) == u32::from(super::super::OP_F_ADD))
        );
    }

    #[test]
    fn grid_stride_map_emits_structured_spirv_loop() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::GridStrideId,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Constant {
                result: PcuDispatchValueId(2),
                value: PcuParameterValue::F32(1.0_f32.to_bits()),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
                value_type: fusion_pcu::PcuValueType::f32(),
                result: PcuDispatchValueId(3),
                op: PcuDispatchAluOp::Add,
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::GridStrideId,
                value: PcuDispatchValueId(3),
            }),
        ];
        let ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 2048,
                body: &body,
            },
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
        ];
        let builder = PcuDispatchKernelBuilder::<2>::new(10, "grid_stride", [250, 1, 1])
            .with_bindings(&bindings)
            .with_ops(&ops)
            .expect("builder should accept grid-stride map");
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<512>::new();

        let info = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect("grid-stride map should lower");

        assert_eq!(info.word_count, sink.len());
        for opcode in [
            super::super::OP_PHI,
            super::super::OP_LOOP_MERGE,
            super::super::OP_BRANCH,
            super::super::OP_BRANCH_CONDITIONAL,
            super::super::OP_I_ADD,
            super::super::OP_U_LESS_THAN,
            super::super::OP_U_LESS_THAN_EQUAL,
        ] {
            assert!(
                sink.as_slice()
                    .iter()
                    .any(|word| (*word & 0xffff) == u32::from(opcode)),
                "missing SPIR-V opcode {opcode}"
            );
        }
    }

    #[test]
    fn typed_f32_builder_program_lowers_to_spirv() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let (builder, input) =
            fusion_pcu::F32MapBuilder::<8>::new(9, "main", [64, 1, 1], &bindings)
                .load_f32(PcuBindingRef::new(0, 0))
                .unwrap();
        let (builder, bias) = builder.constant(0.5).unwrap();
        let (builder, result) = builder.add(input, bias).unwrap();
        let builder = builder.store_f32(PcuBindingRef::new(0, 1), result).unwrap();
        let mut sink = PcuSpirvFixedSink::<256>::new();
        lower_dispatch_to_spirv(
            &builder.ir(),
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .unwrap();
        assert_eq!(sink.as_slice()[0], SPIRV_MAGIC);
    }

    #[test]
    fn arithmetic_op_reports_unsupported_instruction() {
        let builder = PcuDispatchKernelBuilder::<1>::new(2, "main", [1, 1, 1])
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("test builder should accept add");
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<64>::new();

        let error = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect_err("add lowering is not implemented yet");

        assert_eq!(
            error,
            PcuSpirvError::UnsupportedInstruction(PcuDispatchOpCaps::ALU_ADD)
        );
        assert!(sink.is_empty());
    }

    #[test]
    fn ray_trace_requires_capability_before_instruction_lowering() {
        let acceleration_structure = PcuBinding::acceleration_structure(
            Some("scene"),
            0,
            0,
            PcuBindingAccess::ReadOnly,
            PcuAccelerationStructureBindingType {
                level: PcuAccelerationStructureLevel::TopLevel,
                mutable: false,
            },
        );
        let bindings = [acceleration_structure];
        let trace = PcuTraceRayOp::new(acceleration_structure.reference());
        let builder = PcuDispatchKernelBuilder::<1>::new(3, "main", [1, 1, 1])
            .with_bindings(&bindings)
            .with_ray_trace_op(PcuDispatchRayTraceOp::TraceRay(trace))
            .expect("test builder should accept trace op");
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<64>::new();

        let error = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect_err("ray trace capability is not enabled");

        assert_eq!(
            error,
            PcuSpirvError::UnsupportedCapability(PcuSpirvCapability::RayTracing)
        );
        assert!(sink.is_empty());
    }

    #[test]
    fn ray_trace_capability_still_reports_unimplemented_instruction() {
        let acceleration_structure = PcuBinding::acceleration_structure(
            Some("scene"),
            0,
            0,
            PcuBindingAccess::ReadOnly,
            PcuAccelerationStructureBindingType {
                level: PcuAccelerationStructureLevel::TopLevel,
                mutable: false,
            },
        );
        let bindings = [acceleration_structure];
        let trace = PcuTraceRayOp::new(acceleration_structure.reference());
        let builder = PcuDispatchKernelBuilder::<1>::new(4, "main", [1, 1, 1])
            .with_bindings(&bindings)
            .with_ray_trace_op(PcuDispatchRayTraceOp::TraceRay(trace))
            .expect("test builder should accept trace op");
        let kernel = builder.ir();
        let options = PcuSpirvLoweringOptions::minimal_shader().with_capabilities(
            PcuSpirvCapabilityCaps::SHADER | PcuSpirvCapabilityCaps::RAY_TRACING,
        );

        let error = validate_dispatch_for_spirv(&kernel, options)
            .expect_err("ray trace lowering is not implemented yet");

        assert_eq!(
            error,
            PcuSpirvError::UnsupportedInstruction(PcuDispatchOpCaps::RAY_TRACE)
        );
    }

    #[test]
    fn unsupported_value_type_is_reported() {
        let parameter = [fusion_pcu::PcuParameter {
            slot: fusion_pcu::PcuParameterSlot(0),
            name: Some("wide"),
            value_type: PcuValueType::f64(),
        }];
        let builder =
            PcuDispatchKernelBuilder::<1>::new(5, "main", [1, 1, 1]).with_parameters(&parameter);
        let kernel = builder.ir();

        let error = validate_dispatch_for_spirv(&kernel, PcuSpirvLoweringOptions::minimal_shader())
            .expect_err("f64 is not in the first SPIR-V lowering subset");

        assert_eq!(
            error,
            PcuSpirvError::UnsupportedValueType(PcuValueType::f64())
        );
    }

    #[test]
    fn sink_capacity_failure_is_reported() {
        let builder = PcuDispatchKernelBuilder::<1>::new(6, "main", [1, 1, 1]);
        let kernel = builder.ir();
        let mut sink = PcuSpirvFixedSink::<4>::new();

        let error = lower_dispatch_to_spirv(
            &kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .expect_err("four words cannot hold a SPIR-V header");

        assert_eq!(error, PcuSpirvError::SinkFull);
    }

    #[test]
    fn derivative_coordinate_op_requires_derivative_capability() {
        let builder = PcuDispatchKernelBuilder::<1>::new(7, "main", [1, 1, 1])
            .with_coordinate_op(PcuDispatchCoordinateOp::DerivativeX)
            .expect("test builder should accept derivative op");
        let kernel = builder.ir();

        let error = validate_dispatch_for_spirv(&kernel, PcuSpirvLoweringOptions::minimal_shader())
            .expect_err("derivative capability is not enabled");

        assert_eq!(
            error,
            PcuSpirvError::UnsupportedCapability(PcuSpirvCapability::CoordinateDerivative)
        );
    }

    fn execution_local_size(words: &[u32]) -> [u32; 3] {
        let mut offset = 5;
        while offset < words.len() {
            let instruction = words[offset];
            let word_count = (instruction >> 16) as usize;
            let opcode = u16::try_from(instruction & u32::from(u16::MAX))
                .expect("mask limits the opcode to 16 bits");
            assert!(word_count > 0, "malformed SPIR-V instruction");
            if opcode == super::super::OP_EXECUTION_MODE
                && words[offset + 2] == super::super::EXECUTION_MODE_LOCAL_SIZE
            {
                return [words[offset + 3], words[offset + 4], words[offset + 5]];
            }
            offset += word_count;
        }
        panic!("module has no LocalSize execution mode");
    }
}
