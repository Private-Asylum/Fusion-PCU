//! PCU dispatch to SPIR-V lowering entry points.

use crate::{
    PcuBindingAccess,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchFeatureCaps,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchOpCaps,
    PcuDispatchRayTraceOp,
    PcuDispatchResourceOp,
    PcuDispatchValueOp,
    PcuScalarType,
    PcuValueType,
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
    if is_parallel_float_map_kernel(kernel) {
        return writer.emit_parallel_float_map_module(
            kernel.entry.name,
            kernel.entry.logical_shape,
            options,
        );
    }
    writer.emit_minimal_compute_module(kernel.entry.name, kernel.entry.logical_shape, options)
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
    validate_kernel_signature(kernel, options)?;
    validate_dispatch_features(kernel.required_feature_support())?;

    if is_parallel_float_map_kernel(kernel) {
        return Ok(());
    }

    for op in kernel.ops.iter().copied() {
        validate_op_for_spirv(op, kernel, options)?;
    }

    Ok(())
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

fn validate_dispatch_features(features: PcuDispatchFeatureCaps) -> Result<(), PcuSpirvError> {
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
        PcuDispatchOp::Control(_) => unsupported(op),
        PcuDispatchOp::Resource(resource) => validate_resource_op(resource, options),
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
        PcuDispatchOp::Value(_)
        | PcuDispatchOp::Arithmetic(_)
        | PcuDispatchOp::Port(_)
        | PcuDispatchOp::Sync(_)
        | PcuDispatchOp::Intrinsic { .. } => unsupported(op),
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

fn validate_scalar_type(scalar: PcuScalarType) -> Result<(), PcuSpirvError> {
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

fn require_capability(
    options: PcuSpirvLoweringOptions,
    capability: PcuSpirvCapability,
) -> Result<(), PcuSpirvError> {
    if options.capabilities.supports(capability) {
        Ok(())
    } else {
        Err(PcuSpirvError::UnsupportedCapability(capability))
    }
}

fn unsupported(op: PcuDispatchOp<'_>) -> Result<(), PcuSpirvError> {
    Err(PcuSpirvError::UnsupportedInstruction(op.support_flag()))
}

fn is_parallel_float_map_kernel(kernel: &PcuDispatchKernelIr<'_>) -> bool {
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
    binding: crate::PcuBinding<'_>,
    set: u32,
    slot: u32,
    access: PcuBindingAccess,
) -> bool {
    binding.access == access && is_storage_f32_binding_type(binding, set, slot)
}

fn is_storage_f32_binding_type(binding: crate::PcuBinding<'_>, set: u32, slot: u32) -> bool {
    binding.set == set
        && binding.binding == slot
        && binding.storage == PcuBindingStorageClass::Storage
        && binding.binding_type == PcuBindingType::Value(PcuValueType::f32())
}

fn has_parallel_float_ops(kernel: &PcuDispatchKernelIr<'_>) -> bool {
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

#[cfg(test)]
mod tests {
    use super::{
        lower_dispatch_to_spirv,
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
    use crate::{
        PcuAccelerationStructureBindingType,
        PcuAccelerationStructureLevel,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingStorageClass,
        PcuDispatchAluOp,
        PcuDispatchCoordinateOp,
        PcuDispatchOpCaps,
        PcuDispatchResourceOp,
        PcuDispatchRayTraceOp,
        PcuDispatchValueOp,
        PcuTraceRayOp,
        PcuValueType,
    };
    use crate::model::PcuDispatchKernelBuilder;

    #[test]
    fn minimal_dispatch_lowers_to_spirv_header_and_compute_entry() {
        let builder = PcuDispatchKernelBuilder::<1>::new(1, "main", [1, 1, 1])
            .with_control_op(crate::PcuDispatchControlOp::Return)
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
            .with_control_op(crate::PcuDispatchControlOp::Return)
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
        let parameter = [crate::PcuParameter {
            slot: crate::PcuParameterSlot(0),
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
}
