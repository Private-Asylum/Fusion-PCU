//! Backend-neutral validation helpers for PCU model payloads.

use crate::{
    PcuAccelerationStructureLevel,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuImageDimension,
    PcuParameterSlot,
    PcuInvocationBindings,
    PcuInvocationParameters,
    PcuPortDirection,
    PcuPortRate,
    PcuSampleOp,
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
    PcuTraceRayOp,
    PcuValueType,
};
use crate::model::{
    PcuCommandEffectKind,
    PcuCommandKernelIr,
    PcuCommandOp,
    PcuCommandPredicate,
    PcuCommandResultId,
    PcuOperand,
    PcuTarget,
};

/// Contract failures surfaced when a typed command result flow is malformed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCommandValidationError {
    DuplicateResultId(PcuCommandResultId),
    ResultUsedBeforeDefinition(PcuCommandResultId),
    LegacyPreviousResultUnverified,
    NonScalarReadResult(PcuValueType),
    ReadWidthMismatch {
        step: usize,
        expected: u8,
        found: u8,
    },
    ReadTargetTypeMismatch {
        expected: PcuValueType,
        found: PcuValueType,
    },
    ReadTargetNotReadable(PcuBindingRef),
    ReadTargetHasNoValueType(PcuBindingRef),
    MissingReadTargetBinding(PcuBindingRef),
    UndeclaredReadPort,
    OpaqueReadTarget,
    ReadEffectMismatch {
        step: usize,
        expected: PcuCommandEffectKind,
        found: PcuCommandEffectKind,
    },
}

/// Validates typed results produced by ordered command reads and their subsequent uses.
///
/// Result IDs are local to one command kernel. A result is visible only after its defining
/// `ReadResult` step, and typed reads must state a scalar type whose bit width matches the
/// declared access width. Targets with declared binding or port types are checked against that
/// type; named and intrinsic targets remain intentionally opaque to this backend-neutral layer.
pub fn validate_command_kernel(
    kernel: &PcuCommandKernelIr<'_>,
) -> Result<(), PcuCommandValidationError> {
    for (index, step) in kernel.steps.iter().enumerate() {
        if let PcuCommandOp::ReadResult {
            target,
            result,
            width_bits,
            effect,
        } = step.op
        {
            let expected_effect = PcuCommandEffectKind::Read;
            if effect != expected_effect {
                return Err(PcuCommandValidationError::ReadEffectMismatch {
                    step: index,
                    expected: expected_effect,
                    found: effect,
                });
            }
            let PcuValueType::Scalar(scalar) = result.value_type else {
                return Err(PcuCommandValidationError::NonScalarReadResult(
                    result.value_type,
                ));
            };
            let expected_width = scalar.bit_width();
            if width_bits != expected_width {
                return Err(PcuCommandValidationError::ReadWidthMismatch {
                    step: index,
                    expected: expected_width,
                    found: width_bits,
                });
            }
            match target {
                PcuTarget::Binding(reference) => {
                    let Some(binding) = find_binding(kernel.bindings, reference) else {
                        return Err(PcuCommandValidationError::MissingReadTargetBinding(
                            reference,
                        ));
                    };
                    if matches!(binding.access, PcuBindingAccess::WriteOnly) {
                        return Err(PcuCommandValidationError::ReadTargetNotReadable(reference));
                    }
                    let Some(target_type) = binding.value_type() else {
                        return Err(PcuCommandValidationError::ReadTargetHasNoValueType(
                            reference,
                        ));
                    };
                    if target_type != result.value_type {
                        return Err(PcuCommandValidationError::ReadTargetTypeMismatch {
                            expected: target_type,
                            found: result.value_type,
                        });
                    }
                }
                PcuTarget::Port(name) => {
                    let Some(port) = kernel.ports.iter().find(|port| port.name == Some(name))
                    else {
                        return Err(PcuCommandValidationError::UndeclaredReadPort);
                    };
                    if port.direction != PcuPortDirection::Input {
                        return Err(PcuCommandValidationError::UndeclaredReadPort);
                    }
                    if port.value_type != result.value_type {
                        return Err(PcuCommandValidationError::ReadTargetTypeMismatch {
                            expected: port.value_type,
                            found: result.value_type,
                        });
                    }
                }
                PcuTarget::Named(_) | PcuTarget::Intrinsic(_) => {
                    return Err(PcuCommandValidationError::OpaqueReadTarget);
                }
            }
            if kernel.steps[..index].iter().any(|prior| {
                matches!(prior.op, PcuCommandOp::ReadResult { result: prior_result, .. }
                    if prior_result.id == result.id)
            }) {
                return Err(PcuCommandValidationError::DuplicateResultId(result.id));
            }
        }

        match step.op {
            PcuCommandOp::Write { value, .. } | PcuCommandOp::Modify { value, .. } => {
                validate_command_result_use(kernel, index, value)?;
            }
            PcuCommandOp::Invoke { args, .. } => {
                for argument in args.iter().copied() {
                    validate_command_result_use(kernel, index, argument)?;
                }
            }
            PcuCommandOp::Await { predicate } => match predicate {
                PcuCommandPredicate::Equals { left, right } => {
                    validate_command_result_use(kernel, index, left)?;
                    validate_command_result_use(kernel, index, right)?;
                }
                PcuCommandPredicate::NonZero(value) => {
                    validate_command_result_use(kernel, index, value)?;
                }
                PcuCommandPredicate::Ready(_) | PcuCommandPredicate::Named(_) => {}
            },
            PcuCommandOp::Return { value: Some(value) } => {
                validate_command_result_use(kernel, index, value)?;
            }
            PcuCommandOp::Read { .. }
            | PcuCommandOp::ReadResult { .. }
            | PcuCommandOp::Copy { .. }
            | PcuCommandOp::Stall { .. }
            | PcuCommandOp::Sleep { .. }
            | PcuCommandOp::Barrier
            | PcuCommandOp::Return { value: None } => {}
        }
    }
    Ok(())
}

fn validate_command_result_use(
    kernel: &PcuCommandKernelIr<'_>,
    index: usize,
    operand: PcuOperand<'_>,
) -> Result<(), PcuCommandValidationError> {
    match operand {
        PcuOperand::Result(id)
            if !kernel.steps[..index].iter().any(|prior| {
                matches!(prior.op, PcuCommandOp::ReadResult { result, .. } if result.id == id)
            }) => return Err(PcuCommandValidationError::ResultUsedBeforeDefinition(id)),
        PcuOperand::PreviousResult => {
            return Err(PcuCommandValidationError::LegacyPreviousResultUnverified);
        }
        _ => {}
    }
    Ok(())
}

/// Contract failures surfaced when one sample op does not actually match the binding graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSampleValidationError {
    MissingImageBinding(PcuBindingRef),
    MissingSamplerBinding(PcuBindingRef),
    ImageBindingIsNotImage(PcuBindingRef),
    SamplerBindingIsNotSampler(PcuBindingRef),
    ImageBindingNotReadable(PcuBindingRef),
    SamplerBindingNotReadable(PcuBindingRef),
    ResultTypeMismatch {
        expected: PcuValueType,
        found: PcuValueType,
    },
    CoordinateTypeMismatch {
        dimension: PcuImageDimension,
        arrayed: bool,
        found: PcuValueType,
    },
    OffsetComponentMismatch {
        expected: u8,
        found: u8,
    },
}

/// Validates that one sample op targets one readable image binding and one sampler binding.
///
/// # Errors
///
/// Returns the first contract mismatch that makes the operation dishonest.
pub fn validate_sample_op(
    sample: PcuSampleOp,
    bindings: &[PcuBinding<'_>],
) -> Result<(), PcuSampleValidationError> {
    let image = find_binding(bindings, sample.image)
        .ok_or(PcuSampleValidationError::MissingImageBinding(sample.image))?;
    let sampler = find_binding(bindings, sample.sampler).ok_or(
        PcuSampleValidationError::MissingSamplerBinding(sample.sampler),
    )?;
    let Some(image_type) = image.image_type() else {
        return Err(PcuSampleValidationError::ImageBindingIsNotImage(
            sample.image,
        ));
    };
    if matches!(image.access, PcuBindingAccess::WriteOnly) {
        return Err(PcuSampleValidationError::ImageBindingNotReadable(
            sample.image,
        ));
    }
    if sampler.sampler_type().is_none() {
        return Err(PcuSampleValidationError::SamplerBindingIsNotSampler(
            sample.sampler,
        ));
    }
    if !matches!(sampler.access, PcuBindingAccess::ReadOnly) {
        return Err(PcuSampleValidationError::SamplerBindingNotReadable(
            sample.sampler,
        ));
    }
    if sample.result_type != image_type.texel_type {
        return Err(PcuSampleValidationError::ResultTypeMismatch {
            expected: image_type.texel_type,
            found: sample.result_type,
        });
    }

    let required_lanes = image_type.dimension.coordinate_lanes();
    let Some(actual_lanes) = sample.coordinates.linear_lanes() else {
        return Err(PcuSampleValidationError::CoordinateTypeMismatch {
            dimension: image_type.dimension,
            arrayed: image_type.arrayed,
            found: sample.coordinates,
        });
    };
    let coordinate_lanes_are_valid = actual_lanes == required_lanes
        || (image_type.arrayed && actual_lanes == required_lanes + 1);
    if !coordinate_lanes_are_valid {
        return Err(PcuSampleValidationError::CoordinateTypeMismatch {
            dimension: image_type.dimension,
            arrayed: image_type.arrayed,
            found: sample.coordinates,
        });
    }
    if sample.offset_components > 0 && sample.offset_components != required_lanes {
        return Err(PcuSampleValidationError::OffsetComponentMismatch {
            expected: required_lanes,
            found: sample.offset_components,
        });
    }

    Ok(())
}

/// Contract failures surfaced when one ray trace op does not match the binding graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuTraceRayValidationError {
    MissingAccelerationStructureBinding(PcuBindingRef),
    BindingIsNotAccelerationStructure(PcuBindingRef),
    AccelerationStructureNotReadable(PcuBindingRef),
    AccelerationStructureIsNotTraceable(PcuBindingRef),
    ZeroMaxRecursionDepth,
}

/// Validates that one trace op targets one readable top-level acceleration-structure binding.
///
/// # Errors
///
/// Returns the first contract mismatch that makes the trace operation dishonest.
pub fn validate_trace_ray_op(
    trace: PcuTraceRayOp,
    bindings: &[PcuBinding<'_>],
) -> Result<(), PcuTraceRayValidationError> {
    if trace.max_recursion_depth == 0 {
        return Err(PcuTraceRayValidationError::ZeroMaxRecursionDepth);
    }

    let acceleration_structure = find_binding(bindings, trace.acceleration_structure).ok_or(
        PcuTraceRayValidationError::MissingAccelerationStructureBinding(
            trace.acceleration_structure,
        ),
    )?;
    let Some(acceleration_structure_type) = acceleration_structure.acceleration_structure_type()
    else {
        return Err(
            PcuTraceRayValidationError::BindingIsNotAccelerationStructure(
                trace.acceleration_structure,
            ),
        );
    };
    if matches!(acceleration_structure.access, PcuBindingAccess::WriteOnly) {
        return Err(
            PcuTraceRayValidationError::AccelerationStructureNotReadable(
                trace.acceleration_structure,
            ),
        );
    }
    if matches!(
        acceleration_structure_type.level,
        PcuAccelerationStructureLevel::BottomLevel
    ) {
        return Err(
            PcuTraceRayValidationError::AccelerationStructureIsNotTraceable(
                trace.acceleration_structure,
            ),
        );
    }

    Ok(())
}

/// Contract failures surfaced when one stream kernel is not an honest simple transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuStreamSimpleTransformValidationError {
    InvalidPortCount,
    InvalidPortShape,
    UnsupportedValueType(PcuValueType),
    MismatchedValueTypes {
        input: PcuValueType,
        output: PcuValueType,
    },
    DuplicateParameterSlot(PcuParameterSlot),
    ParameterTypeMismatch {
        slot: PcuParameterSlot,
        expected: PcuValueType,
        found: PcuValueType,
    },
    UnsupportedPattern {
        pattern: PcuStreamPattern,
        value_type: PcuStreamValueType,
    },
}

/// Precise rejection reasons for the common RP2350 PIO U32 stream profile.
///
/// The profile intentionally admits exactly one operation per program, with one U32 stream
/// input and output and no resources or runtime parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPioU32StreamProfileError {
    InvalidPortCount,
    InvalidPortShape,
    InvalidValueType(PcuValueType),
    KernelBindingsPresent,
    ParametersPresent,
    InvalidPatternCount { found: usize },
    UnsupportedPattern(PcuStreamPattern),
    InvalidPattern(PcuStreamPattern),
    RuntimeBindingsPresent,
    RuntimeParametersPresent,
}

/// Validates one program against the common one-pattern U32 subset shared with RP2350 PIO.
///
/// Port order is significant: port zero is a stream input and port one a stream output. The
/// accepted operation patterns are bit reverse, bit invert, increment, decrement, logical shifts,
/// bit extraction, low-bit mask, and U32 byte swap. Parameterized arithmetic and resource-backed
/// programs are outside this profile.
pub fn validate_pio_u32_stream_profile(
    kernel: &PcuStreamKernelIr<'_>,
) -> Result<(), PcuPioU32StreamProfileError> {
    let [input, output] = kernel.ports else {
        return Err(PcuPioU32StreamProfileError::InvalidPortCount);
    };
    if input.direction != PcuPortDirection::Input
        || output.direction != PcuPortDirection::Output
        || input.rate != PcuPortRate::Stream
        || output.rate != PcuPortRate::Stream
    {
        return Err(PcuPioU32StreamProfileError::InvalidPortShape);
    }
    for value_type in [input.value_type, output.value_type] {
        if value_type != PcuValueType::u32() {
            return Err(PcuPioU32StreamProfileError::InvalidValueType(value_type));
        }
    }
    if !kernel.bindings.is_empty() {
        return Err(PcuPioU32StreamProfileError::KernelBindingsPresent);
    }
    if !kernel.parameters.is_empty() {
        return Err(PcuPioU32StreamProfileError::ParametersPresent);
    }
    let [pattern] = kernel.patterns else {
        return Err(PcuPioU32StreamProfileError::InvalidPatternCount {
            found: kernel.patterns.len(),
        });
    };
    let pattern = *pattern;
    let supported = match pattern {
        PcuStreamPattern::BitReverse
        | PcuStreamPattern::BitInvert
        | PcuStreamPattern::Increment
        | PcuStreamPattern::Decrement
        | PcuStreamPattern::ByteSwap32 => true,
        PcuStreamPattern::ShiftLeft { bits } | PcuStreamPattern::ShiftRight { bits } => {
            (1..=32).contains(&bits)
        }
        PcuStreamPattern::ExtractBits { offset, width } => {
            width >= 1 && offset < 32 && offset as u16 + width as u16 <= 32
        }
        PcuStreamPattern::MaskLower { bits } => (1..=32).contains(&bits),
        PcuStreamPattern::AddParameter { .. } | PcuStreamPattern::XorParameter { .. } => false,
    };
    if !supported {
        return if matches!(
            pattern,
            PcuStreamPattern::AddParameter { .. } | PcuStreamPattern::XorParameter { .. }
        ) {
            Err(PcuPioU32StreamProfileError::UnsupportedPattern(pattern))
        } else {
            Err(PcuPioU32StreamProfileError::InvalidPattern(pattern))
        };
    }
    Ok(())
}

/// Validates both the static program shape and the empty runtime binding tables required by the
/// common RP2350 PIO U32 profile.
pub fn validate_pio_u32_stream_invocation(
    kernel: &PcuStreamKernelIr<'_>,
    bindings: PcuInvocationBindings<'_>,
    parameters: PcuInvocationParameters<'_>,
) -> Result<(), PcuPioU32StreamProfileError> {
    validate_pio_u32_stream_profile(kernel)?;
    if !bindings.is_empty() {
        return Err(PcuPioU32StreamProfileError::RuntimeBindingsPresent);
    }
    if !parameters.bindings.is_empty() {
        return Err(PcuPioU32StreamProfileError::RuntimeParametersPresent);
    }
    Ok(())
}

/// A shared input/output example for the common PIO U32 profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuPioU32StreamVector {
    pub pattern: PcuStreamPattern,
    pub input: u32,
    pub expected: u32,
}

/// Small deterministic vectors suitable for CPU and hardware conformance checks.
pub const PCU_PIO_U32_STREAM_VECTORS: &[PcuPioU32StreamVector] = &[
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::BitReverse,
        input: 0x0000_0001,
        expected: 0x8000_0000,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::BitInvert,
        input: 0x00ff_00ff,
        expected: 0xff00_ff00,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::Increment,
        input: u32::MAX,
        expected: 0,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::Decrement,
        input: 0,
        expected: u32::MAX,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::ShiftLeft { bits: 3 },
        input: 0x8000_0003,
        expected: 0x0000_0018,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::ShiftRight { bits: 4 },
        input: 0x8000_003f,
        expected: 0x0800_0003,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::ShiftLeft { bits: 32 },
        input: 0x1234_5678,
        expected: 0,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::ExtractBits {
            offset: 8,
            width: 8,
        },
        input: 0x1234_56ab,
        expected: 0x56,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::MaskLower { bits: 12 },
        input: 0xabcd_1234,
        expected: 0x234,
    },
    PcuPioU32StreamVector {
        pattern: PcuStreamPattern::ByteSwap32,
        input: 0x1234_56ab,
        expected: 0xab56_3412,
    },
];

/// Executes one admitted U32 profile pattern as a CPU reference operation.
pub fn execute_pio_u32_stream_reference(pattern: PcuStreamPattern, input: u32) -> Option<u32> {
    Some(match pattern {
        PcuStreamPattern::BitReverse => input.reverse_bits(),
        PcuStreamPattern::BitInvert => !input,
        PcuStreamPattern::Increment => input.wrapping_add(1),
        PcuStreamPattern::Decrement => input.wrapping_sub(1),
        PcuStreamPattern::ShiftLeft { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                0
            } else {
                input << bits
            }
        }
        PcuStreamPattern::ShiftRight { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                0
            } else {
                input >> bits
            }
        }
        PcuStreamPattern::ExtractBits { offset, width }
            if width >= 1 && offset < 32 && offset as u16 + width as u16 <= 32 =>
        {
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1_u32 << width) - 1
            };
            (input >> offset) & mask
        }
        PcuStreamPattern::MaskLower { bits } if (1..=32).contains(&bits) => {
            if bits == 32 {
                input
            } else {
                input & ((1_u32 << bits) - 1)
            }
        }
        PcuStreamPattern::ByteSwap32 => input.swap_bytes(),
        _ => return None,
    })
}

/// Validates one stream kernel as a simple typed unary transform and returns the element type.
///
/// # Errors
///
/// Returns the first contract mismatch that makes the stream transform dishonest.
pub fn validate_stream_simple_transform(
    kernel: &PcuStreamKernelIr<'_>,
) -> Result<PcuStreamValueType, PcuStreamSimpleTransformValidationError> {
    let [input, output] = kernel.ports else {
        return Err(PcuStreamSimpleTransformValidationError::InvalidPortCount);
    };
    if input.direction != PcuPortDirection::Input
        || output.direction != PcuPortDirection::Output
        || input.rate != PcuPortRate::Stream
        || output.rate != PcuPortRate::Stream
    {
        return Err(PcuStreamSimpleTransformValidationError::InvalidPortShape);
    }

    let input_type = PcuStreamValueType::from_value_type(input.value_type)
        .ok_or(PcuStreamSimpleTransformValidationError::UnsupportedValueType(input.value_type))?;
    let output_type = PcuStreamValueType::from_value_type(output.value_type)
        .ok_or(PcuStreamSimpleTransformValidationError::UnsupportedValueType(output.value_type))?;
    if input_type != output_type {
        return Err(
            PcuStreamSimpleTransformValidationError::MismatchedValueTypes {
                input: input.value_type,
                output: output.value_type,
            },
        );
    }

    for (index, parameter) in kernel.parameters.iter().enumerate() {
        if kernel.parameters[..index]
            .iter()
            .any(|existing| existing.slot == parameter.slot)
        {
            return Err(
                PcuStreamSimpleTransformValidationError::DuplicateParameterSlot(parameter.slot),
            );
        }
    }

    for pattern in kernel.patterns.iter().copied() {
        match pattern {
            PcuStreamPattern::AddParameter { parameter }
            | PcuStreamPattern::XorParameter { parameter } => {
                let Some(declared) = kernel
                    .parameters
                    .iter()
                    .copied()
                    .find(|candidate| candidate.slot == parameter)
                else {
                    return Err(
                        PcuStreamSimpleTransformValidationError::ParameterTypeMismatch {
                            slot: parameter,
                            expected: input_type.as_value_type(),
                            found: PcuValueType::bool(),
                        },
                    );
                };
                if declared.value_type != input_type.as_value_type() {
                    return Err(
                        PcuStreamSimpleTransformValidationError::ParameterTypeMismatch {
                            slot: parameter,
                            expected: input_type.as_value_type(),
                            found: declared.value_type,
                        },
                    );
                }
            }
            _ => {}
        }

        if !pattern.supports_value_type(input_type) {
            return Err(
                PcuStreamSimpleTransformValidationError::UnsupportedPattern {
                    pattern,
                    value_type: input_type,
                },
            );
        }
    }

    Ok(input_type)
}

fn find_binding<'a>(
    bindings: &'a [PcuBinding<'a>],
    reference: PcuBindingRef,
) -> Option<PcuBinding<'a>> {
    bindings
        .iter()
        .copied()
        .find(|binding| binding.reference() == reference)
}

#[cfg(test)]
mod pio_u32_profile_tests {
    use super::*;
    use crate::model::PcuStreamKernelBuilder;

    #[test]
    fn shared_vectors_match_cpu_reference() {
        for vector in PCU_PIO_U32_STREAM_VECTORS {
            assert_eq!(
                execute_pio_u32_stream_reference(vector.pattern, vector.input),
                Some(vector.expected),
                "pattern {:?} input {:#010x}",
                vector.pattern,
                vector.input,
            );
            let builder = PcuStreamKernelBuilder::<1>::words(99, "conformance")
                .with_pattern(vector.pattern)
                .expect("one pattern fits");
            assert_eq!(validate_pio_u32_stream_profile(&builder.ir()), Ok(()));
        }
    }

    #[test]
    fn admits_one_u32_pattern_without_runtime_state() {
        let builder = PcuStreamKernelBuilder::<1>::words(1, "profile")
            .increment()
            .expect("one pattern fits");
        let kernel = builder.ir();
        assert_eq!(validate_pio_u32_stream_profile(&kernel), Ok(()));
        assert_eq!(
            validate_pio_u32_stream_invocation(
                &kernel,
                PcuInvocationBindings::empty(),
                PcuInvocationParameters::empty(),
            ),
            Ok(())
        );
    }

    #[test]
    fn profile_rejects_multiple_patterns_parameters_and_non_u32() {
        let multiple_builder = PcuStreamKernelBuilder::<2>::words(2, "multiple")
            .increment()
            .expect("pattern fits")
            .decrement()
            .expect("pattern fits");
        let multiple = multiple_builder.ir();
        assert_eq!(
            validate_pio_u32_stream_profile(&multiple),
            Err(PcuPioU32StreamProfileError::InvalidPatternCount { found: 2 })
        );

        let parameter =
            crate::PcuParameter::named(PcuParameterSlot(0), "delta", PcuValueType::u32());
        let parameterized_builder = PcuStreamKernelBuilder::<1>::words(3, "parameterized")
            .with_parameters(core::slice::from_ref(&parameter))
            .with_pattern(PcuStreamPattern::AddParameter {
                parameter: PcuParameterSlot(0),
            })
            .expect("pattern fits");
        let parameterized = parameterized_builder.ir();
        assert_eq!(
            validate_pio_u32_stream_profile(&parameterized),
            Err(PcuPioU32StreamProfileError::ParametersPresent)
        );

        let non_u32_builder = PcuStreamKernelBuilder::<1>::half_words(4, "u16")
            .increment()
            .expect("pattern fits");
        let non_u32 = non_u32_builder.ir();
        assert_eq!(
            validate_pio_u32_stream_profile(&non_u32),
            Err(PcuPioU32StreamProfileError::InvalidValueType(
                PcuValueType::u16()
            ))
        );
    }
}

#[cfg(test)]
mod command_result_tests {
    use super::*;
    use crate::model::{
        PcuCommandResult,
        PcuCommandStep,
    };

    fn typed_read(id: u16, value_type: PcuValueType, width_bits: u8) -> PcuCommandOp<'static> {
        PcuCommandOp::ReadResult {
            target: PcuTarget::Port("status"),
            result: PcuCommandResult {
                id: PcuCommandResultId(id),
                value_type,
            },
            width_bits,
            effect: PcuCommandEffectKind::Read,
        }
    }

    fn validate_steps(steps: &[PcuCommandStep<'_>]) -> Result<(), PcuCommandValidationError> {
        let ports = &[crate::PcuPort::stream_input(
            Some("status"),
            PcuValueType::u32(),
        )];
        let kernel = PcuCommandKernelIr {
            id: crate::PcuKernelId(1),
            entry_point: "command",
            bindings: &[],
            ports,
            parameters: &[],
            steps,
        };
        validate_command_kernel(&kernel)
    }

    fn validate_steps_with_bindings(
        steps: &[PcuCommandStep<'_>],
        bindings: &[PcuBinding<'_>],
    ) -> Result<(), PcuCommandValidationError> {
        let kernel = PcuCommandKernelIr {
            id: crate::PcuKernelId(1),
            entry_point: "command",
            bindings,
            ports: &[],
            parameters: &[],
            steps,
        };
        validate_command_kernel(&kernel)
    }

    #[test]
    fn validates_ordered_typed_read_return_flow() {
        let steps = [
            PcuCommandStep {
                name: Some("read"),
                op: typed_read(3, PcuValueType::u32(), 32),
            },
            PcuCommandStep {
                name: Some("return"),
                op: PcuCommandOp::Return {
                    value: Some(PcuOperand::Result(PcuCommandResultId(3))),
                },
            },
        ];
        assert_eq!(validate_steps(&steps), Ok(()));
    }

    #[test]
    fn rejects_undefined_and_duplicate_result_ids() {
        let undefined = [PcuCommandStep {
            name: Some("return"),
            op: PcuCommandOp::Return {
                value: Some(PcuOperand::Result(PcuCommandResultId(3))),
            },
        }];
        assert_eq!(
            validate_steps(&undefined),
            Err(PcuCommandValidationError::ResultUsedBeforeDefinition(
                PcuCommandResultId(3)
            ))
        );

        let duplicate = [
            PcuCommandStep {
                name: None,
                op: typed_read(3, PcuValueType::u32(), 32),
            },
            PcuCommandStep {
                name: None,
                op: typed_read(3, PcuValueType::u32(), 32),
            },
        ];
        assert_eq!(
            validate_steps(&duplicate),
            Err(PcuCommandValidationError::DuplicateResultId(
                PcuCommandResultId(3)
            ))
        );
    }

    #[test]
    fn rejects_wrong_width_type_access_and_opaque_target() {
        let wrong_width = [PcuCommandStep {
            name: Some("read"),
            op: typed_read(3, PcuValueType::u32(), 16),
        }];
        assert_eq!(
            validate_steps(&wrong_width),
            Err(PcuCommandValidationError::ReadWidthMismatch {
                step: 0,
                expected: 32,
                found: 16
            })
        );

        let wrong_type = [PcuCommandStep {
            name: Some("read"),
            op: typed_read(3, PcuValueType::u16(), 16),
        }];
        assert_eq!(
            validate_steps(&wrong_type),
            Err(PcuCommandValidationError::ReadTargetTypeMismatch {
                expected: PcuValueType::u32(),
                found: PcuValueType::u16()
            })
        );

        let opaque = [PcuCommandStep {
            name: Some("read"),
            op: PcuCommandOp::ReadResult {
                target: PcuTarget::Named("opaque-register"),
                result: PcuCommandResult {
                    id: PcuCommandResultId(3),
                    value_type: PcuValueType::u32(),
                },
                width_bits: 32,
                effect: PcuCommandEffectKind::Read,
            },
        }];
        assert_eq!(
            validate_steps(&opaque),
            Err(PcuCommandValidationError::OpaqueReadTarget)
        );

        let write_only = [crate::PcuBinding::value(
            Some("status"),
            0,
            0,
            crate::PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::u32(),
        )];
        let binding_read = [PcuCommandStep {
            name: Some("read"),
            op: PcuCommandOp::ReadResult {
                target: PcuTarget::Binding(PcuBindingRef::new(0, 0)),
                result: PcuCommandResult {
                    id: PcuCommandResultId(3),
                    value_type: PcuValueType::u32(),
                },
                width_bits: 32,
                effect: PcuCommandEffectKind::Read,
            },
        }];
        assert_eq!(
            validate_steps_with_bindings(&binding_read, &write_only),
            Err(PcuCommandValidationError::ReadTargetNotReadable(
                PcuBindingRef::new(0, 0)
            ))
        );

        let wrong_effect = [PcuCommandStep {
            name: Some("read"),
            op: PcuCommandOp::ReadResult {
                target: PcuTarget::Port("status"),
                result: PcuCommandResult {
                    id: PcuCommandResultId(3),
                    value_type: PcuValueType::u32(),
                },
                width_bits: 32,
                effect: PcuCommandEffectKind::Write,
            },
        }];
        assert_eq!(
            validate_steps(&wrong_effect),
            Err(PcuCommandValidationError::ReadEffectMismatch {
                step: 0,
                expected: PcuCommandEffectKind::Read,
                found: PcuCommandEffectKind::Write,
            })
        );

        let previous_result = [PcuCommandStep {
            name: Some("return"),
            op: PcuCommandOp::Return {
                value: Some(PcuOperand::PreviousResult),
            },
        }];
        assert_eq!(
            validate_steps(&previous_result),
            Err(PcuCommandValidationError::LegacyPreviousResultUnverified)
        );
    }
}
