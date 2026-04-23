//! Backend-neutral validation helpers for PCU model payloads.

use crate::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuImageDimension,
    PcuParameterSlot,
    PcuPortDirection,
    PcuPortRate,
    PcuSampleOp,
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
    PcuValueType,
};

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
