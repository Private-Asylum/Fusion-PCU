//! Validation for stream kernel transforms.

use crate::{
    PcuParameterSlot,
    PcuPortDirection,
    PcuPortRate,
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
    PcuValueType,
};

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
