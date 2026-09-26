//! Validation for graphics sampling and ray tracing operations.

use super::find_binding;
use crate::{
    PcuAccelerationStructureLevel,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuImageDimension,
    PcuSampleOp,
    PcuTraceRayOp,
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
