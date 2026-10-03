//! Checked floating resource admission uses the shared scalar projection.
#[rustfmt::skip]
use super::{
    validate_checked_float_map_kernel,
    CheckedFloatMapValidationError,
};
#[rustfmt::skip]
use crate::{
    CheckedScalarMapResourceError,
    CheckedScalarMapResourceSchema,
    PcuDispatchKernelIr,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Projects actual resources of a homogeneous composed checked floating map.
///
/// Reuses the existing six-format validator, including complete declaration,
/// typed SSA, operation, capability, geometry and access checks. Direct maps use
/// their invocation count; grid-stride maps use their semantic visited extent.
/// Each actual load/store contributes its independently required span. Repeated
/// accesses merge by binding reference; declarations with no load/store vanish
/// from the projection without losing their metadata validation.
/// Original content requirements are projected separately from view capacity.
/// In this flat body, an indexed store supplies later same-binding indexed
/// loads for its logical lane; earlier loads still require original contents.
/// Element-zero reads retain their original span when multiple logical lanes
/// could write that binding. No general control-flow dominance is inferred.
///
/// Per-instruction range/underflow policies remain authoritative; this helper
/// does not require them to equal the function header or normalize them. A
/// provider unable to implement a scoped policy must reject separately. The
/// result is not a Portable conformance token, physical layout plan, alias proof,
/// fault-attribution law, output-publication rule or new executable offer.
/// A resource read at zero and written across lanes can have cross-lane hazards;
/// rejecting it or providing a proved input snapshot remains the provider's job.
/// Omitted writable declarations do not become readonly frontend arguments.
/// Preparation may retain the detached result; warm paths need no IR scans.
///
/// # Errors
/// Rejects any existing checked-map validation failure or more actual unique
/// resources than `CAPACITY`. No partial schema, runtime or allocation escapes.
pub fn assess_checked_float_map_resources<const CAPACITY: usize>(
    kernel: &PcuDispatchKernelIr<'_>,
    value_type: PcuValueType,
    scalar_caps: PcuValueTypeCaps,
) -> Result<
    CheckedScalarMapResourceSchema<CAPACITY>,
    CheckedScalarMapResourceError<CheckedFloatMapValidationError>,
> {
    use CheckedScalarMapResourceError as Error;
    validate_checked_float_map_kernel(kernel, value_type, scalar_caps)
        .map_err(Error::InvalidMap)?;
    crate::map_validation::checked_scalar_resources::project(
        kernel,
        value_type,
        CheckedFloatMapValidationError::InvalidBinding,
    )
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
