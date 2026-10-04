//! Exact integer composition eligibility; provider conformance is separate.
#[rustfmt::skip]
use crate::{
    assess_checked_integer_map_resources,
    CheckedIntegerMapValidationError,
    CheckedScalarMapResourceError,
    CheckedScalarMapResourceSchema,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuReproducibility,
    PcuValueType,
    PcuValueTypeCaps,
};

/// Structural eligibility failure, never an execution capability or device fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuPortableV1IntegerComposedMapError {
    NotRequested,
    UnsupportedScalar,
    UnsupportedStructure,
    InvalidResources(CheckedScalarMapResourceError<CheckedIntegerMapValidationError>),
    /// The checked instruction differs from the original header disposition.
    RangeMismatch(usize),
    /// Multiple logical invocations could read a location another invocation writes.
    CrossIndexDependency(PcuBindingRef),
}

/// Describe homogeneous exact checked integer Add/Sub/Mul composition.
///
/// Fourteen sealed integer widths use mathematical results at their named width.
/// Reject publishes no output after a fatal fault; Clamp retains a useful result
/// and reports the earliest recovered fault. Every executed checked instruction,
/// including dead results and overwritten stores, remains observable. Faults are
/// ordered by logical invocation and then authored instruction order, independent
/// of device scheduling. A later fatal fault supersedes recovered notices.
///
/// Direct and one canonical grid-stride body, scalar constants, independent
/// indexed/element-zero reads, and same-invocation store/reload use the shared
/// typed resource rules. Cross-index read/write dependencies are refused rather
/// than granting an implicit snapshot. All checked instructions must match the
/// header range policy. This bounded contract grants no unchecked/wrapping ALU,
/// division, float operation, tensor, arbitrary control flow or parameter ABI.
/// Other numerical axes remain exactly as requested; they cannot weaken explicit
/// checked integer instructions. `CAPACITY` is a caller-chosen representation
/// bound, not a universal IR limit.
///
/// Providers must independently prove exact payloads, faults, private publication,
/// alias/extent safety and terminal resource lifetime before admitting this
/// description. Success never enables CPU fallback or any backend automatically.
///
/// # Errors
/// Refuses an absent Portable request, malformed typed resources/SSA, unequal
/// instruction/header range policies or cross-invocation dependencies.
pub fn describe_portable_v1_checked_integer_composed_map<const CAPACITY: usize>(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<CheckedScalarMapResourceSchema<CAPACITY>, PcuPortableV1IntegerComposedMapError> {
    use PcuPortableV1IntegerComposedMapError as Error;
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::PortableV1
    {
        return Err(Error::NotRequested);
    }
    // Public borrowed IR can contain cyclic grid bodies. Reject nesting before
    // any shared type/feature scanner or derived hash can recursively inspect it.
    let body = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, extent },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] if *extent != 0 => *body,
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => body,
        _ => return Err(Error::UnsupportedStructure),
    };
    if body
        .iter()
        .any(|operation| matches!(operation, PcuDispatchOp::GridStrideLoop { .. }))
    {
        return Err(Error::UnsupportedStructure);
    }
    let value_type = kernel
        .bindings
        .first()
        .and_then(|binding| match binding.binding_type {
            PcuBindingType::Value(value_type @ PcuValueType::Scalar(_)) => Some(value_type),
            _ => None,
        })
        .ok_or(Error::UnsupportedScalar)?;
    let PcuValueType::Scalar(scalar) = value_type else {
        return Err(Error::UnsupportedScalar);
    };
    let resources = assess_checked_integer_map_resources::<CAPACITY>(
        kernel,
        value_type,
        PcuValueTypeCaps::for_scalar(scalar),
    )
    .map_err(Error::InvalidResources)?;
    for (ordinal, instruction) in body.iter().enumerate() {
        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { range_policy, .. }) =
            instruction
            && *range_policy != kernel.numerical_requirements.range_policy
        {
            return Err(Error::RangeMismatch(ordinal));
        }
    }
    for resource in resources.resources() {
        if resource.has_cross_index_read_write() {
            return Err(Error::CrossIndexDependency(resource.binding));
        }
    }
    Ok(resources)
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
