//! Exact integer joint division eligibility; not a provider conformance certificate.
//!
//! Quotient truncates toward zero; signed remainder has the dividend's sign. Zero
//! divisor and signed MIN/-1 report fatal faults, with neither output published.
//! Lowest logical invocation selects the fault, independently of physical arrival.
//! The two result prefixes commit only after complete terminal domain validation;
//! tails remain intact. These finite-width integer rules have no IEEE floating
//! rounding, contraction, precision or tininess dependence. Strict and Boundary
//! therefore share this one-operation law without weakening either Result delivery.
//!
//! Admission requires independently proved values, joint rollback, fault ordering,
//! actual read spans and physical resource safety. This descriptor enables no
//! provider, implementation, CPU fallback, wrapping, total or Clamp division.

#[rustfmt::skip]
use crate::{
    assess_checked_integer_div_rem_operands,
    CheckedIntegerDivRemOperandSchema,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
use super::PcuPortableV1IntegerMapError as Error;

/// Eligible checked joint division over one named integer width.
///
/// Provider conformance and executable admission remain separate obligations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuPortableV1IntegerDivRemMapDescription {
    pub scalar: PcuScalarType,
    pub submitted_invocations: u32,
    pub logical_extent: u32,
    /// Actual unique reads, dividend/divisor roles, spans and quotient/remainder outputs.
    pub operands: CheckedIntegerDivRemOperandSchema,
}

/// Describes one exact checked quotient/remainder map over all fourteen integer widths.
///
/// Repeated resources/SSA, unread declarations, declaration/store permutation and
/// independent zero-index reads use the shared detached resource schema. Explicit
/// load obligations survive even when the loaded SSA value has no mathematical
/// consumer. Integer-irrelevant numerical options remain in the request envelope;
/// they cannot license undefined behavior or hide joint domain faults.
///
/// # Errors
/// Rejects a missing Portable request, noninteger type, malformed resource/SSA
/// structure, zero or multidimensional geometry, unsupported flags and Clamp.
/// No discovery, preparation, allocation or backend execution occurs.
pub fn describe_portable_v1_integer_div_rem_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuPortableV1IntegerDivRemMapDescription, Error> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::PortableV1
    {
        return Err(Error::NotRequested);
    }
    let [submitted_invocations, 1, 1] = kernel.entry.logical_shape else {
        return Err(Error::InvalidLogicalShape(kernel.entry.logical_shape));
    };
    if submitted_invocations == 0 {
        return Err(Error::InvalidLogicalShape(kernel.entry.logical_shape));
    }
    let (body, logical_extent) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { body, extent },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] if *extent != 0 => (*body, *extent),
        [
            body @ ..,
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] => (body, submitted_invocations),
        _ => return Err(Error::UnsupportedStructure),
    };
    let [
        _,
        _,
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(scalar),
            ..
        }),
        _,
        _,
    ] = body
    else {
        return Err(Error::UnsupportedStructure);
    };
    let value_type = PcuValueType::Scalar(*scalar);
    if !crate::map_validation::typed_dispatch::is_supported_checked_integer(value_type) {
        return Err(Error::UnsupportedScalar);
    }
    let operands = assess_checked_integer_div_rem_operands(
        kernel,
        value_type,
        PcuValueTypeCaps::for_scalar(*scalar),
    )
    .map_err(Error::InvalidOperands)?;
    Ok(PcuPortableV1IntegerDivRemMapDescription {
        scalar: *scalar,
        submitted_invocations,
        logical_extent,
        operands,
    })
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
