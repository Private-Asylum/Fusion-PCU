//! Requested Portable unary eligibility delegates to the neutral checked matcher.
//!
//! The request gate is separate from exact load/unary/store structure and actual
//! resource roles. Neither helper is a provider conformance or execution token.

#[rustfmt::skip]
use crate::{
    describe_checked_float_unary_map,
    CheckedFloatMapValidationError,
    PcuBindingRef,
    PcuCheckedFloatUnaryMapDescription,
    PcuCheckedFloatUnaryMapError,
    PcuDispatchKernelIr,
    PcuReproducibility,
};

/// Structural eligibility refusal, distinct from runtime/backend failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuPortableV1UnaryMapError {
    NotRequested,
    UnsupportedScalar,
    InvalidLogicalShape([u32; 3]),
    UnsupportedStructure,
    RangeMismatch,
    UnderflowMismatch,
    InvalidAccess(PcuBindingRef),
    InvalidMap(CheckedFloatMapValidationError),
}

impl From<PcuCheckedFloatUnaryMapError> for PcuPortableV1UnaryMapError {
    fn from(error: PcuCheckedFloatUnaryMapError) -> Self {
        match error {
            PcuCheckedFloatUnaryMapError::UnsupportedScalar => Self::UnsupportedScalar,
            PcuCheckedFloatUnaryMapError::InvalidLogicalShape(shape) => {
                Self::InvalidLogicalShape(shape)
            }
            PcuCheckedFloatUnaryMapError::UnsupportedStructure => Self::UnsupportedStructure,
            PcuCheckedFloatUnaryMapError::RangeMismatch => Self::RangeMismatch,
            PcuCheckedFloatUnaryMapError::UnderflowMismatch => Self::UnderflowMismatch,
            PcuCheckedFloatUnaryMapError::InvalidAccess(binding) => Self::InvalidAccess(binding),
            PcuCheckedFloatUnaryMapError::InvalidMap(error) => Self::InvalidMap(error),
        }
    }
}

/// Describes a requested Portable Neg/`ReLU` map over the six checked formats.
///
/// The neutral checked matcher retains the exact original header, actual input
/// and output references, unused declarations, geometry and typed SSA rules.
/// A successful match never enables a backend or expands a composed profile.
///
/// # Errors
/// Rejects absent Portable requests and every neutral checked-map failure.
pub fn describe_portable_v1_unary_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuCheckedFloatUnaryMapDescription, PcuPortableV1UnaryMapError> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::PortableV1
    {
        return Err(PcuPortableV1UnaryMapError::NotRequested);
    }
    describe_checked_float_unary_map(kernel).map_err(Into::into)
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
