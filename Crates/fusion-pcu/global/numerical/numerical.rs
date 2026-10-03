//! Cold invocation eligibility shared by host and resident-affinity preparation.
//!
//! A neutral descriptor is a necessary structural condition, never provider
//! conformance. Every provider still matches the original request and admits
//! only its independently qualified executable profile. Cached warm calls do
//! not enter this module, rescore devices or revalidate numerical requirements.

#[rustfmt::skip]
use crate::{
    describe_portable_v1_integer_div_rem_map,
    describe_portable_v1_integer_map,
    describe_portable_v1_map,
    describe_portable_v1_unary_map,
    PcuDispatchKernelIr,
    PcuReproducibility,
};
use super::PcuExecutionError;

pub(super) fn validate_invocation_contract(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<(), PcuExecutionError> {
    let options = kernel.numerical_requirements.numerical_options;
    if options.reproducibility == PcuReproducibility::Unspecified
        || describe_portable_v1_map(kernel).is_ok()
        || describe_portable_v1_integer_map(kernel).is_ok()
        || describe_portable_v1_integer_div_rem_map(kernel).is_ok()
        || describe_portable_v1_unary_map(kernel).is_ok()
    {
        return Ok(());
    }
    Err(PcuExecutionError::UnsupportedNumericalOptions(options))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
