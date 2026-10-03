//! Cold structural description for homogeneous representation-preserving maps.
//!
//! Transport copies encodings, including nonfinite floating payloads and signed
//! zeros. It does not perform arithmetic, rounding, conversion or numeric input
//! validation. This separate profile leaves checked-map arithmetic requirements
//! and existing exact identity/broadcast implementations unchanged.

#[path = "resources/resources.rs"]
mod resources;
#[path = "validation/validation.rs"]
mod validation;

#[rustfmt::skip]
pub use resources::{
    PcuScalarTransportDescription,
    PcuScalarTransportResource,
};
#[rustfmt::skip]
use crate::{
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuScalarType,
    PcuTypedDispatchValidationError,
    PcuValueType,
};

/// Structural or caller-capacity refusal; never a runtime completion status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuScalarTransportError {
    UnsupportedInterface,
    InvalidLogicalShape,
    UnsupportedRequirements,
    InvalidBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnsupportedOperation(usize),
    InvalidIndex(usize),
    MissingLoadOrStore,
    MissingReturn,
    InvalidValues(PcuTypedDispatchValidationError),
    InsufficientCapacity {
        capacity: usize,
        required_at_least: usize,
    },
}

/// Describes ordered scalar loads/stores using 256 cold SSA metadata slots.
///
/// Every scalar identity is structurally eligible, including the 22 sealed Rust
/// carriers. Eligibility does not supply a packed representation for Bool/I4/U4
/// or executable provider support for any type. Providers must independently
/// preserve the complete encoding: native floating canonicalization, FTZ and
/// arithmetic substitutions are not legal copies.
///
/// Only homogeneous storage bindings, indexed or element-zero loads, indexed
/// stores, and direct/one canonical grid-stride region are admitted. Unused
/// declarations retain validation and original access; actual resources alone
/// occupy the caller-chosen `CAPACITY`. There is no universal binding limit.
/// Use [`describe_scalar_transport_map_with_scratch`] for sparse/larger SSA IDs.
///
/// The original numerical request is retained, not rewritten to a Portable
/// profile. This description grants no numerical conformance, layout, physical
/// aliasing, snapshot, synchronization, lifetime or publication rights. It is
/// detached preparation metadata; warm paths need not revisit source IR.
///
/// # Errors
/// Returns the first interface, access, type-flow, indexing or capacity refusal.
pub fn describe_scalar_transport_map<const CAPACITY: usize>(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
) -> Result<PcuScalarTransportDescription<CAPACITY>, PcuScalarTransportError> {
    let mut scratch = [None; 256];
    describe_scalar_transport_map_with_scratch(kernel, scalar, &mut scratch)
}

/// Describes the same transport profile with caller-owned cold type-flow scratch.
///
/// Scratch is cleared by the existing typed SSA verifier before validating the
/// accepted region. It is indexed by original SSA IDs, which are not renumbered.
/// No allocation or backend/device interaction occurs. All declared bindings
/// must match `scalar`, including unread declarations; actual resource spans
/// distinguish view capacity from original contents needed before ordered stores.
/// Cross-index read-zero/write-index dependencies remain conservative and must
/// be rejected or safely snapshotted by an independently admitted provider.
///
/// # Errors
/// Returns structural refusal, insufficient SSA scratch or resource capacity.
pub fn describe_scalar_transport_map_with_scratch<const CAPACITY: usize>(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
    scratch: &mut [Option<PcuValueType>],
) -> Result<PcuScalarTransportDescription<CAPACITY>, PcuScalarTransportError> {
    let (body, logical_extent) = validation::validate(kernel, scalar, scratch)?;
    resources::project(kernel, scalar, body, logical_extent)
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
