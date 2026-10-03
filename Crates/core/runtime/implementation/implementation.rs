//! Cold, operation-specific implementation offers and cost evidence.
//!
//! Providers admit concrete requests; consumers choose among the admitted offers. This module
//! defines neither backend preference nor a universal performance score. Querying and selection
//! belong to preparation. A prepared executor retains the chosen implementation and performs no
//! offer enumeration, calibration or ranking on its warm path.
//!
//! An offer is not an allocation import permission. Even equal physical device identities do not
//! establish compatible backing, layout, external handles or synchronization. Region planners
//! need separately proved transitions and their costs before composing different providers.

#[rustfmt::skip]
use crate::{
    PcuDeviceIdentity,
    PcuExecutorId,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuRangePolicy,
};

/// Exact numerical requirements for which an operation implementation is admitted.
///
/// This is one admitted combination, not a promise that every combination of individually
/// advertised options works. Operation-specific dtype, shape, layout and resource requirements
/// remain in [`PcuImplementationRequest::operation`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct PcuImplementationRequirements {
    pub numerical_mode: PcuNumericalMode,
    pub numerical_options: PcuNumericalOptions,
    pub float_underflow: PcuFloatUnderflowPolicy,
    pub range_policy: PcuRangePolicy,
}

impl PcuImplementationRequirements {
    /// Canonical checked boundary requirements, also usable in constant IR declarations.
    pub const DEFAULT: Self = Self {
        numerical_mode: PcuNumericalMode::Boundary,
        numerical_options: PcuNumericalOptions {
            compound_arithmetic: crate::PcuCompoundArithmeticPolicy::Checked,
            precision: crate::PcuPrecisionPolicy::Preserve,
            reproducibility: crate::PcuReproducibility::Unspecified,
        },
        float_underflow: PcuFloatUnderflowPolicy::IeeeAfterRounding,
        range_policy: PcuRangePolicy::Reject,
    };
}

/// Concrete identity of an implementation within a discovered device snapshot.
///
/// `local_id` names a provider-defined kernel, library algorithm or delegated implementation.
/// `revision` changes when its relevant implementation/configuration changes. The device
/// includes its provider and discovery generation. This identity is only part of a prepared
/// cache key: the complete operation, shapes/layouts, policies, runtime configuration and
/// resource constraints still participate. Equal identities never imply resource interop.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcuImplementationId {
    pub device: PcuDeviceIdentity,
    pub executor: PcuExecutorId,
    pub local_id: u32,
    pub revision: u64,
}

/// Broad implementation mechanism, independent of vendor or relative speed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcuImplementationMechanism {
    NativeKernel,
    NativeLibrary,
    DelegatedRuntime,
    Interpreter,
}

/// Provenance of a cost estimate for this exact operation/device/policy scope.
///
/// IDs are provider-local evidence revisions, not universal calibration identities. Measured
/// evidence must retain its method and conditions outside the execution path. Estimates are
/// ranking hints; they neither admit semantics nor guarantee a future duration.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcuCostProvenance {
    Model { revision: u64 },
    Measured { calibration: u64 },
}

/// A known duration estimate. Unknown is represented by `None`, never invented zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcuDurationEstimate {
    pub nanoseconds: u64,
    pub provenance: PcuCostProvenance,
}

/// The resource/publication boundary covered by execution cost evidence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcuCostBoundary {
    /// Inputs and outputs remain resident; required device completion is included.
    Resident,
    /// Ordinary host inputs through terminal host output publication, including staging.
    Host,
    /// All selected inputs begin in host memory; the terminal output remains resident.
    ///
    /// Includes required input staging, execution, completion and publication of the escaped
    /// resident owner. It excludes later explicit host readback and owner destruction. This
    /// differs from both a fully resident operation and a full host-output boundary.
    HostInputsResidentOutput,
    /// Selected inputs mix host memory and already-resident resources; output remains resident.
    ///
    /// Includes the required host-input staging, execution, completion and escaped-owner
    /// publication. Resident inputs retain their required leases/affinity; this boundary is
    /// not permission to import or migrate them. Later readback/destruction is excluded.
    /// Cost evidence must additionally identify which inputs were staged and their extents.
    MixedInputsResidentOutput,
}

/// Cold cost facts for one concrete admitted operation and resource boundary.
///
/// `completion` is elapsed time from submission start through terminal completion/publication
/// for `boundary`; it already includes host submission. Do not add those two durations.
/// `device_execution`, when known, is separately measured device work, not subtraction of host
/// wall timings. Preparation is separate. None of these fields includes an unproved interop
/// transition; region planners must price their separately admitted transitions explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcuImplementationCost {
    pub boundary: PcuCostBoundary,
    pub preparation: Option<PcuDurationEstimate>,
    pub host_submission: Option<PcuDurationEstimate>,
    pub completion: Option<PcuDurationEstimate>,
    pub device_execution: Option<PcuDurationEstimate>,
}

impl PcuImplementationCost {
    /// Describes the requested boundary without inventing performance evidence.
    #[must_use]
    pub const fn unknown(boundary: PcuCostBoundary) -> Self {
        Self {
            boundary,
            preparation: None,
            host_submission: None,
            completion: None,
            device_execution: None,
        }
    }
}

/// Backend-neutral envelope around a borrowed operation-specific cold request.
///
/// Providers implement [`PcuImplementationOffers`] for their request representation, for
/// example borrowed dispatch IR or a tensor graph region. Numerical requirements, device
/// and executor are explicit; providers must also validate all operation/resource constraints.
pub struct PcuImplementationRequest<'a, O: ?Sized> {
    pub device: PcuDeviceIdentity,
    pub executor: PcuExecutorId,
    pub requirements: PcuImplementationRequirements,
    pub boundary: PcuCostBoundary,
    pub operation: &'a O,
}

impl<'a, 'ir> PcuImplementationRequest<'a, crate::PcuDispatchKernelIr<'ir>> {
    /// Borrows the complete dispatch contract without reconstructing its numerical tuple.
    ///
    /// This is a cold descriptive request, not device admission or an executable handle.
    #[must_use]
    pub const fn for_dispatch(
        device: PcuDeviceIdentity,
        executor: PcuExecutorId,
        boundary: PcuCostBoundary,
        operation: &'a crate::PcuDispatchKernelIr<'ir>,
    ) -> Self {
        Self {
            device,
            executor,
            requirements: operation.numerical_requirements,
            boundary,
            operation,
        }
    }
}

/// An implementation admitted for the entire concrete request supplied to its query.
///
/// Unknown workspace differs from zero bytes. Workspace excludes ordinary input/output
/// storage and must be included in allocation budgets during preparation. This descriptive
/// record owns no executable or resource lease; actual preparation revalidates the request,
/// acquires storage and retains the selected concrete implementation until terminal quiescence.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PcuImplementationOffer {
    pub implementation: PcuImplementationId,
    pub kind: PcuImplementationMechanism,
    pub requirements: PcuImplementationRequirements,
    pub workspace_bytes: Option<u64>,
    pub cost: PcuImplementationCost,
}

/// Why a retained offer does not belong to the current request envelope.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PcuImplementationOfferMismatch {
    Device,
    Executor,
    Requirements,
    Boundary,
}

impl PcuImplementationOffer {
    /// Checks the request envelope before selection or preparation.
    ///
    /// This rejects stale/different targets and mismatched policies or cost boundaries. It
    /// cannot prove operation equivalence or interop: concrete preparation must validate the
    /// full operation and resources even after this succeeds.
    ///
    /// # Errors
    /// Returns the first mismatched envelope field in device/executor/requirements/boundary order.
    pub fn validate_request<O: ?Sized>(
        &self,
        request: &PcuImplementationRequest<'_, O>,
    ) -> Result<(), PcuImplementationOfferMismatch> {
        if self.implementation.device != request.device {
            return Err(PcuImplementationOfferMismatch::Device);
        }
        if self.implementation.executor != request.executor {
            return Err(PcuImplementationOfferMismatch::Executor);
        }
        if self.requirements != request.requirements {
            return Err(PcuImplementationOfferMismatch::Requirements);
        }
        if self.cost.boundary != request.boundary {
            return Err(PcuImplementationOfferMismatch::Boundary);
        }
        Ok(())
    }
}

/// Operation-specific cold offer discovery with bounded caller-owned result storage.
///
/// Implementations return the total admitted offer count and fill the available prefix with
/// `Some(offer)`, even when the output is too small. Unsupported valid work has zero offers;
/// invalid/stale requests and provider failures retain the backend's structured error. Queries
/// must not execute the requested workload or publish its outputs. Cold provider inspection or
/// library heuristics may be necessary and must not migrate into warm submission.
///
/// Every returned offer admits the complete request, including its operation-specific types,
/// layouts and resource/policy constraints. Costs may be unknown. Consumers retain explicit
/// selection and ranking; this trait has no dynamic-dispatch requirement.
pub trait PcuImplementationOffers<O: ?Sized> {
    type Error;

    /// Enumerates implementations for this concrete request, without executing its workload.
    ///
    /// # Errors
    /// Returns the provider error for invalid references/requests or inspection failures.
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, O>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error>;
}

#[cfg(test)]
mod tests;
