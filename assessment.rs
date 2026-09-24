//! Backend-neutral Dispatch assessment and preparation contracts.
//!
//! Selection remains a consumer decision: a request names its device and executor explicitly.
//! This layer performs common admission checks, compares the request with the selected executor's
//! surfaced capability floor and limits, then lets the backend create an owned prepared object.
//! Capability masks here are the current PCU vocabulary, not a complete inventory of every
//! possible backend property. Backends should expose additional constraints as structured
//! diagnostics through their own error type.

use crate::{
    PcuDispatchFeatureCaps,
    PcuDispatchOpCaps,
    PcuDispatchPolicyCaps,
    PcuDispatchSubmission,
    PcuError,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuInvocationParameters,
    PcuKernel,
    PcuOwnedCompletion,
    PcuKernelIrContract,
    PcuValueTypeCaps,
    PcuBaseContract,
    PcuDeviceIdentity,
};
use crate::dispatch::{
    validate_dispatch_submission,
    validate_parameters,
};

/// Extra minimum capabilities requested by the consumer for one Dispatch operation.
///
/// The fields extend the capabilities implied by the kernel. They describe only current
/// PCU-discoverable dispatch dimensions; they are not a complete semantic capability system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchCapabilityFloor {
    pub value_types: PcuValueTypeCaps,
    pub features: PcuDispatchFeatureCaps,
    pub instructions: PcuDispatchOpCaps,
    pub policy: PcuDispatchPolicyCaps,
}

impl PcuDispatchCapabilityFloor {
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            value_types: PcuValueTypeCaps::empty(),
            features: PcuDispatchFeatureCaps::empty(),
            instructions: PcuDispatchOpCaps::empty(),
            policy: PcuDispatchPolicyCaps::empty(),
        }
    }
}

impl Default for PcuDispatchCapabilityFloor {
    fn default() -> Self {
        Self::empty()
    }
}

/// Explicit target and workload to assess before backend preparation.
#[derive(Debug, Clone, Copy)]
pub struct PcuDispatchPreparationRequest<'a> {
    pub device: PcuDeviceIdentity,
    pub executor: PcuExecutorId,
    pub submission: PcuDispatchSubmission<'a>,
    pub parameters: PcuInvocationParameters<'a>,
    pub capability_floor: PcuDispatchCapabilityFloor,
    /// Reject assessment when any limit used by this contract is unknown.
    pub require_known_limits: bool,
}

/// Dispatch limit fields currently represented by the common assessment vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchLimits {
    /// Maximum total logical threads in one PCU submission.
    pub max_logical_threads: Option<u32>,
    pub max_bindings: Option<u32>,
    pub max_parameters: Option<u32>,
}

/// One common backend limit that can be exceeded or left unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDispatchLimitKind {
    LogicalThreads,
    Bindings,
    Parameters,
}

/// Structured reason a request did not pass common assessment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDispatchAssessmentIssue {
    Invalid(PcuError),
    WrongDevice,
    UnknownExecutor(PcuExecutorId),
    KernelRequirementsUnsupported,
    ValueTypeFloorUnsupported,
    FeatureFloorUnsupported,
    InstructionFloorUnsupported,
    PolicyFloorUnsupported,
    LimitUnknown(PcuDispatchLimitKind),
    LimitExceeded {
        kind: PcuDispatchLimitKind,
        requested: u32,
        maximum: u32,
    },
    CountOverflow(PcuDispatchLimitKind),
}

/// Assessment failure before backend-specific preparation begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchAssessmentError {
    pub device: PcuDeviceIdentity,
    pub executor: PcuExecutorId,
    pub issue: PcuDispatchAssessmentIssue,
}

/// Lifecycle surface of one prepared Dispatch object.
///
/// Preparation may allocate backend-side descriptors, compiled code, or other resources, but it
/// must not begin the requested work. A prepared object can be submitted once, cancelled, or
/// dropped. Dropping it must release preparation resources safely. Submission consumes the
/// prepared object. An error must either return a retryable prepared object after proving no work
/// was enqueued, or return an owned completion that retains resources while launch/completion
/// status is uncertain.
pub trait PcuPreparedDispatch {
    type Completion: PcuOwnedCompletion;
    type Error;

    fn device(&self) -> PcuDeviceIdentity;
    fn executor(&self) -> PcuExecutorId;

    fn submit(
        self,
    ) -> Result<Self::Completion, PcuPreparedDispatchSubmitError<Self, Self::Completion, Self::Error>>
    where
        Self: Sized;

    fn cancel(self) -> Result<(), PcuPreparedDispatchCancelError<Self, Self::Error>>
    where
        Self: Sized;
}

/// Failed submission with an explicit safe-retry or in-flight ownership outcome.
#[derive(Debug)]
pub enum PcuPreparedDispatchSubmitError<P, C, E> {
    /// Backend proved no work was enqueued, so retry or cancellation is safe.
    Retryable { prepared: P, error: E },
    /// Work may have started; completion retains resources until quiescence is proven.
    InFlight { completion: C, error: E },
}

/// Failed prepared cancellation that retains the object for a later cleanup attempt.
///
/// The object must keep any resources whose release could be unsafe until a later cancellation
/// attempt proves that releasing them is safe.
#[derive(Debug)]
pub struct PcuPreparedDispatchCancelError<P, E> {
    pub prepared: P,
    pub error: E,
}

/// Combined result of common assessment and backend preparation.
#[derive(Debug)]
pub enum PcuDispatchPreparationError<E> {
    Assessment(PcuDispatchAssessmentError),
    Backend(E),
}

/// Backend contract for assessing and preparing an explicitly selected Dispatch target.
pub trait PcuDispatchPreparationBackend: PcuBaseContract {
    type Prepared: PcuPreparedDispatch;
    type Error;

    /// The device identity bound to this backend session.
    fn device_identity(&self) -> PcuDeviceIdentity;

    /// Limits for this executor. `None` means no limit record or field was surfaced; if the
    /// request requires known limits, the first unknown diagnostic is `LogicalThreads` for a missing
    /// record, otherwise it names the missing field.
    fn dispatch_limits(&self, executor: PcuExecutorId) -> Option<PcuDispatchLimits>;

    /// Creates a backend-owned prepared operation after common validation succeeds.
    ///
    /// The request borrows kernel IR and parameters only for this call. The prepared object must
    /// snapshot or finish consuming everything it needs before this method returns. Since
    /// availability may change between assessment and preparation, the backend must recheck any
    /// mutable native constraints here before reserving resources.
    fn prepare_dispatch_direct(
        &mut self,
        request: PcuDispatchPreparationRequest<'_>,
    ) -> Result<Self::Prepared, Self::Error>;

    /// Assesses the explicit target and common requirements without preparing backend state.
    fn assess_dispatch(
        &self,
        request: PcuDispatchPreparationRequest<'_>,
    ) -> Result<(), PcuDispatchAssessmentError> {
        let reject = |issue| PcuDispatchAssessmentError {
            device: request.device,
            executor: request.executor,
            issue,
        };

        if request.device != self.device_identity() {
            return Err(reject(PcuDispatchAssessmentIssue::WrongDevice));
        }
        validate_dispatch_submission(request.submission)
            .map_err(|error| reject(PcuDispatchAssessmentIssue::Invalid(error)))?;
        validate_parameters(request.submission.kernel.signature(), request.parameters)
            .map_err(|error| reject(PcuDispatchAssessmentIssue::Invalid(error)))?;

        let descriptor = self.executor(request.executor).ok_or_else(|| {
            reject(PcuDispatchAssessmentIssue::UnknownExecutor(
                request.executor,
            ))
        })?;
        let kernel = PcuKernel::Dispatch(*request.submission.kernel);
        if !descriptor.support.supports_kernel_direct(kernel) {
            return Err(reject(
                PcuDispatchAssessmentIssue::KernelRequirementsUnsupported,
            ));
        }
        assess_floor(descriptor, request).map_err(reject)?;
        assess_limits(self.dispatch_limits(request.executor), request).map_err(reject)?;

        Ok(())
    }

    /// Assesses the target, then creates a backend-owned prepared operation.
    fn prepare_dispatch(
        &mut self,
        request: PcuDispatchPreparationRequest<'_>,
    ) -> Result<Self::Prepared, PcuDispatchPreparationError<Self::Error>> {
        self.assess_dispatch(request)
            .map_err(PcuDispatchPreparationError::Assessment)?;
        self.prepare_dispatch_direct(request)
            .map_err(PcuDispatchPreparationError::Backend)
    }
}

fn assess_floor(
    descriptor: PcuExecutorDescriptor,
    request: PcuDispatchPreparationRequest<'_>,
) -> Result<(), PcuDispatchAssessmentIssue> {
    let support = descriptor.support;
    let floor = request.capability_floor;
    if !support.value_types.contains(floor.value_types) {
        return Err(PcuDispatchAssessmentIssue::ValueTypeFloorUnsupported);
    }
    if !support.dispatch_features.contains(floor.features) {
        return Err(PcuDispatchAssessmentIssue::FeatureFloorUnsupported);
    }
    if !support.dispatch_instructions.contains(floor.instructions) {
        return Err(PcuDispatchAssessmentIssue::InstructionFloorUnsupported);
    }
    if !support.dispatch_policy.contains(floor.policy) {
        return Err(PcuDispatchAssessmentIssue::PolicyFloorUnsupported);
    }
    Ok(())
}

fn assess_limits(
    limits: Option<PcuDispatchLimits>,
    request: PcuDispatchPreparationRequest<'_>,
) -> Result<(), PcuDispatchAssessmentIssue> {
    let threads = request.submission.shape.thread_count().get();
    let bindings = u32::try_from(request.submission.kernel.bindings.len())
        .map_err(|_| PcuDispatchAssessmentIssue::CountOverflow(PcuDispatchLimitKind::Bindings))?;
    let parameters = u32::try_from(request.submission.kernel.parameters.len())
        .map_err(|_| PcuDispatchAssessmentIssue::CountOverflow(PcuDispatchLimitKind::Parameters))?;
    let Some(limits) = limits else {
        return if request.require_known_limits {
            Err(PcuDispatchAssessmentIssue::LimitUnknown(
                PcuDispatchLimitKind::LogicalThreads,
            ))
        } else {
            Ok(())
        };
    };
    check_limit(
        PcuDispatchLimitKind::LogicalThreads,
        threads,
        limits.max_logical_threads,
        request.require_known_limits,
    )?;
    check_limit(
        PcuDispatchLimitKind::Bindings,
        bindings,
        limits.max_bindings,
        request.require_known_limits,
    )?;
    check_limit(
        PcuDispatchLimitKind::Parameters,
        parameters,
        limits.max_parameters,
        request.require_known_limits,
    )
}

fn check_limit(
    kind: PcuDispatchLimitKind,
    requested: u32,
    maximum: Option<u32>,
    require_known: bool,
) -> Result<(), PcuDispatchAssessmentIssue> {
    let Some(maximum) = maximum else {
        return if require_known {
            Err(PcuDispatchAssessmentIssue::LimitUnknown(kind))
        } else {
            Ok(())
        };
    };
    if requested > maximum {
        Err(PcuDispatchAssessmentIssue::LimitExceeded {
            kind,
            requested,
            maximum,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroU32;
    use crate::{
        PcuExecutorClass,
        PcuExecutorOrigin,
        PcuExecutorSupport,
        PcuPrimitiveCaps,
        PcuProviderId,
        PcuObjectKind,
        PcuObjectRef,
        PcuSupport,
        PcuStreamCapabilities,
        PcuCommandOpCaps,
        PcuSignalOpCaps,
        PcuTransactionFeatureCaps,
        PcuDispatchKernelIr,
    };
    use crate::model::PcuDispatchKernelBuilder;

    #[derive(Debug)]
    struct Prepared {
        device: PcuDeviceIdentity,
        executor: PcuExecutorId,
    }

    struct Completion;

    impl PcuOwnedCompletion for Completion {
        type Error = PcuError;

        fn state(&self) -> Result<crate::PcuCompletionState, Self::Error> {
            Ok(crate::PcuCompletionState::Succeeded)
        }

        fn wait(&mut self) -> Result<crate::PcuCompletionOutcome, Self::Error> {
            Ok(crate::PcuCompletionOutcome::Succeeded)
        }
    }

    impl PcuPreparedDispatch for Prepared {
        type Completion = Completion;
        type Error = PcuError;

        fn device(&self) -> PcuDeviceIdentity {
            self.device
        }
        fn executor(&self) -> PcuExecutorId {
            self.executor
        }

        fn submit(
            self,
        ) -> Result<
            Self::Completion,
            PcuPreparedDispatchSubmitError<Self, Self::Completion, Self::Error>,
        > {
            Ok(Completion)
        }

        fn cancel(self) -> Result<(), PcuPreparedDispatchCancelError<Self, Self::Error>> {
            Ok(())
        }
    }

    struct Backend {
        device: PcuDeviceIdentity,
        executor: PcuExecutorDescriptor,
        limits: PcuDispatchLimits,
        prepared: bool,
    }

    impl PcuBaseContract for Backend {
        fn support(&self) -> PcuSupport {
            PcuSupport::unsupported()
        }
        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            // Tests use the backend's explicit descriptor override below; this empty static list
            // keeps the fixture no-allocation and the unknown-executor path honest.
            &[]
        }
        fn executor(&self, id: PcuExecutorId) -> Option<PcuExecutorDescriptor> {
            (id == self.executor.id).then_some(self.executor)
        }
    }

    impl PcuDispatchPreparationBackend for Backend {
        type Prepared = Prepared;
        type Error = PcuError;

        fn device_identity(&self) -> PcuDeviceIdentity {
            self.device
        }
        fn dispatch_limits(&self, executor: PcuExecutorId) -> Option<PcuDispatchLimits> {
            (executor == self.executor.id).then_some(self.limits)
        }
        fn prepare_dispatch_direct(
            &mut self,
            request: PcuDispatchPreparationRequest<'_>,
        ) -> Result<Self::Prepared, Self::Error> {
            self.prepared = true;
            Ok(Prepared {
                device: request.device,
                executor: request.executor,
            })
        }
    }

    fn device() -> PcuDeviceIdentity {
        PcuDeviceIdentity::from_device_ref(PcuObjectRef {
            provider: PcuProviderId(4),
            generation: 9,
            kind: PcuObjectKind::Device,
            id: 2,
        })
        .unwrap()
    }

    fn backend() -> Backend {
        Backend {
            device: device(),
            executor: PcuExecutorDescriptor {
                id: PcuExecutorId(3),
                name: "mock",
                class: PcuExecutorClass::Compute,
                origin: PcuExecutorOrigin::TopologyBound,
                support: PcuExecutorSupport {
                    primitives: PcuPrimitiveCaps::DISPATCH,
                    dispatch_policy: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
                    value_types: PcuValueTypeCaps::empty(),
                    dispatch_instructions: PcuDispatchOpCaps::empty(),
                    dispatch_features: PcuDispatchFeatureCaps::empty(),
                    stream_instructions: PcuStreamCapabilities::empty(),
                    command_instructions: PcuCommandOpCaps::empty(),
                    transaction_features: PcuTransactionFeatureCaps::empty(),
                    signal_instructions: PcuSignalOpCaps::empty(),
                },
            },
            limits: PcuDispatchLimits {
                max_logical_threads: Some(4),
                max_bindings: Some(0),
                max_parameters: Some(0),
            },
            prepared: false,
        }
    }

    fn request<'a>(kernel: &'a PcuDispatchKernelIr<'a>) -> PcuDispatchPreparationRequest<'a> {
        PcuDispatchPreparationRequest {
            device: device(),
            executor: PcuExecutorId(3),
            submission: PcuDispatchSubmission {
                kernel,
                shape: crate::PcuInvocationShape::threads(NonZeroU32::new(4).unwrap()),
            },
            parameters: PcuInvocationParameters::empty(),
            capability_floor: PcuDispatchCapabilityFloor::empty(),
            require_known_limits: true,
        }
    }

    #[test]
    fn assessment_prepares_only_after_common_validation_and_captures_target() {
        let builder = PcuDispatchKernelBuilder::<4>::new(1, "entry", [4, 1, 1]);
        let kernel = builder.ir();
        let mut backend = backend();
        assert_eq!(backend.assess_dispatch(request(&kernel)), Ok(()));
        assert!(!backend.prepared);
        let prepared = backend.prepare_dispatch(request(&kernel)).unwrap();
        assert!(backend.prepared);
        assert_eq!(prepared.device(), device());
        assert_eq!(prepared.executor(), PcuExecutorId(3));
        assert!(prepared.cancel().is_ok());
    }

    #[test]
    fn backend_limit_failure_is_structured_and_does_not_prepare() {
        let builder = PcuDispatchKernelBuilder::<4>::new(1, "entry", [4, 1, 1]);
        let kernel = builder.ir();
        let mut backend = backend();
        backend.limits.max_logical_threads = Some(3);
        let error = backend.prepare_dispatch(request(&kernel)).unwrap_err();
        assert!(matches!(
            error,
            PcuDispatchPreparationError::Assessment(PcuDispatchAssessmentError {
                issue: PcuDispatchAssessmentIssue::LimitExceeded {
                    kind: PcuDispatchLimitKind::LogicalThreads,
                    requested: 4,
                    maximum: 3,
                },
                ..
            })
        ));
        assert!(!backend.prepared);
    }
}
