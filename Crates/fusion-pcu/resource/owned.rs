//! Additive owned-resource and finite-completion vocabulary for PCU backends.
//!
//! The existing invocation API describes borrowed host slices and remains useful for synchronous
//! adapters. This module provides the small ownership vocabulary needed by asynchronous backends:
//! a binding owns its backend resource, and a completion implementation keeps those bindings alive
//! until it can prove that device access has stopped.

use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuError,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuKernel,
    PcuKernelIrContract,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderId,
    PcuBaseContract,
    PcuMemoryPoolId,
    PcuMemoryProvider,
};
use crate::dispatch::{
    PcuDispatchSubmission,
    validate_dispatch_submission,
    validate_parameters,
};

/// Backend- and discovery-generation-bound identity for one physical/logical device.
///
/// Unlike a local executor ordinal, this identity cannot alias a device from another provider or
/// discovery generation. Backends should derive it from the validated device reference used to
/// open their session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDeviceIdentity {
    provider: PcuProviderId,
    generation: u64,
    device_id: u32,
}

impl PcuDeviceIdentity {
    /// Creates an identity from a discovered device reference.
    ///
    /// The provider must still validate freshness and physical identity when opening the device;
    /// this value only preserves the reference's namespace and generation.
    #[must_use]
    pub const fn from_device_ref(reference: PcuObjectRef) -> Option<Self> {
        if matches!(reference.kind, PcuObjectKind::Device) {
            Some(Self {
                provider: reference.provider,
                generation: reference.generation,
                device_id: reference.id,
            })
        } else {
            None
        }
    }

    #[must_use]
    pub const fn provider(self) -> PcuProviderId {
        self.provider
    }

    #[must_use]
    pub const fn generation(self) -> u64 {
        self.generation
    }

    #[must_use]
    pub const fn device_id(self) -> u32 {
        self.device_id
    }
}

/// Metadata and owned backend resource for one invocation binding.
///
/// `R` should be a backend-owned allocation or lease, not a naked device pointer. The backend
/// must verify that its resource belongs to `device`, supports `access`, and is large enough for
/// the kernel before submission.
#[derive(Debug)]
pub struct PcuOwnedBinding<R> {
    pub target: PcuBindingRef,
    pub device: PcuDeviceIdentity,
    pub byte_len: u64,
    pub access: PcuBindingAccess,
    pub binding_type: PcuBindingType,
    pub resource: R,
}

impl<R> PcuOwnedBinding<R> {
    #[must_use]
    pub const fn new(
        target: PcuBindingRef,
        device: PcuDeviceIdentity,
        byte_len: u64,
        access: PcuBindingAccess,
        binding_type: PcuBindingType,
        resource: R,
    ) -> Self {
        Self {
            target,
            device,
            byte_len,
            access,
            binding_type,
            resource,
        }
    }

    #[must_use]
    pub fn into_resource(self) -> R {
        self.resource
    }
}

/// Binding admission failures for an owned Dispatch submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuOwnedDispatchBindingError {
    Duplicate(PcuBindingRef),
    Missing(PcuBindingRef),
    Unexpected(PcuBindingRef),
    WrongDevice(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    TypeMismatch(PcuBindingRef),
    UnsupportedLayout(PcuBindingRef),
    UnsupportedPorts,
    BindingCountMismatch {
        expected: usize,
        available: usize,
    },
    BufferTooSmall {
        binding: PcuBindingRef,
        required: u64,
        available: u64,
    },
}

/// Minimum owned binding metadata required by one prepared Dispatch executable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuOwnedBindingRequirement {
    pub target: PcuBindingRef,
    pub access: PcuBindingAccess,
    pub binding_type: PcuBindingType,
    pub min_required_bytes: u64,
}

/// Owned, backend-neutral binding admission schema for a prepared Dispatch executable.
///
/// The fixed array keeps this usable in `no_std` builds without requiring an allocator.
/// Construct it from a verified kernel before discarding the source IR. `N` must equal the
/// declared binding count; construction rejects a mismatch explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuOwnedDispatchBindingSchema<const N: usize> {
    shape: PcuInvocationShape,
    requirements: [PcuOwnedBindingRequirement; N],
}

impl<const N: usize> PcuOwnedDispatchBindingSchema<N> {
    /// Copies binding requirements from a verified kernel for one fixed invocation shape.
    ///
    /// # Errors
    ///
    /// Returns a binding-count mismatch or unsupported layout when the schema cannot represent
    /// the verified kernel's requirements.
    pub fn from_verified_kernel(
        kernel: &PcuDispatchKernelIr<'_>,
        shape: PcuInvocationShape,
    ) -> Result<Self, PcuOwnedDispatchBindingError> {
        if kernel.bindings.len() != N {
            return Err(PcuOwnedDispatchBindingError::BindingCountMismatch {
                expected: N,
                available: kernel.bindings.len(),
            });
        }
        let mut requirements = [PcuOwnedBindingRequirement {
            target: PcuBindingRef::new(0, 0),
            access: PcuBindingAccess::ReadOnly,
            binding_type: PcuBindingType::Value(crate::PcuValueType::f32()),
            min_required_bytes: 0,
        }; N];
        for (slot, declared) in requirements.iter_mut().zip(kernel.bindings) {
            *slot = PcuOwnedBindingRequirement::from_verified_binding(
                kernel,
                declared.reference(),
                shape,
            )?;
        }
        Ok(Self {
            shape,
            requirements,
        })
    }

    #[must_use]
    pub const fn shape(&self) -> PcuInvocationShape {
        self.shape
    }

    #[must_use]
    pub const fn requirements(&self) -> &[PcuOwnedBindingRequirement; N] {
        &self.requirements
    }

    /// Validates complete coverage and metadata without accessing source IR.
    ///
    /// # Errors
    ///
    /// Returns the first duplicate, missing, unexpected, mismatched, or undersized binding.
    pub fn validate<R>(
        &self,
        device: PcuDeviceIdentity,
        bindings: &[PcuOwnedBinding<R>],
    ) -> Result<(), PcuOwnedDispatchBindingError> {
        validate_owned_binding_requirements(&self.requirements, device, bindings)
    }
}

impl PcuOwnedBindingRequirement {
    /// Copies one binding's metadata and minimum byte extent from a verified kernel.
    ///
    /// # Errors
    ///
    /// Returns `UnsupportedLayout` when the binding has no scalar byte-layout rule.
    pub fn from_verified_binding(
        kernel: &PcuDispatchKernelIr<'_>,
        target: PcuBindingRef,
        shape: PcuInvocationShape,
    ) -> Result<Self, PcuOwnedDispatchBindingError> {
        let Some(declared) = kernel.bindings.iter().find(|b| b.reference() == target) else {
            return Err(PcuOwnedDispatchBindingError::Unexpected(target));
        };
        let Some(bytes_per_element) = scalar_value_bytes(declared.binding_type) else {
            return Err(PcuOwnedDispatchBindingError::UnsupportedLayout(target));
        };
        let elements = kernel.minimum_binding_elements_for(target, shape.invocation_count().get());
        let Some(min_required_bytes) = bytes_per_element.checked_mul(u64::from(elements)) else {
            return Err(PcuOwnedDispatchBindingError::UnsupportedLayout(target));
        };
        Ok(Self {
            target,
            access: declared.access,
            binding_type: declared.binding_type,
            min_required_bytes,
        })
    }
}

/// Type-erased validation interface for fixed-size owned binding schemas.
pub trait PcuOwnedDispatchBindingSchemaContract {
    /// Validates bindings against this schema and the selected device.
    ///
    /// # Errors
    ///
    /// Returns the first duplicate, missing, unexpected, mismatched, or undersized binding.
    fn validate<R>(
        &self,
        device: PcuDeviceIdentity,
        bindings: &[PcuOwnedBinding<R>],
    ) -> Result<(), PcuOwnedDispatchBindingError>;
}

impl<const N: usize> PcuOwnedDispatchBindingSchemaContract for PcuOwnedDispatchBindingSchema<N> {
    fn validate<R>(
        &self,
        device: PcuDeviceIdentity,
        bindings: &[PcuOwnedBinding<R>],
    ) -> Result<(), PcuOwnedDispatchBindingError> {
        Self::validate(self, device, bindings)
    }
}

impl PcuOwnedDispatchBindingSchemaContract for [PcuOwnedBindingRequirement] {
    fn validate<R>(
        &self,
        device: PcuDeviceIdentity,
        bindings: &[PcuOwnedBinding<R>],
    ) -> Result<(), PcuOwnedDispatchBindingError> {
        validate_owned_binding_requirements(self, device, bindings)
    }
}

/// Validates owned bindings against an owned requirement slice.
///
/// # Errors
///
/// Returns the first duplicate, missing, unexpected, mismatched, or undersized binding.
pub fn validate_owned_binding_requirements<R>(
    requirements: &[PcuOwnedBindingRequirement],
    device: PcuDeviceIdentity,
    bindings: &[PcuOwnedBinding<R>],
) -> Result<(), PcuOwnedDispatchBindingError> {
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|previous| previous.target == binding.target)
        {
            return Err(PcuOwnedDispatchBindingError::Duplicate(binding.target));
        }
        let Some(required) = requirements.iter().find(|r| r.target == binding.target) else {
            return Err(PcuOwnedDispatchBindingError::Unexpected(binding.target));
        };
        if binding.device != device {
            return Err(PcuOwnedDispatchBindingError::WrongDevice(binding.target));
        }
        if !access_supports(binding.access, required.access) {
            return Err(PcuOwnedDispatchBindingError::AccessMismatch(binding.target));
        }
        if binding.binding_type != required.binding_type {
            return Err(PcuOwnedDispatchBindingError::TypeMismatch(binding.target));
        }
        if binding.byte_len < required.min_required_bytes {
            return Err(PcuOwnedDispatchBindingError::BufferTooSmall {
                binding: binding.target,
                required: required.min_required_bytes,
                available: binding.byte_len,
            });
        }
    }
    for required in requirements {
        if !bindings
            .iter()
            .any(|binding| binding.target == required.target)
        {
            return Err(PcuOwnedDispatchBindingError::Missing(required.target));
        }
    }
    Ok(())
}

/// Failure returned by the common owned Dispatch admission path.
#[derive(Debug)]
pub enum PcuOwnedDispatchError<E> {
    Admission(PcuError),
    Binding(PcuOwnedDispatchBindingError),
    Backend(E),
}

/// Direct Dispatch backend that consumes owned bindings for one selected device/session.
///
/// The selected device identity is fixed by the backend value and checked against every binding
/// before implementation-specific submission. The direct method consumes the binding container.
/// Because the kernel IR and parameter table are borrowed, the backend must finish lowering or
/// otherwise snapshot any data it needs from them before returning the completion handle.
/// If it returns an error before enqueue, it may release those resources. If it returns an error
/// after enqueue or after launch outcome becomes uncertain, it must first establish device
/// quiescence, or conservatively retain/leak every resource that device work may still reference.
pub trait PcuOwnedDispatchBackend: PcuBaseContract {
    type Resource;
    /// Fixed-size or allocating owned container selected by the backend implementation.
    type Bindings: AsRef<[PcuOwnedBinding<Self::Resource>]>;
    type Completion: PcuOwnedCompletion;
    type Error;

    /// Reusable executable prepared for one kernel, shape, and parameter set.
    type Prepared<'kernel, 'parameters>: PcuPreparedOwnedDispatch<
            Resource = Self::Resource,
            Bindings = Self::Bindings,
            Completion = Self::Completion,
            Error = Self::Error,
        >
    where
        Self: 'kernel;

    /// Returns the identity of the device/session selected by this backend instance.
    fn device_identity(&self) -> PcuDeviceIdentity;

    /// Submits a validated owned Dispatch operation.
    ///
    /// # Errors
    ///
    /// Returns a backend error when the selected device cannot submit the operation.
    fn submit_dispatch_owned_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: Self::Bindings,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::Completion, Self::Error>;

    /// Performs backend preparation such as lowering, compilation, and executable creation.
    ///
    /// Parameters may be borrowed for the lifetime of the prepared executable. Implementations
    /// must snapshot any borrowed data they need if they wish to return a longer-lived value.
    ///
    /// # Errors
    ///
    /// Returns a backend error when lowering or executable preparation fails.
    fn prepare_dispatch_owned_direct<'kernel, 'parameters>(
        &self,
        submission: PcuDispatchSubmission<'kernel>,
        parameters: PcuInvocationParameters<'parameters>,
    ) -> Result<Self::Prepared<'kernel, 'parameters>, Self::Error>;

    /// Performs common shape, parameter, support, device, and binding admission before submission.
    ///
    /// # Errors
    ///
    /// Returns a common validation error or the backend's submission error.
    fn submit_dispatch_owned(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: Self::Bindings,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::Completion, PcuOwnedDispatchError<Self::Error>> {
        if !self
            .support()
            .supports_kernel_direct(PcuKernel::Dispatch(*submission.kernel))
        {
            return Err(PcuOwnedDispatchError::Admission(PcuError::unsupported()));
        }
        validate_dispatch_submission(submission).map_err(PcuOwnedDispatchError::Admission)?;
        validate_parameters(submission.kernel.signature(), parameters)
            .map_err(PcuOwnedDispatchError::Admission)?;
        if !submission.kernel.ports.is_empty() {
            return Err(PcuOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::UnsupportedPorts,
            ));
        }
        validate_owned_dispatch_bindings(
            submission.kernel,
            submission.shape,
            self.device_identity(),
            bindings.as_ref(),
        )
        .map_err(PcuOwnedDispatchError::Binding)?;
        self.submit_dispatch_owned_direct(submission, bindings, parameters)
            .map_err(PcuOwnedDispatchError::Backend)
    }

    /// Validates and prepares one Dispatch executable for repeated owned-binding submissions.
    ///
    /// Common support, shape, parameter, and port admission runs once here. Every later call to
    /// [`PcuPreparedOwnedDispatch::submit_owned`] repeats device, binding, and extent validation.
    ///
    /// # Errors
    ///
    /// Returns a common admission error or a backend preparation error.
    fn prepare_dispatch_owned<'kernel, 'parameters>(
        &self,
        submission: PcuDispatchSubmission<'kernel>,
        parameters: PcuInvocationParameters<'parameters>,
    ) -> Result<Self::Prepared<'kernel, 'parameters>, PcuOwnedDispatchError<Self::Error>> {
        if !self
            .support()
            .supports_kernel_direct(PcuKernel::Dispatch(*submission.kernel))
        {
            return Err(PcuOwnedDispatchError::Admission(PcuError::unsupported()));
        }
        validate_dispatch_submission(submission).map_err(PcuOwnedDispatchError::Admission)?;
        validate_parameters(submission.kernel.signature(), parameters)
            .map_err(PcuOwnedDispatchError::Admission)?;
        if !submission.kernel.ports.is_empty() {
            return Err(PcuOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::UnsupportedPorts,
            ));
        }
        self.prepare_dispatch_owned_direct(submission, parameters)
            .map_err(PcuOwnedDispatchError::Backend)
    }
}

/// Backend-specific reusable executable for repeated owned Dispatch submissions.
///
/// Preparation captures or snapshots the kernel and parameter data needed for later launches.
/// Each submission consumes its own binding container so the backend can move all resource
/// leases into the returned completion. It must keep those leases until the completion law in
/// [`PcuOwnedCompletion`] establishes quiescence.
pub trait PcuPreparedOwnedDispatch {
    type Resource;
    type Bindings: AsRef<[PcuOwnedBinding<Self::Resource>]>;
    type BindingSchema: PcuOwnedDispatchBindingSchemaContract + ?Sized;
    type Completion: PcuOwnedCompletion;
    type Error;

    /// Owned admission metadata captured during preparation; independent of source IR lifetime.
    fn binding_schema(&self) -> &Self::BindingSchema;
    fn shape(&self) -> PcuInvocationShape;
    fn device_identity(&self) -> PcuDeviceIdentity;

    /// Submits after common binding admission. Implementations must retain resources through
    /// terminal quiescence, including on uncertain post-enqueue errors.
    ///
    /// # Errors
    ///
    /// Returns a backend submission error.
    fn submit_owned_direct(
        &self,
        bindings: Self::Bindings,
    ) -> Result<Self::Completion, Self::Error>;

    /// Validates complete binding coverage, device identity, metadata, and minimum extent before
    /// handing the owned bindings to the backend.
    ///
    /// # Errors
    ///
    /// Returns a common binding admission error or a backend submission error.
    fn submit_owned(
        &self,
        bindings: Self::Bindings,
    ) -> Result<Self::Completion, PcuOwnedDispatchError<Self::Error>> {
        self.binding_schema()
            .validate(self.device_identity(), bindings.as_ref())
            .map_err(PcuOwnedDispatchError::Binding)?;
        self.submit_owned_direct(bindings)
            .map_err(PcuOwnedDispatchError::Backend)
    }
}

/// Connects an owned Dispatch session to the same backend's memory provider and binding model.
///
/// A session returns a provider scoped to the requested stable pool. Callers allocate or import
/// resources through that provider, then pass references to those resources to [`Self::bind`]
/// before submitting the resulting owned bindings through [`PcuOwnedDispatchBackend`]. Binding
/// may retain, clone, or otherwise acquire a backend lease; the returned binding must keep every
/// resource needed by dispatch alive through completion. It must also report the selected device,
/// actual byte length, access, and type accurately so common owned-dispatch admission can validate
/// it. The provider resource remains caller-owned, so implementations must not consume or silently
/// invalidate it during binding.
///
/// This is an extension contract: a backend may implement [`PcuOwnedDispatchBackend`] without
/// exposing its memory service through this trait.
pub trait PcuOwnedDispatchMemorySession: PcuOwnedDispatchBackend {
    /// Provider tied to this dispatch backend's device/session domain.
    type MemoryProvider: PcuMemoryProvider;

    /// Creates a provider view for a stable pool identity.
    ///
    /// The returned provider must address the same device generation as `self`. The caller passes
    /// a pool identity obtained through discovery; provider operations must reject requests for
    /// pools other than the one supplied here.
    fn memory_provider(&self, pool: PcuMemoryPoolId) -> Self::MemoryProvider;

    /// Creates an owned dispatch binding from a resource allocated or imported by this session's
    /// memory provider.
    ///
    /// Implementations must preserve the resource's lifetime and enforce the requested access and
    /// binding type. The returned binding's device identity and byte length are checked again by
    /// [`PcuOwnedDispatchBackend::submit_dispatch_owned`].
    ///
    /// # Errors
    ///
    /// Returns the backend error if the resource cannot be bound for the requested target or
    /// metadata.
    fn bind(
        &self,
        target: PcuBindingRef,
        access: PcuBindingAccess,
        binding_type: PcuBindingType,
        resource: &<Self::MemoryProvider as PcuMemoryProvider>::Resource,
    ) -> Result<PcuOwnedBinding<Self::Resource>, Self::Error>;
}

/// Validates complete owned binding coverage and metadata for scalar value buffers.
///
/// This v1 sizing rule assumes a tightly packed, contiguous buffer with one scalar element per
/// logical invocation. Vector, matrix, image, sampler, and acceleration-structure layouts are rejected
/// until a backend-specific layout contract can describe their actual storage size and stride.
///
/// # Errors
///
/// Returns the first missing, duplicate, mismatched, or undersized binding error.
pub fn validate_owned_dispatch_bindings<R>(
    kernel: &PcuDispatchKernelIr<'_>,
    shape: PcuInvocationShape,
    device: PcuDeviceIdentity,
    bindings: &[PcuOwnedBinding<R>],
) -> Result<(), PcuOwnedDispatchBindingError> {
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|previous| previous.target == binding.target)
        {
            return Err(PcuOwnedDispatchBindingError::Duplicate(binding.target));
        }
        let Some(declared) = kernel
            .bindings
            .iter()
            .find(|declared| declared.reference() == binding.target)
        else {
            return Err(PcuOwnedDispatchBindingError::Unexpected(binding.target));
        };
        if binding.device != device {
            return Err(PcuOwnedDispatchBindingError::WrongDevice(binding.target));
        }
        if !access_supports(binding.access, declared.access) {
            return Err(PcuOwnedDispatchBindingError::AccessMismatch(binding.target));
        }
        if binding.binding_type != declared.binding_type {
            return Err(PcuOwnedDispatchBindingError::TypeMismatch(binding.target));
        }
        let Some(bytes_per_element) = scalar_value_bytes(declared.binding_type) else {
            return Err(PcuOwnedDispatchBindingError::UnsupportedLayout(
                binding.target,
            ));
        };
        let required_elements =
            kernel.minimum_binding_elements_for(binding.target, shape.invocation_count().get());
        let Some(required) = bytes_per_element.checked_mul(u64::from(required_elements)) else {
            return Err(PcuOwnedDispatchBindingError::UnsupportedLayout(
                binding.target,
            ));
        };
        if binding.byte_len < required {
            return Err(PcuOwnedDispatchBindingError::BufferTooSmall {
                binding: binding.target,
                required,
                available: binding.byte_len,
            });
        }
    }
    for declared in kernel.bindings {
        let target = declared.reference();
        if !bindings.iter().any(|binding| binding.target == target) {
            return Err(PcuOwnedDispatchBindingError::Missing(target));
        }
    }
    Ok(())
}

fn access_supports(resource: PcuBindingAccess, required: PcuBindingAccess) -> bool {
    match required {
        PcuBindingAccess::ReadOnly => {
            matches!(
                resource,
                PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
            )
        }
        PcuBindingAccess::WriteOnly => {
            matches!(
                resource,
                PcuBindingAccess::WriteOnly | PcuBindingAccess::ReadWrite
            )
        }
        PcuBindingAccess::ReadWrite => resource == PcuBindingAccess::ReadWrite,
    }
}

fn scalar_value_bytes(binding_type: PcuBindingType) -> Option<u64> {
    let crate::PcuValueType::Scalar(scalar) = binding_type.value_type()? else {
        return None;
    };
    Some(u64::from(scalar.bit_width()).div_ceil(8))
}

/// Observable progress for one finite submitted operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCompletionState {
    Pending,
    Running,
    Succeeded,
    Failed,
}

impl PcuCompletionState {
    /// Whether this state guarantees that the device has stopped accessing submission resources.
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Succeeded | Self::Failed)
    }
}

/// Terminal result of a finite operation.
///
/// Every variant means the operation is quiescent: the device no longer accesses resources owned
/// by the completion handle. A structured arithmetic fault identifies its kind and first
/// invocation; backends may retain additional diagnostics on the handle or its error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCompletionOutcome {
    Succeeded,
    Failed,
    Fault(crate::PcuExecutionFault),
}

/// Error returned by [`PcuOwnedSubmission::wait_result`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSubmissionWaitError<E> {
    /// The backend could not establish completion; the submission remains owned and retryable.
    Backend(E),
    /// Completion is terminal and quiescent, but failed without a structured execution fault.
    Failed,
    /// Completion is terminal and quiescent with a deterministic arithmetic fault.
    Fault(crate::PcuExecutionFault),
}

/// Owned completion contract for one finite operation.
///
/// Implementations must retain every submitted resource until success or failure is confirmed to
/// be terminal and quiescent. An `Err` from `wait` means completion is uncertain; the handle must
/// remain usable for a later retry and must continue retaining all resources. Returning either
/// terminal outcome, or reporting a terminal state, guarantees that device access has stopped.
///
/// Drop law: dropping a non-terminal handle must establish quiescence before releasing resources,
/// or conservatively retain/leak those resources if the backend cannot establish it. A backend
/// may document a stronger cancellation guarantee, but it must never free in-flight resources on
/// an unconfirmed error.
pub trait PcuOwnedCompletion {
    type Error;

    /// Returns current progress. A terminal state guarantees quiescence.
    /// An error leaves completion uncertain and does not permit releasing retained resources.
    ///
    /// # Errors
    ///
    /// Returns a backend error when progress cannot be determined safely.
    fn state(&self) -> Result<PcuCompletionState, Self::Error>;

    /// Waits for completion while preserving the handle on uncertain errors.
    ///
    /// # Errors
    ///
    /// Returns a backend error while retaining completion ownership for retry.
    fn wait(&mut self) -> Result<PcuCompletionOutcome, Self::Error>;
}

/// Coarse status exposed by the allocation-free core submission queue.
///
/// `Unknown` means the backend could not establish progress. It is deliberately distinct from
/// either terminal result and must never be treated as permission to reuse or release storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuQueueCompletionStatus {
    Pending,
    Running,
    Complete,
    Failed,
    Unknown,
}

/// One queued finite operation together with every resource whose lifetime it protects.
///
/// This is a small provider-independent ownership envelope, not a scheduler. The backend may
/// complete synchronously or asynchronously through [`PcuOwnedCompletion`]. The resource bundle
/// is returned only after terminal quiescence is observed. If the envelope is dropped while its
/// status is pending, running, or unknown, both completion and resources are intentionally
/// forgotten so their destructors cannot free storage that may still be in use.
#[derive(Debug)]
pub struct PcuOwnedSubmission<C, R>
where
    C: PcuOwnedCompletion,
{
    completion: Option<C>,
    resources: Option<R>,
    terminal: Option<PcuCompletionOutcome>,
    terminal_confirmed_by_wait: bool,
}

impl<C, R> PcuOwnedSubmission<C, R>
where
    C: PcuOwnedCompletion,
{
    /// Places one backend completion and its retained resource bundle in the core queue.
    #[must_use]
    pub const fn new(completion: C, resources: R) -> Self {
        Self {
            completion: Some(completion),
            resources: Some(resources),
            terminal: None,
            terminal_confirmed_by_wait: false,
        }
    }

    /// Queries progress, mapping a backend query error to the conservative `Unknown` state.
    pub fn status(&mut self) -> PcuQueueCompletionStatus {
        if let Some(outcome) = self.terminal {
            return status_from_outcome(outcome);
        }
        let Some(completion) = self.completion.as_ref() else {
            return PcuQueueCompletionStatus::Unknown;
        };
        match completion.state() {
            Ok(PcuCompletionState::Pending) => PcuQueueCompletionStatus::Pending,
            Ok(PcuCompletionState::Running) => PcuQueueCompletionStatus::Running,
            Ok(PcuCompletionState::Succeeded) => {
                self.terminal = Some(PcuCompletionOutcome::Succeeded);
                PcuQueueCompletionStatus::Complete
            }
            Ok(PcuCompletionState::Failed) => {
                self.terminal = Some(PcuCompletionOutcome::Failed);
                PcuQueueCompletionStatus::Failed
            }
            Err(_) => PcuQueueCompletionStatus::Unknown,
        }
    }

    /// Waits for terminal completion. An error leaves ownership in the queue for retry.
    ///
    /// # Errors
    ///
    /// Returns the backend's error when it cannot establish completion.
    ///
    /// # Panics
    ///
    /// Panics only if internal ownership invariants have been violated; public methods never
    /// remove the completion handle.
    pub fn wait(&mut self) -> Result<PcuCompletionOutcome, C::Error> {
        if let Some(outcome) = self.terminal
            && (outcome != PcuCompletionOutcome::Failed || self.terminal_confirmed_by_wait)
        {
            return Ok(outcome);
        }
        let outcome = self
            .completion
            .as_mut()
            .expect("live submission always owns its completion")
            .wait()?;
        self.terminal = Some(outcome);
        self.terminal_confirmed_by_wait = true;
        Ok(outcome)
    }

    /// Waits for completion and lifts terminal failure and arithmetic fault into a typed result.
    ///
    /// A backend error is uncertain and leaves completion and resources owned for retry. The
    /// `Failed` and `Fault` variants are terminal and quiescent, so retained resources may be
    /// extracted safely after either result.
    ///
    /// # Errors
    ///
    /// Returns [`PcuSubmissionWaitError::Backend`] when backend completion remains uncertain,
    /// [`PcuSubmissionWaitError::Failed`] for an unstructured terminal failure, or
    /// [`PcuSubmissionWaitError::Fault`] for a structured execution fault.
    ///
    /// # Panics
    ///
    /// Panics only if the submission's internal completion ownership invariant is violated.
    pub fn wait_result(&mut self) -> Result<(), PcuSubmissionWaitError<C::Error>> {
        let outcome = match self.terminal {
            Some(PcuCompletionOutcome::Succeeded) => return Ok(()),
            Some(PcuCompletionOutcome::Fault(fault)) => {
                return Err(PcuSubmissionWaitError::Fault(fault));
            }
            Some(PcuCompletionOutcome::Failed) if self.terminal_confirmed_by_wait => {
                return Err(PcuSubmissionWaitError::Failed);
            }
            Some(PcuCompletionOutcome::Failed) => {
                self.wait().map_err(PcuSubmissionWaitError::Backend)?
            }
            None => self.wait().map_err(PcuSubmissionWaitError::Backend)?,
        };
        match outcome {
            PcuCompletionOutcome::Succeeded => Ok(()),
            PcuCompletionOutcome::Failed => Err(PcuSubmissionWaitError::Failed),
            PcuCompletionOutcome::Fault(fault) => Err(PcuSubmissionWaitError::Fault(fault)),
        }
    }

    /// Extracts the retained resource bundle after terminal quiescence has been confirmed.
    ///
    /// Returns `None` while the operation is pending, running, or unknown.
    pub fn take_resources(&mut self) -> Option<R> {
        self.terminal.map(|_| ())?;
        self.resources.take()
    }
}

impl<C, R> Drop for PcuOwnedSubmission<C, R>
where
    C: PcuOwnedCompletion,
{
    fn drop(&mut self) {
        if self.terminal.is_none() {
            // `forget` is safe and intentionally prevents unconfirmed native/device resources
            // from being released by either their own destructor or the completion destructor.
            if let Some(completion) = self.completion.take() {
                core::mem::forget(completion);
            }
            if let Some(resources) = self.resources.take() {
                core::mem::forget(resources);
            }
        }
    }
}

const fn status_from_outcome(outcome: PcuCompletionOutcome) -> PcuQueueCompletionStatus {
    match outcome {
        PcuCompletionOutcome::Succeeded => PcuQueueCompletionStatus::Complete,
        PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_) => {
            PcuQueueCompletionStatus::Failed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuCompletionOutcome,
        PcuCompletionState,
        PcuSubmissionWaitError,
        PcuDeviceIdentity,
        PcuOwnedBinding,
        PcuOwnedCompletion,
        PcuOwnedSubmission,
        PcuQueueCompletionStatus,
        PcuOwnedDispatchBackend,
        PcuOwnedDispatchBindingError,
        PcuPreparedOwnedDispatch,
        validate_owned_dispatch_bindings,
    };
    use crate::{
        PcuBaseContract,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuBindingType,
        PcuDispatchFeatureCaps,
        PcuDispatchPolicyCaps,
        PcuDispatchSubmission,
        PcuDispatchSupport,
        PcuExecutorDescriptor,
        PcuFeatureSupport,
        PcuInvocationParameters,
        PcuInvocationShape,
        PcuObjectKind,
        PcuObjectRef,
        PcuPrimitiveCaps,
        PcuPrimitiveSupport,
        PcuProviderId,
        PcuSupport,
        PcuValueType,
        PcuValueTypeCaps,
    };
    use crate::model::PcuDispatchKernelBuilder;
    use crate::PcuScalarType;
    use std::{
        boxed::Box,
        cell::Cell,
        rc::Rc,
    };

    struct Resource(Rc<Cell<bool>>);

    impl Drop for Resource {
        fn drop(&mut self) {
            self.0.set(true);
        }
    }

    struct FakeCompletion {
        resource: Option<PcuOwnedBinding<Resource>>,
        attempts: u8,
        terminal: Option<PcuCompletionOutcome>,
    }

    fn device(id: u32) -> PcuDeviceIdentity {
        PcuDeviceIdentity::from_device_ref(PcuObjectRef {
            provider: PcuProviderId(7),
            generation: 9,
            kind: PcuObjectKind::Device,
            id,
        })
        .unwrap()
    }

    fn make_kernel<'a>(bindings: &'a [PcuBinding<'a>]) -> crate::PcuDispatchKernelIr<'a> {
        let builder = Box::leak(Box::new(
            PcuDispatchKernelBuilder::<1>::new(1, "owned", [4, 1, 1]).with_bindings(bindings),
        ));
        builder.ir()
    }

    fn support() -> PcuSupport {
        let mut support = PcuSupport::unsupported();
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(
                PcuPrimitiveCaps::DISPATCH,
                PcuPrimitiveCaps::empty(),
            ),
        };
        let mut dispatch = PcuDispatchSupport::unsupported();
        dispatch.flags = PcuDispatchPolicyCaps::ORDERED_SUBMISSION;
        dispatch.features.direct = PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
        support.dispatch_support = dispatch;
        support.value_type_support.direct =
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::SCALAR_VALUES;
        support
    }

    struct FakeBackend {
        device: PcuDeviceIdentity,
    }

    struct FakePrepared<'kernel> {
        schema: super::PcuOwnedDispatchBindingSchema<1>,
        shape: PcuInvocationShape,
        device: PcuDeviceIdentity,
        marker: core::marker::PhantomData<&'kernel ()>,
    }

    impl PcuPreparedOwnedDispatch for FakePrepared<'_> {
        type Resource = Resource;
        type Bindings = [PcuOwnedBinding<Resource>; 1];
        type BindingSchema = super::PcuOwnedDispatchBindingSchema<1>;
        type Completion = FakeCompletion;
        type Error = &'static str;

        fn binding_schema(&self) -> &Self::BindingSchema {
            &self.schema
        }

        fn shape(&self) -> PcuInvocationShape {
            self.shape
        }

        fn device_identity(&self) -> PcuDeviceIdentity {
            self.device
        }

        fn submit_owned_direct(
            &self,
            [binding]: Self::Bindings,
        ) -> Result<Self::Completion, Self::Error> {
            Ok(FakeCompletion {
                resource: Some(binding),
                attempts: 0,
                terminal: None,
            })
        }
    }

    impl PcuBaseContract for FakeBackend {
        fn support(&self) -> PcuSupport {
            support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            &[]
        }
    }

    impl PcuOwnedDispatchBackend for FakeBackend {
        type Resource = Resource;
        type Bindings = [PcuOwnedBinding<Resource>; 1];
        type Completion = FakeCompletion;
        type Error = &'static str;
        type Prepared<'kernel, 'parameters> = FakePrepared<'kernel>;

        fn device_identity(&self) -> PcuDeviceIdentity {
            self.device
        }

        fn submit_dispatch_owned_direct(
            &self,
            _submission: PcuDispatchSubmission<'_>,
            [binding]: Self::Bindings,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::Completion, Self::Error> {
            Ok(FakeCompletion {
                resource: Some(binding),
                attempts: 0,
                terminal: None,
            })
        }

        fn prepare_dispatch_owned_direct<'kernel, 'parameters>(
            &self,
            submission: PcuDispatchSubmission<'kernel>,
            _parameters: PcuInvocationParameters<'parameters>,
        ) -> Result<Self::Prepared<'kernel, 'parameters>, Self::Error> {
            Ok(FakePrepared {
                schema: super::PcuOwnedDispatchBindingSchema::from_verified_kernel(
                    submission.kernel,
                    submission.shape,
                )
                .expect("test kernel has supported binding layout"),
                shape: submission.shape,
                device: self.device,
                marker: core::marker::PhantomData,
            })
        }
    }

    fn declared_binding() -> PcuBinding<'static> {
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        )
    }

    fn owned_binding(
        device: PcuDeviceIdentity,
        bytes: u64,
        access: PcuBindingAccess,
        dropped: Rc<Cell<bool>>,
    ) -> PcuOwnedBinding<Resource> {
        PcuOwnedBinding::new(
            PcuBindingRef::new(0, 0),
            device,
            bytes,
            access,
            PcuBindingType::Value(PcuValueType::f32()),
            Resource(dropped),
        )
    }

    #[test]
    fn owned_dispatch_admission_rejects_wrong_device_access_size_and_missing_binding() {
        let declarations = [declared_binding()];
        let kernel = make_kernel(&declarations);
        let shape = PcuInvocationShape::invocations(core::num::NonZeroU32::new(4).unwrap());
        let target = PcuBindingRef::new(0, 0);
        let not_dropped = || Rc::new(Cell::new(false));

        assert_eq!(
            validate_owned_dispatch_bindings(
                &kernel,
                shape,
                device(1),
                &[owned_binding(
                    device(2),
                    16,
                    PcuBindingAccess::ReadOnly,
                    not_dropped()
                )],
            ),
            Err(PcuOwnedDispatchBindingError::WrongDevice(target))
        );
        assert_eq!(
            validate_owned_dispatch_bindings(
                &kernel,
                shape,
                device(1),
                &[owned_binding(
                    device(1),
                    16,
                    PcuBindingAccess::WriteOnly,
                    not_dropped()
                )],
            ),
            Err(PcuOwnedDispatchBindingError::AccessMismatch(target))
        );
        assert_eq!(
            validate_owned_dispatch_bindings(
                &kernel,
                shape,
                device(1),
                &[owned_binding(
                    device(1),
                    15,
                    PcuBindingAccess::ReadOnly,
                    not_dropped()
                )],
            ),
            Err(PcuOwnedDispatchBindingError::BufferTooSmall {
                binding: target,
                required: 16,
                available: 15,
            })
        );
        assert_eq!(
            validate_owned_dispatch_bindings::<Resource>(&kernel, shape, device(1), &[]),
            Err(PcuOwnedDispatchBindingError::Missing(target))
        );
    }

    #[test]
    fn owned_dispatch_rejects_non_scalar_layout_until_layout_contract_exists() {
        let declaration = PcuBinding::value(
            Some("vector"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::vector(PcuScalarType::F32, 2),
        );
        let declarations = [declaration];
        let kernel = make_kernel(&declarations);
        let binding = PcuOwnedBinding::new(
            PcuBindingRef::new(0, 0),
            device(1),
            32,
            PcuBindingAccess::ReadOnly,
            declaration.binding_type,
            (),
        );
        assert_eq!(
            validate_owned_dispatch_bindings(
                &kernel,
                PcuInvocationShape::invocations(core::num::NonZeroU32::new(4).unwrap()),
                device(1),
                &[binding],
            ),
            Err(PcuOwnedDispatchBindingError::UnsupportedLayout(
                PcuBindingRef::new(0, 0)
            ))
        );
    }

    #[test]
    fn owned_schema_copies_binding_element_zero_and_invocation_extents() {
        use crate::{
            PcuDispatchDataOp,
            PcuDispatchIndex,
            PcuDispatchOp,
            PcuDispatchValueId,
        };
        let declarations = [
            PcuBinding::scalar::<f32>(
                Some("scalar"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
            ),
            PcuBinding::scalar::<f32>(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
            ),
        ];
        let ops = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index: PcuDispatchIndex::BindingElementZero,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index: PcuDispatchIndex::InvocationId,
                value: PcuDispatchValueId(1),
            }),
        ];
        let builder = PcuDispatchKernelBuilder::<2>::new(1, "broadcast", [64, 1, 1])
            .with_bindings(&declarations)
            .with_ops(&ops)
            .expect("operations fit");
        let shape = PcuInvocationShape::invocations(core::num::NonZeroU32::new(64).unwrap());
        let schema =
            super::PcuOwnedDispatchBindingSchema::<2>::from_verified_kernel(&builder.ir(), shape)
                .unwrap();
        assert_eq!(schema.shape(), shape);
        assert_eq!(schema.requirements()[0].min_required_bytes, 4);
        assert_eq!(schema.requirements()[1].min_required_bytes, 256);
        assert_eq!(
            super::PcuOwnedDispatchBindingSchema::<1>::from_verified_kernel(&builder.ir(), shape),
            Err(PcuOwnedDispatchBindingError::BindingCountMismatch {
                expected: 1,
                available: 2,
            })
        );
    }

    impl PcuOwnedCompletion for FakeCompletion {
        type Error = &'static str;

        fn state(&self) -> Result<PcuCompletionState, Self::Error> {
            Ok(match self.terminal {
                Some(PcuCompletionOutcome::Succeeded) => PcuCompletionState::Succeeded,
                Some(PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_)) => {
                    PcuCompletionState::Failed
                }
                None if self.attempts == 0 => PcuCompletionState::Pending,
                None => PcuCompletionState::Running,
            })
        }

        fn wait(&mut self) -> Result<PcuCompletionOutcome, Self::Error> {
            self.attempts += 1;
            if self.attempts == 1 {
                return Err("completion status uncertain");
            }
            self.terminal = Some(PcuCompletionOutcome::Succeeded);
            self.resource.take();
            Ok(PcuCompletionOutcome::Succeeded)
        }
    }

    #[test]
    fn resources_survive_uncertain_wait_and_release_after_confirmed_completion() {
        let declarations = [declared_binding()];
        let kernel = make_kernel(&declarations);
        let device = device(1);
        let backend = FakeBackend { device };
        let dropped = Rc::new(Cell::new(false));
        let binding = owned_binding(device, 16, PcuBindingAccess::ReadOnly, Rc::clone(&dropped));
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(4).unwrap()),
        };
        let mut completion = backend
            .submit_dispatch_owned(submission, [binding], PcuInvocationParameters::empty())
            .unwrap();

        assert_eq!(completion.wait(), Err("completion status uncertain"));
        assert!(!dropped.get());
        assert_eq!(completion.state().unwrap(), PcuCompletionState::Running);

        assert_eq!(completion.wait(), Ok(PcuCompletionOutcome::Succeeded));
        assert!(dropped.get());
        assert_eq!(completion.state().unwrap(), PcuCompletionState::Succeeded);
    }

    #[test]
    fn prepared_owned_dispatch_revalidates_each_binding_set() {
        let declarations = [declared_binding()];
        let kernel = make_kernel(&declarations);
        let selected = device(1);
        let backend = FakeBackend { device: selected };
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(core::num::NonZeroU32::new(4).unwrap()),
        };
        let prepared = backend
            .prepare_dispatch_owned(submission, PcuInvocationParameters::empty())
            .unwrap();
        let target = PcuBindingRef::new(0, 0);
        let rejected_drop = Rc::new(Cell::new(false));
        let result = prepared.submit_owned([owned_binding(
            device(2),
            16,
            PcuBindingAccess::ReadOnly,
            Rc::clone(&rejected_drop),
        )]);
        let Err(error) = result else {
            panic!("prepared submit should reject a binding from another device");
        };
        assert!(matches!(
            error,
            crate::PcuOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::WrongDevice(binding)
            ) if binding == target
        ));
        assert!(rejected_drop.get());

        let accepted_drop = Rc::new(Cell::new(false));
        let completion = prepared
            .submit_owned([owned_binding(
                selected,
                16,
                PcuBindingAccess::ReadOnly,
                Rc::clone(&accepted_drop),
            )])
            .unwrap();
        assert!(!accepted_drop.get());
        drop(completion);
        assert!(accepted_drop.get());
    }

    struct QueueFixtureCompletion {
        outcome: Option<PcuCompletionOutcome>,
        query_fails: bool,
        wait_fails: bool,
        drops: Rc<Cell<u32>>,
    }

    impl Drop for QueueFixtureCompletion {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    impl PcuOwnedCompletion for QueueFixtureCompletion {
        type Error = &'static str;

        fn state(&self) -> Result<PcuCompletionState, Self::Error> {
            if self.query_fails {
                return Err("status unavailable");
            }
            Ok(match self.outcome {
                Some(PcuCompletionOutcome::Succeeded) => PcuCompletionState::Succeeded,
                Some(PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_)) => {
                    PcuCompletionState::Failed
                }
                None => PcuCompletionState::Running,
            })
        }

        fn wait(&mut self) -> Result<PcuCompletionOutcome, Self::Error> {
            if self.wait_fails {
                return Err("wait uncertain");
            }
            let outcome = self.outcome.unwrap_or(PcuCompletionOutcome::Succeeded);
            self.outcome = Some(outcome);
            Ok(outcome)
        }
    }

    struct QueueFixtureResource(Rc<Cell<u32>>);

    impl Drop for QueueFixtureResource {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn queue_quarantines_unknown_and_releases_after_terminal_wait() {
        let completion_drops = Rc::new(Cell::new(0));
        let resource_drops = Rc::new(Cell::new(0));
        {
            let mut queued = PcuOwnedSubmission::new(
                QueueFixtureCompletion {
                    outcome: None,
                    query_fails: true,
                    wait_fails: true,
                    drops: Rc::clone(&completion_drops),
                },
                QueueFixtureResource(Rc::clone(&resource_drops)),
            );
            assert_eq!(queued.status(), PcuQueueCompletionStatus::Unknown);
            assert!(queued.take_resources().is_none());
            assert_eq!(queued.wait(), Err("wait uncertain"));
        }
        assert_eq!(completion_drops.get(), 0);
        assert_eq!(resource_drops.get(), 0);

        let completion_drops = Rc::new(Cell::new(0));
        let resource_drops = Rc::new(Cell::new(0));
        {
            let mut queued = PcuOwnedSubmission::new(
                QueueFixtureCompletion {
                    outcome: None,
                    query_fails: false,
                    wait_fails: false,
                    drops: Rc::clone(&completion_drops),
                },
                QueueFixtureResource(Rc::clone(&resource_drops)),
            );
            assert_eq!(queued.status(), PcuQueueCompletionStatus::Running);
            assert_eq!(queued.wait(), Ok(PcuCompletionOutcome::Succeeded));
            assert_eq!(queued.status(), PcuQueueCompletionStatus::Complete);
            drop(queued.take_resources());
        }
        assert_eq!(completion_drops.get(), 1);
        assert_eq!(resource_drops.get(), 1);
    }

    #[test]
    fn queue_accepts_synchronous_terminal_failure_as_quiescent() {
        let completion_drops = Rc::new(Cell::new(0));
        let resource_drops = Rc::new(Cell::new(0));
        let mut queued = PcuOwnedSubmission::new(
            QueueFixtureCompletion {
                outcome: Some(PcuCompletionOutcome::Failed),
                query_fails: false,
                wait_fails: false,
                drops: Rc::clone(&completion_drops),
            },
            QueueFixtureResource(Rc::clone(&resource_drops)),
        );
        assert_eq!(queued.status(), PcuQueueCompletionStatus::Failed);
        drop(queued);
        assert_eq!(completion_drops.get(), 1);
        assert_eq!(resource_drops.get(), 1);
    }

    #[test]
    fn wait_result_distinguishes_backend_failure_generic_failure_and_fault() {
        let completion_drops = Rc::new(Cell::new(0));
        let resource_drops = Rc::new(Cell::new(0));
        let mut uncertain = PcuOwnedSubmission::new(
            QueueFixtureCompletion {
                outcome: None,
                query_fails: false,
                wait_fails: true,
                drops: Rc::clone(&completion_drops),
            },
            QueueFixtureResource(Rc::clone(&resource_drops)),
        );
        assert_eq!(
            uncertain.wait_result(),
            Err(PcuSubmissionWaitError::Backend("wait uncertain"))
        );
        assert!(uncertain.take_resources().is_none());
        drop(uncertain);
        assert_eq!(completion_drops.get(), 0);
        assert_eq!(resource_drops.get(), 0);

        let completion_drops = Rc::new(Cell::new(0));
        let resource_drops = Rc::new(Cell::new(0));
        let fault = crate::PcuExecutionFault {
            kind: crate::PcuExecutionFaultKind::DivideByZero,
            invocation_id: 17,
        };
        let mut faulted = PcuOwnedSubmission::new(
            QueueFixtureCompletion {
                outcome: Some(PcuCompletionOutcome::Fault(fault)),
                query_fails: false,
                wait_fails: false,
                drops: Rc::clone(&completion_drops),
            },
            QueueFixtureResource(Rc::clone(&resource_drops)),
        );
        assert_eq!(faulted.status(), PcuQueueCompletionStatus::Failed);
        assert_eq!(
            faulted.wait_result(),
            Err(PcuSubmissionWaitError::Fault(fault))
        );
        drop(faulted.take_resources());
        drop(faulted);
        assert_eq!(completion_drops.get(), 1);
        assert_eq!(resource_drops.get(), 1);

        let mut failed = PcuOwnedSubmission::new(
            QueueFixtureCompletion {
                outcome: Some(PcuCompletionOutcome::Failed),
                query_fails: false,
                wait_fails: false,
                drops: Rc::new(Cell::new(0)),
            },
            (),
        );
        assert_eq!(failed.wait_result(), Err(PcuSubmissionWaitError::Failed));
        assert_eq!(failed.status(), PcuQueueCompletionStatus::Failed);
    }

    #[test]
    fn wait_refreshes_status_only_failure_to_recover_structured_fault() {
        let completion_drops = Rc::new(Cell::new(0));
        let resource_drops = Rc::new(Cell::new(0));
        let fault = crate::PcuExecutionFault {
            kind: crate::PcuExecutionFaultKind::SignedDivisionOverflow,
            invocation_id: 29,
        };
        let mut queued = PcuOwnedSubmission::new(
            QueueFixtureCompletion {
                outcome: Some(PcuCompletionOutcome::Fault(fault)),
                query_fails: false,
                wait_fails: false,
                drops: Rc::clone(&completion_drops),
            },
            QueueFixtureResource(Rc::clone(&resource_drops)),
        );
        assert_eq!(queued.status(), PcuQueueCompletionStatus::Failed);
        assert_eq!(queued.wait(), Ok(PcuCompletionOutcome::Fault(fault)));
        assert_eq!(
            queued.wait_result(),
            Err(PcuSubmissionWaitError::Fault(fault))
        );
        drop(queued.take_resources());
        drop(queued);
        assert_eq!(completion_drops.get(), 1);
        assert_eq!(resource_drops.get(), 1);
    }
}
