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
    BufferTooSmall {
        binding: PcuBindingRef,
        required: u64,
        available: u64,
    },
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

    /// Returns the identity of the device/session selected by this backend instance.
    fn device_identity(&self) -> PcuDeviceIdentity;

    /// Submits a validated owned Dispatch operation.
    fn submit_dispatch_owned_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: Self::Bindings,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::Completion, Self::Error>;

    /// Performs common shape, parameter, support, device, and binding admission before submission.
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
}

/// Validates complete owned binding coverage and metadata for scalar value buffers.
///
/// This v1 sizing rule assumes a tightly packed, contiguous buffer with one scalar element per
/// logical thread. Vector, matrix, image, sampler, and acceleration-structure layouts are rejected
/// until a backend-specific layout contract can describe their actual storage size and stride.
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
        let Some(required) = bytes_per_element.checked_mul(u64::from(shape.thread_count().get()))
        else {
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
/// Both variants mean the operation is quiescent: the device no longer accesses resources owned
/// by the completion handle. Backend-specific diagnostics can be retained on the handle or
/// exposed through its error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCompletionOutcome {
    Succeeded,
    Failed,
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
    fn state(&self) -> Result<PcuCompletionState, Self::Error>;

    /// Waits for completion while preserving the handle on uncertain errors.
    fn wait(&mut self) -> Result<PcuCompletionOutcome, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::{
        PcuCompletionOutcome,
        PcuCompletionState,
        PcuDeviceIdentity,
        PcuOwnedBinding,
        PcuOwnedCompletion,
        PcuOwnedDispatchBackend,
        PcuOwnedDispatchBindingError,
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
        let shape = PcuInvocationShape::threads(core::num::NonZeroU32::new(4).unwrap());
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
                PcuInvocationShape::threads(core::num::NonZeroU32::new(4).unwrap()),
                device(1),
                &[binding],
            ),
            Err(PcuOwnedDispatchBindingError::UnsupportedLayout(
                PcuBindingRef::new(0, 0)
            ))
        );
    }

    impl PcuOwnedCompletion for FakeCompletion {
        type Error = &'static str;

        fn state(&self) -> Result<PcuCompletionState, Self::Error> {
            Ok(match self.terminal {
                Some(PcuCompletionOutcome::Succeeded) => PcuCompletionState::Succeeded,
                Some(PcuCompletionOutcome::Failed) => PcuCompletionState::Failed,
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
            shape: PcuInvocationShape::threads(core::num::NonZeroU32::new(4).unwrap()),
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
}
