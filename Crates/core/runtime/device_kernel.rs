//! Typed device ownership and reusable resident kernel calls.

#[rustfmt::skip]
use core::{
    fmt,
    marker::PhantomData,
};

#[rustfmt::skip]
use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuMemoryAllocationRequest,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryRequestError,
    PcuScalar,
    PcuScalarType,
};

/// Error while allocating typed device storage through an existing memory provider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcuDeviceBufferAllocationError {
    ZeroElements,
    SizeOverflow,
    RequestTooSmall { required: u64, requested: u64 },
    InsufficientScalarAlignment { minimum: u64, requested: u64 },
    InvalidRequest(PcuMemoryRequestError),
    Provider(PcuMemoryProviderError),
    ResourceContractViolation,
}

impl fmt::Display for PcuDeviceBufferAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroElements => {
                f.write_str("typed device buffers must have at least one element")
            }
            Self::SizeOverflow => f.write_str("typed device buffer byte size overflows"),
            Self::RequestTooSmall {
                required,
                requested,
            } => write!(
                f,
                "typed device buffer needs {required} bytes, allocation request has {requested}"
            ),
            Self::InsufficientScalarAlignment { minimum, requested } => write!(
                f,
                "typed device buffer needs alignment {minimum}, request has {requested}"
            ),
            Self::InvalidRequest(error) => {
                write!(f, "invalid device allocation request: {error:?}")
            }
            Self::Provider(error) => {
                write!(f, "device memory provider allocation failed: {error:?}")
            }
            Self::ResourceContractViolation => f.write_str(
                "device memory provider returned storage outside its allocation request",
            ),
        }
    }
}

impl core::error::Error for PcuDeviceBufferAllocationError {}

/// Typed allocation over the existing provider allocation contract.
///
/// Providers are selected and scoped by the backend/session. This extension adds scalar extent
/// checking and a typed owner; it does not add a second allocator or imply that contents are
/// initialized. Admission policy and reservation ownership remain the caller's responsibility.
pub trait PcuDeviceBufferAllocator: PcuMemoryProvider {
    /// Allocates typed storage from a validated provider request without uploading or filling it.
    ///
    /// # Errors
    ///
    /// Returns a typed extent/request error, the provider's allocation error, or a resource
    /// contract violation when returned storage does not satisfy the request.
    fn allocate_device_buffer<T: PcuScalar>(
        &mut self,
        request: PcuMemoryAllocationRequest,
        elements: usize,
    ) -> Result<PcuDeviceBuffer<T, Self::Resource>, PcuDeviceBufferAllocationError>
    where
        Self: Sized,
    {
        if elements == 0 {
            return Err(PcuDeviceBufferAllocationError::ZeroElements);
        }
        let required = elements
            .checked_mul(T::ENCODED_SIZE)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or(PcuDeviceBufferAllocationError::SizeOverflow)?;
        let scalar_alignment = u64::try_from(T::ENCODED_SIZE)
            .map_err(|_| PcuDeviceBufferAllocationError::SizeOverflow)?;
        if request.alignment_bytes < scalar_alignment {
            return Err(
                PcuDeviceBufferAllocationError::InsufficientScalarAlignment {
                    minimum: scalar_alignment,
                    requested: request.alignment_bytes,
                },
            );
        }
        if request.size_bytes < required {
            return Err(PcuDeviceBufferAllocationError::RequestTooSmall {
                required,
                requested: request.size_bytes,
            });
        }
        request
            .validate()
            .map_err(PcuDeviceBufferAllocationError::InvalidRequest)?;
        let resource = self
            .allocate(request)
            .map_err(PcuDeviceBufferAllocationError::Provider)?;
        if !crate::resource::memory::resource_matches_request(&resource, request) {
            return Err(PcuDeviceBufferAllocationError::ResourceContractViolation);
        }
        Ok(PcuDeviceBuffer::new(resource, elements))
    }
}

impl<P: PcuMemoryProvider> PcuDeviceBufferAllocator for P {}

/// An owned backend resource with a scalar element type and logical length.
///
/// This carries device data, not a host slice or an implicit result cache. Backends validate its
/// real allocation, device identity and byte extent before use. It deliberately does not implement
/// Clone: ordinary Rust borrowing expresses read-only and exclusive mutable kernel arguments.
pub struct PcuDeviceBuffer<T: PcuScalar, R> {
    resource: R,
    len: usize,
    scalar: PhantomData<T>,
}

impl<T: PcuScalar, R> PcuDeviceBuffer<T, R> {
    /// Wraps owned backend storage. Declared length is checked against actual storage at execution.
    #[must_use]
    pub const fn new(resource: R, len: usize) -> Self {
        Self {
            resource,
            len,
            scalar: PhantomData,
        }
    }
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Backend storage, exposed for advanced integrations and validated again on use.
    #[must_use]
    pub const fn resource(&self) -> &R {
        &self.resource
    }
    /// Exclusive backend storage for provider updates; execution revalidates any replacement.
    #[must_use]
    pub const fn resource_mut(&mut self) -> &mut R {
        &mut self.resource
    }
    /// Recovers the owned backend storage.
    #[must_use]
    pub fn into_resource(self) -> R {
        self.resource
    }
}

/// A typed resource borrow for a synchronous device-resident call.
pub struct PcuDeviceArgument<'a, R> {
    target: PcuBindingRef,
    scalar: PcuScalarType,
    access: PcuBindingAccess,
    elements: usize,
    resource: &'a R,
}

impl<'a, R> PcuDeviceArgument<'a, R> {
    #[must_use]
    pub const fn read<T: PcuScalar>(
        target: PcuBindingRef,
        buffer: &'a PcuDeviceBuffer<T, R>,
    ) -> Self {
        Self {
            target,
            scalar: T::TYPE,
            access: PcuBindingAccess::ReadOnly,
            elements: buffer.len,
            resource: &buffer.resource,
        }
    }
    #[must_use]
    pub const fn read_write<T: PcuScalar>(
        target: PcuBindingRef,
        buffer: &'a mut PcuDeviceBuffer<T, R>,
    ) -> Self {
        Self {
            target,
            scalar: T::TYPE,
            access: PcuBindingAccess::ReadWrite,
            elements: buffer.len,
            resource: &buffer.resource,
        }
    }
    /// Creates an exclusive write-only device argument.
    ///
    /// A kernel must be admitted for write-only access before this argument may be submitted.
    /// This borrow does not establish that overwriting or aliasing the backing is semantically
    /// permitted by the operation.
    #[must_use]
    pub const fn write<T: PcuScalar>(
        target: PcuBindingRef,
        buffer: &'a mut PcuDeviceBuffer<T, R>,
    ) -> Self {
        Self {
            target,
            scalar: T::TYPE,
            access: PcuBindingAccess::WriteOnly,
            elements: buffer.len,
            resource: &buffer.resource,
        }
    }
    #[must_use]
    pub const fn target(&self) -> PcuBindingRef {
        self.target
    }
    #[must_use]
    pub const fn scalar(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn access(&self) -> PcuBindingAccess {
        self.access
    }
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }
    #[must_use]
    pub const fn resource(&self) -> &R {
        self.resource
    }
}

/// Backend-neutral compilation for typed device-resident calls.
pub trait PcuDeviceKernelBackend {
    type Resource;
    type Prepared: PcuPreparedDeviceKernel<Resource = Self::Resource>;
    type Error;
    /// Captures an executable independently of the temporary source IR.
    ///
    /// # Errors
    /// Returns unsupported capabilities, validation or compilation errors.
    fn prepare_device_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error>;
}

/// Reusable executable operating directly on typed device data.
pub trait PcuPreparedDeviceKernel {
    type Resource;
    type Error;
    /// Validates and completes a device-resident call without implicit host transfers.
    ///
    /// Every call validates actual storage, type, access, extent and device identity. Retain device
    /// resources through completion or quarantine. Complete before returning exclusive borrows;
    /// a subsequent kernel may then safely consume the updated buffer through ordinary Rust code.
    ///
    /// # Errors
    /// Returns admission, submission, completion or checked-operation errors.
    /// A failed kernel may have changed mutable buffers before its fault; this is not a
    /// transactional rollback contract. Uncertain completion must prevent unsafe resource reuse.
    fn call(
        &mut self,
        arguments: &mut [PcuDeviceArgument<'_, Self::Resource>],
    ) -> Result<(), Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_borrows_describe_ownership_without_copying_storage() {
        let mut buffer = PcuDeviceBuffer::<u64, _>::new([9_u8; 16], 2);
        {
            let argument = PcuDeviceArgument::read(PcuBindingRef::new(0, 3), &buffer);
            assert_eq!(argument.scalar(), PcuScalarType::U64);
            assert_eq!(argument.elements(), 2);
            assert_eq!(argument.access(), PcuBindingAccess::ReadOnly);
            assert!(core::ptr::eq(argument.resource(), buffer.resource()));
        }
        {
            let argument = PcuDeviceArgument::read_write(PcuBindingRef::new(0, 3), &mut buffer);
            assert_eq!(argument.access(), PcuBindingAccess::ReadWrite);
        }
        {
            let argument = PcuDeviceArgument::write(PcuBindingRef::new(0, 3), &mut buffer);
            assert_eq!(argument.access(), PcuBindingAccess::WriteOnly);
            assert_eq!(argument.scalar(), PcuScalarType::U64);
            assert_eq!(argument.elements(), 2);
        }
        assert_eq!(buffer.into_resource(), [9; 16]);
    }
}

#[cfg(test)]
mod allocation_tests {
    use super::{PcuDeviceBufferAllocationError, PcuDeviceBufferAllocator};
    use crate::{
        PcuMemoryAccess, PcuMemoryAllocationRequest, PcuMemoryDisposition, PcuMemoryHostAccess,
        PcuMemoryImportDescriptor, PcuMemoryImportOwnership, PcuMemoryMapping, PcuMemoryOverlap,
        PcuMemoryPoolId, PcuMemoryPoolSnapshot, PcuMemoryProvider, PcuMemoryProviderError,
        PcuMemoryProviderFailure, PcuMemoryProviderOperation, PcuMemoryRange, PcuMemoryResource,
        PcuMemoryResourceOrigin, PcuMemoryUsage,
    };

    struct FakeResource {
        request: PcuMemoryAllocationRequest,
    }

    impl PcuMemoryResource for FakeResource {
        fn pool(&self) -> PcuMemoryPoolId {
            self.request.pool
        }
        fn size_bytes(&self) -> u64 {
            self.request.size_bytes
        }
        fn alignment_bytes(&self) -> u64 {
            self.request.alignment_bytes
        }
        fn access(&self) -> PcuMemoryAccess {
            self.request.access
        }
        fn is_device_local(&self) -> Option<bool> {
            Some(self.request.require_device_local)
        }
        fn origin(&self) -> PcuMemoryResourceOrigin {
            PcuMemoryResourceOrigin::ProviderManaged
        }
        fn overlap(&self, other: &Self, _: PcuMemoryRange, _: PcuMemoryRange) -> PcuMemoryOverlap {
            if core::ptr::eq(self, other) {
                PcuMemoryOverlap::Overlapping
            } else {
                PcuMemoryOverlap::Disjoint
            }
        }
    }

    struct FakeImport;

    impl PcuMemoryImportDescriptor for FakeImport {
        fn pool(&self) -> PcuMemoryPoolId {
            PcuMemoryPoolId(1)
        }
        fn size_bytes(&self) -> u64 {
            1
        }
        fn alignment_bytes(&self) -> u64 {
            1
        }
        fn access(&self) -> PcuMemoryAccess {
            PcuMemoryAccess::ReadWrite
        }
        fn ownership(&self) -> PcuMemoryImportOwnership {
            PcuMemoryImportOwnership::Borrowed
        }
    }

    struct FakeMapping;

    impl PcuMemoryMapping for FakeMapping {
        fn as_bytes(&self) -> &[u8] {
            &[]
        }
        fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
            None
        }
    }

    #[derive(Default)]
    struct FakeProvider {
        calls: usize,
        last_request: Option<PcuMemoryAllocationRequest>,
    }

    impl FakeProvider {
        fn unsupported(
            pool: PcuMemoryPoolId,
            operation: PcuMemoryProviderOperation,
        ) -> PcuMemoryProviderError {
            PcuMemoryProviderError {
                pool,
                operation,
                disposition: PcuMemoryDisposition::Reject,
                failure: PcuMemoryProviderFailure::Unsupported,
            }
        }
    }

    impl PcuMemoryProvider for FakeProvider {
        type Resource = FakeResource;
        type ImportDescriptor = FakeImport;
        type Mapping<'a> = FakeMapping;

        fn snapshot(
            &self,
            pool: PcuMemoryPoolId,
        ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
            Ok(PcuMemoryPoolSnapshot {
                id: pool,
                capacity_bytes: None,
                system_used_bytes: PcuMemoryUsage::Unknown,
                process_used_bytes: PcuMemoryUsage::Unknown,
                system_ledger_reserved_bytes: 0,
                process_ledger_reserved_bytes: 0,
            })
        }

        fn allocate(
            &mut self,
            request: PcuMemoryAllocationRequest,
        ) -> Result<Self::Resource, PcuMemoryProviderError> {
            self.calls += 1;
            self.last_request = Some(request);
            Ok(FakeResource { request })
        }

        fn import(
            &mut self,
            descriptor: Self::ImportDescriptor,
        ) -> Result<Self::Resource, PcuMemoryProviderError> {
            let request = PcuMemoryAllocationRequest {
                pool: descriptor.pool(),
                size_bytes: descriptor.size_bytes(),
                alignment_bytes: descriptor.alignment_bytes(),
                access: descriptor.access(),
                host_access: PcuMemoryHostAccess::TransferOnly,
                require_device_local: false,
            };
            Ok(FakeResource { request })
        }

        fn map<'a>(
            &'a mut self,
            _: &'a mut Self::Resource,
            _: PcuMemoryRange,
        ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError> {
            Err(Self::unsupported(
                PcuMemoryPoolId(1),
                PcuMemoryProviderOperation::Map,
            ))
        }

        fn transfer_to(
            &mut self,
            _: &mut Self::Resource,
            _: u64,
            _: &[u8],
        ) -> Result<(), PcuMemoryProviderError> {
            Err(Self::unsupported(
                PcuMemoryPoolId(1),
                PcuMemoryProviderOperation::TransferTo,
            ))
        }

        fn transfer_from(
            &mut self,
            _: &Self::Resource,
            _: u64,
            _: &mut [u8],
        ) -> Result<(), PcuMemoryProviderError> {
            Err(Self::unsupported(
                PcuMemoryPoolId(1),
                PcuMemoryProviderOperation::TransferFrom,
            ))
        }
    }

    fn request(bytes: u64) -> PcuMemoryAllocationRequest {
        PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(7),
            size_bytes: bytes,
            alignment_bytes: 8,
            access: PcuMemoryAccess::WriteOnly,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: true,
        }
    }

    #[test]
    fn typed_output_allocation_uses_provider_request_without_transfer() {
        let mut provider = FakeProvider::default();
        let buffer = provider
            .allocate_device_buffer::<u64>(request(32), 4)
            .expect("provider-backed typed allocation");
        assert_eq!(buffer.len(), 4);
        assert_eq!(buffer.resource().size_bytes(), 32);
        assert_eq!(provider.calls, 1);
        assert_eq!(provider.last_request, Some(request(32)));
        assert_eq!(buffer.resource().access(), PcuMemoryAccess::WriteOnly);
    }

    #[test]
    fn typed_allocation_rejects_invalid_extent_before_provider_call() {
        let mut provider = FakeProvider::default();
        assert!(matches!(
            provider.allocate_device_buffer::<u64>(request(8), 2),
            Err(PcuDeviceBufferAllocationError::RequestTooSmall {
                required: 16,
                requested: 8,
            })
        ));
        assert_eq!(provider.calls, 0);

        assert!(matches!(
            provider.allocate_device_buffer::<u64>(request(u64::MAX), usize::MAX),
            Err(PcuDeviceBufferAllocationError::SizeOverflow)
        ));
        assert_eq!(provider.calls, 0);
    }
}
