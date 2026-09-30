//! Word-aligned shared Metal resources under the neutral memory contract.

#[rustfmt::skip]
use std::{
    cell::RefCell,
    rc::Rc,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryBackingOwnership,
    PcuMemoryDisposition,
    PcuMemoryHostAccess,
    PcuMemoryImportDescriptor,
    PcuMemoryImportOwnership,
    PcuMemoryMapping,
    PcuMemoryOverlap,
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryProviderFailure,
    PcuMemoryProviderOperation,
    PcuMemoryRange,
    PcuMemoryResource,
    PcuMemoryResourceCapability,
    PcuMemoryResourceOrigin,
    PcuMemoryUsage,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalSession,
};

/// One session-affine pool of shared allocations. Physical residency/accounting remain unknown.
#[derive(Clone)]
pub struct MetalMemoryProvider {
    session: MetalSession,
    pool: PcuMemoryPoolId,
}
impl MetalSession {
    #[must_use]
    pub fn memory_provider(&self, pool: PcuMemoryPoolId) -> MetalMemoryProvider {
        MetalMemoryProvider {
            session: self.clone(),
            pool,
        }
    }
}
impl MetalMemoryProvider {
    const fn error(
        &self,
        operation: PcuMemoryProviderOperation,
        failure: PcuMemoryProviderFailure,
    ) -> PcuMemoryProviderError {
        PcuMemoryProviderError {
            pool: self.pool,
            operation,
            disposition: PcuMemoryDisposition::Reject,
            failure,
        }
    }
    fn validate_resource(
        &self,
        resource: &MetalMemoryResource,
        operation: PcuMemoryProviderOperation,
    ) -> Result<(), PcuMemoryProviderError> {
        if resource.pool != self.pool
            || !self
                .session
                .same_session(resource.buffer.borrow().session())
        {
            return Err(self.error(operation, PcuMemoryProviderFailure::PoolUnavailable));
        }
        self.session
            .ensure_quiescent()
            .map_err(|_| self.error(operation, PcuMemoryProviderFailure::BackendFailure))
    }
}

/// Independently owned shared allocation, retaining its native session affinity.
pub struct MetalMemoryResource {
    pool: PcuMemoryPoolId,
    buffer: Rc<RefCell<MetalBuffer>>,
    alignment: u64,
    access: PcuMemoryAccess,
}
impl MetalMemoryResource {
    /// Validates that terminal completion still permits safe use of this allocation.
    ///
    /// # Errors
    /// Returns the retained session's quarantine error after uncertain native completion.
    pub fn validate_access_available(&self) -> Result<(), crate::MetalError> {
        self.buffer.borrow().session().ensure_quiescent()
    }
    pub(crate) fn lease(&self) -> Rc<RefCell<MetalBuffer>> {
        Rc::clone(&self.buffer)
    }
}
impl PcuMemoryResource for MetalMemoryResource {
    fn pool(&self) -> PcuMemoryPoolId {
        self.pool
    }
    fn size_bytes(&self) -> u64 {
        self.buffer.borrow().len() as u64 * 4
    }
    fn alignment_bytes(&self) -> u64 {
        self.alignment
    }
    fn access(&self) -> PcuMemoryAccess {
        self.access
    }
    fn is_device_local(&self) -> Option<bool> {
        None
    }
    fn origin(&self) -> PcuMemoryResourceOrigin {
        PcuMemoryResourceOrigin::ProviderManaged
    }
    fn supports(&self, capability: PcuMemoryResourceCapability) -> bool {
        capability == PcuMemoryResourceCapability::ReusableStorage
    }
    fn backing_ownership(&self) -> PcuMemoryBackingOwnership {
        if Rc::strong_count(&self.buffer) == 1 {
            PcuMemoryBackingOwnership::Exclusive
        } else {
            PcuMemoryBackingOwnership::Shared
        }
    }
    fn overlap(
        &self,
        other: &Self,
        left: PcuMemoryRange,
        right: PcuMemoryRange,
    ) -> PcuMemoryOverlap {
        if left.checked_end().is_none_or(|end| end > self.size_bytes())
            || right
                .checked_end()
                .is_none_or(|end| end > other.size_bytes())
        {
            return PcuMemoryOverlap::Unknown;
        }
        if std::ptr::eq(self, other)
            && left.offset_bytes < right.checked_end().unwrap_or(0)
            && right.offset_bytes < left.checked_end().unwrap_or(0)
        {
            PcuMemoryOverlap::Overlapping
        } else {
            PcuMemoryOverlap::Disjoint
        }
    }
}

/// External import descriptor has no public constructor; imports remain unsupported.
pub struct MetalMemoryImport {
    pool: PcuMemoryPoolId,
    size: u64,
    alignment: u64,
    access: PcuMemoryAccess,
}
impl PcuMemoryImportDescriptor for MetalMemoryImport {
    fn pool(&self) -> PcuMemoryPoolId {
        self.pool
    }
    fn size_bytes(&self) -> u64 {
        self.size
    }
    fn alignment_bytes(&self) -> u64 {
        self.alignment
    }
    fn access(&self) -> PcuMemoryAccess {
        self.access
    }
    fn ownership(&self) -> PcuMemoryImportOwnership {
        PcuMemoryImportOwnership::Borrowed
    }
}
/// Scoped mapping is unsupported; mediated transfers retain all ownership checks.
pub struct MetalMemoryMapping {
    bytes: Vec<u8>,
}
impl PcuMemoryMapping for MetalMemoryMapping {
    fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
    fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.bytes)
    }
}
impl PcuMemoryProvider for MetalMemoryProvider {
    type Resource = MetalMemoryResource;
    type ImportDescriptor = MetalMemoryImport;
    type Mapping<'a> = MetalMemoryMapping;
    fn snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
        let operation = PcuMemoryProviderOperation::Snapshot;
        if pool != self.pool {
            return Err(self.error(operation, PcuMemoryProviderFailure::PoolUnavailable));
        }
        self.session
            .ensure_quiescent()
            .map_err(|_| self.error(operation, PcuMemoryProviderFailure::BackendFailure))?;
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
    ) -> Result<MetalMemoryResource, PcuMemoryProviderError> {
        let operation = PcuMemoryProviderOperation::Allocate;
        if request.pool != self.pool {
            return Err(self.error(operation, PcuMemoryProviderFailure::PoolUnavailable));
        }
        request.validate().map_err(|error| {
            self.error(operation, PcuMemoryProviderFailure::InvalidRequest(error))
        })?;
        if request.require_device_local {
            return Err(self.error(operation, PcuMemoryProviderFailure::DeviceLocalRequired));
        }
        if request.host_access == PcuMemoryHostAccess::Mapping {
            return Err(self.error(operation, PcuMemoryProviderFailure::MappingUnavailable));
        }
        if !request.size_bytes.is_multiple_of(4) || request.alignment_bytes > 4 {
            return Err(self.error(operation, PcuMemoryProviderFailure::Unsupported));
        }
        let words = usize::try_from(request.size_bytes / 4)
            .map_err(|_| self.error(operation, PcuMemoryProviderFailure::RangeOutOfBounds))?;
        if request.size_bytes > self.session.facts().max_buffer_bytes
            || u32::try_from(words).is_err()
        {
            return Err(self.error(operation, PcuMemoryProviderFailure::RangeOutOfBounds));
        }
        self.session
            .ensure_quiescent()
            .map_err(|_| self.error(operation, PcuMemoryProviderFailure::BackendFailure))?;
        // Concrete initialization proof for the facade's sealed ownership adapter: every byte
        // is zero-cleared by Metal newBufferWithLength:options:, fields are private, and
        // external resource imports reject. Apple documents this initialization guarantee.
        // The generic PcuDeviceBufferAllocator contract itself permits uninitialized storage.
        let buffer = self
            .session
            .allocate_zeroed(words)
            .map_err(|error| self.error(operation, failure(&error)))?;
        Ok(MetalMemoryResource {
            pool: self.pool,
            buffer: Rc::new(RefCell::new(buffer)),
            alignment: request.alignment_bytes,
            access: request.access,
        })
    }
    fn import(
        &mut self,
        _: MetalMemoryImport,
    ) -> Result<MetalMemoryResource, PcuMemoryProviderError> {
        Err(self.error(
            PcuMemoryProviderOperation::Import,
            PcuMemoryProviderFailure::Unsupported,
        ))
    }
    fn map<'a>(
        &'a mut self,
        resource: &'a mut MetalMemoryResource,
        _: PcuMemoryRange,
    ) -> Result<MetalMemoryMapping, PcuMemoryProviderError> {
        let operation = PcuMemoryProviderOperation::Map;
        self.validate_resource(resource, operation)?;
        Err(self.error(operation, PcuMemoryProviderFailure::MappingUnavailable))
    }
    fn transfer_to(
        &mut self,
        resource: &mut MetalMemoryResource,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<(), PcuMemoryProviderError> {
        let operation = PcuMemoryProviderOperation::TransferTo;
        self.validate_resource(resource, operation)?;
        if resource.access == PcuMemoryAccess::ReadOnly {
            return Err(self.error(operation, PcuMemoryProviderFailure::AccessDenied));
        }
        let offset = offset(resource.size_bytes(), offset_bytes, bytes.len())
            .map_err(|failure| self.error(operation, failure))?;
        resource
            .buffer
            .borrow_mut()
            .write_bytes(offset, bytes)
            .map_err(|error| self.error(operation, failure(&error)))
    }
    fn transfer_from(
        &mut self,
        resource: &MetalMemoryResource,
        offset_bytes: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError> {
        let operation = PcuMemoryProviderOperation::TransferFrom;
        self.validate_resource(resource, operation)?;
        if resource.access == PcuMemoryAccess::WriteOnly {
            return Err(self.error(operation, PcuMemoryProviderFailure::AccessDenied));
        }
        let offset = offset(resource.size_bytes(), offset_bytes, bytes.len())
            .map_err(|failure| self.error(operation, failure))?;
        resource
            .buffer
            .borrow()
            .read_bytes(offset, bytes)
            .map_err(|error| self.error(operation, failure(&error)))
    }
}
fn offset(size: u64, offset: u64, length: usize) -> Result<usize, PcuMemoryProviderFailure> {
    let length = u64::try_from(length).map_err(|_| PcuMemoryProviderFailure::RangeOutOfBounds)?;
    if offset.checked_add(length).is_none_or(|end| end > size) {
        return Err(PcuMemoryProviderFailure::RangeOutOfBounds);
    }
    usize::try_from(offset).map_err(|_| PcuMemoryProviderFailure::RangeOutOfBounds)
}
const fn failure(error: &MetalError) -> PcuMemoryProviderFailure {
    match error {
        MetalError::Unsupported => PcuMemoryProviderFailure::Unsupported,
        MetalError::InvalidExtent => PcuMemoryProviderFailure::RangeOutOfBounds,
        _ => PcuMemoryProviderFailure::BackendFailure,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn byte_transfer_range_checks_overflow_and_boundary_without_pointer_access() {
        assert_eq!(offset(8, 8, 0), Ok(8));
        assert_eq!(offset(8, 7, 1), Ok(7));
        assert_eq!(
            offset(8, 7, 2),
            Err(PcuMemoryProviderFailure::RangeOutOfBounds)
        );
        assert_eq!(
            offset(u64::MAX, u64::MAX, 1),
            Err(PcuMemoryProviderFailure::RangeOutOfBounds)
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    #[ignore = "Requires actual macOS Metal allocation/transfer API."]
    #[allow(
        clippy::too_many_lines,
        reason = "One lifecycle fixture retains the same allocation across transfer, rejection and escaped-owner checks."
    )]
    fn neutral_shared_memory_partial_transfer_affinity_access_and_ownership() {
        let session = MetalSession::open(0).unwrap();
        let pool = PcuMemoryPoolId(7);
        let mut provider = session.memory_provider(pool);
        let request = PcuMemoryAllocationRequest {
            pool,
            size_bytes: 8,
            alignment_bytes: 4,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        };
        let mut resource = provider.allocate(request).unwrap();
        assert_eq!(
            resource.backing_ownership(),
            PcuMemoryBackingOwnership::Exclusive
        );
        assert_eq!(resource.is_device_local(), None);
        provider
            .transfer_to(&mut resource, 1, &[1, 2, 3, 4, 5])
            .unwrap();
        let mut bytes = [99_u8; 8];
        provider.transfer_from(&resource, 0, &mut bytes).unwrap();
        assert_eq!(bytes, [0, 1, 2, 3, 4, 5, 0, 0]);
        assert_eq!(
            resource.overlap(
                &resource,
                PcuMemoryRange {
                    offset_bytes: 0,
                    size_bytes: 4
                },
                PcuMemoryRange {
                    offset_bytes: 4,
                    size_bytes: 4
                }
            ),
            PcuMemoryOverlap::Disjoint
        );
        assert_eq!(
            resource.overlap(
                &resource,
                PcuMemoryRange {
                    offset_bytes: 0,
                    size_bytes: 5
                },
                PcuMemoryRange {
                    offset_bytes: 4,
                    size_bytes: 4
                }
            ),
            PcuMemoryOverlap::Overlapping
        );
        assert_eq!(
            resource.overlap(
                &resource,
                PcuMemoryRange {
                    offset_bytes: 0,
                    size_bytes: 9
                },
                PcuMemoryRange {
                    offset_bytes: 0,
                    size_bytes: 1
                }
            ),
            PcuMemoryOverlap::Unknown
        );
        assert_eq!(
            provider
                .transfer_to(&mut resource, 7, &[1, 2])
                .unwrap_err()
                .failure,
            PcuMemoryProviderFailure::RangeOutOfBounds
        );
        let mut other = MetalSession::open(0).unwrap().memory_provider(pool);
        assert_eq!(
            other
                .transfer_from(&resource, 0, &mut bytes)
                .unwrap_err()
                .failure,
            PcuMemoryProviderFailure::PoolUnavailable
        );
        let mut readonly = provider
            .allocate(PcuMemoryAllocationRequest {
                access: PcuMemoryAccess::ReadOnly,
                ..request
            })
            .unwrap();
        assert_eq!(
            provider
                .transfer_to(&mut readonly, 0, &[1])
                .unwrap_err()
                .failure,
            PcuMemoryProviderFailure::AccessDenied
        );
        let writeonly = provider
            .allocate(PcuMemoryAllocationRequest {
                access: PcuMemoryAccess::WriteOnly,
                ..request
            })
            .unwrap();
        assert_eq!(
            provider
                .transfer_from(&writeonly, 0, &mut bytes)
                .unwrap_err()
                .failure,
            PcuMemoryProviderFailure::AccessDenied
        );
        assert_eq!(
            provider
                .allocate(PcuMemoryAllocationRequest {
                    size_bytes: 6,
                    ..request
                })
                .err()
                .unwrap()
                .failure,
            PcuMemoryProviderFailure::Unsupported
        );
        assert_eq!(
            provider
                .allocate(PcuMemoryAllocationRequest {
                    require_device_local: true,
                    ..request
                })
                .err()
                .unwrap()
                .failure,
            PcuMemoryProviderFailure::DeviceLocalRequired
        );
        let snapshot = provider.snapshot(pool).unwrap();
        assert_eq!(snapshot.capacity_bytes, None);
        assert_eq!(snapshot.process_used_bytes, PcuMemoryUsage::Unknown);
        drop(session);
        provider.transfer_from(&resource, 0, &mut bytes).unwrap();
        assert_eq!(bytes, [0, 1, 2, 3, 4, 5, 0, 0]);
    }
}
