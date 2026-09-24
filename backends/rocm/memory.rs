//! Bounded implementation of the abstract PCU memory contract for one selected HIP device.

use fusion_pcu::{
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryDisposition,
    PcuMemoryHostAccess,
    PcuMemoryImportDescriptor,
    PcuMemoryImportOwnership,
    PcuMemoryMapping,
    PcuMemoryPoolId,
    PcuMemoryPoolSnapshot,
    PcuMemoryProvider,
    PcuMemoryProviderError,
    PcuMemoryProviderFailure,
    PcuMemoryProviderOperation,
    PcuMemoryRange,
    PcuMemoryRequestError,
    PcuMemoryResource,
};

use crate::{
    DeviceBuffer,
    HipError,
    HipRuntime,
};

/// Provider bound to one runtime device and one caller-assigned abstract pool identity.
#[derive(Clone)]
pub struct RocmMemoryProvider {
    runtime: HipRuntime,
    pool: PcuMemoryPoolId,
}

impl RocmMemoryProvider {
    pub(crate) fn new(runtime: HipRuntime, pool: PcuMemoryPoolId) -> Self {
        Self { runtime, pool }
    }

    fn error(
        &self,
        operation: PcuMemoryProviderOperation,
        failure: PcuMemoryProviderFailure,
        disposition: PcuMemoryDisposition,
    ) -> PcuMemoryProviderError {
        PcuMemoryProviderError {
            pool: self.pool,
            operation,
            disposition,
            failure,
        }
    }

    fn validate_resource(
        &self,
        resource: &RocmMemoryResource,
        operation: PcuMemoryProviderOperation,
    ) -> Result<(), PcuMemoryProviderError> {
        if resource.pool != self.pool {
            return Err(self.error(
                operation,
                PcuMemoryProviderFailure::PoolUnavailable,
                PcuMemoryDisposition::Reject,
            ));
        }
        if self
            .runtime
            .ensure_same_runtime(&resource.buffer.allocation.runtime)
            .is_err()
        {
            return Err(self.error(
                operation,
                PcuMemoryProviderFailure::PoolUnavailable,
                PcuMemoryDisposition::Reject,
            ));
        }
        Ok(())
    }
}

/// Provider-owned HIP allocation. Fields remain private so pool and device identity cannot be
/// forged by consumers of the abstract memory API.
pub struct RocmMemoryResource {
    pool: PcuMemoryPoolId,
    buffer: DeviceBuffer,
    alignment: u64,
    access: PcuMemoryAccess,
}

impl PcuMemoryResource for RocmMemoryResource {
    fn pool(&self) -> PcuMemoryPoolId {
        self.pool
    }

    fn size_bytes(&self) -> u64 {
        self.buffer.len() as u64
    }

    fn alignment_bytes(&self) -> u64 {
        self.alignment
    }

    fn access(&self) -> PcuMemoryAccess {
        self.access
    }

    // HIP allocations are device-addressable, but HIP alone does not establish physical local
    // placement across discrete and unified-memory devices.
    fn is_device_local(&self) -> Option<bool> {
        None
    }
}

impl RocmMemoryResource {
    /// Borrow the underlying HIP allocation for direct ROCm PCU dispatch binding.
    ///
    /// The returned buffer uses the same shared busy gate as provider-mediated transfers, so
    /// overlapping synchronous access is rejected by the backend. The caller remains responsible
    /// for honoring this resource's declared [`PcuMemoryAccess`] when building a dispatch.
    pub fn device_buffer(&self) -> &DeviceBuffer {
        &self.buffer
    }
}

/// Marker import descriptor. ROCm import remains unavailable until a safe HIP/OS handle lease
/// protocol is implemented.
#[derive(Debug, Clone, Copy)]
pub struct RocmImportDescriptor {
    pub pool: PcuMemoryPoolId,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub access: PcuMemoryAccess,
    pub ownership: PcuMemoryImportOwnership,
}

impl PcuMemoryImportDescriptor for RocmImportDescriptor {
    fn pool(&self) -> PcuMemoryPoolId {
        self.pool
    }
    fn size_bytes(&self) -> u64 {
        self.size_bytes
    }
    fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }
    fn access(&self) -> PcuMemoryAccess {
        self.access
    }
    fn ownership(&self) -> PcuMemoryImportOwnership {
        self.ownership
    }
}

/// Unconstructible in normal use: the HIP provider always reports mapping unavailable.
pub struct RocmMemoryMapping(Vec<u8>);

impl PcuMemoryMapping for RocmMemoryMapping {
    fn as_bytes(&self) -> &[u8] {
        &self.0
    }
    fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        Some(&mut self.0)
    }
}

impl PcuMemoryProvider for RocmMemoryProvider {
    type Resource = RocmMemoryResource;
    type ImportDescriptor = RocmImportDescriptor;
    type Mapping<'a> = RocmMemoryMapping;

    fn snapshot(
        &self,
        pool: PcuMemoryPoolId,
    ) -> Result<PcuMemoryPoolSnapshot, PcuMemoryProviderError> {
        if pool != self.pool {
            return Err(self.error(
                PcuMemoryProviderOperation::Snapshot,
                PcuMemoryProviderFailure::PoolUnavailable,
                PcuMemoryDisposition::Reject,
            ));
        }
        self.runtime.memory_pool_snapshot(pool).map_err(|_| {
            self.error(
                PcuMemoryProviderOperation::Snapshot,
                PcuMemoryProviderFailure::BackendFailure,
                PcuMemoryDisposition::Reject,
            )
        })
    }

    fn allocate(
        &mut self,
        request: PcuMemoryAllocationRequest,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        let op = PcuMemoryProviderOperation::Allocate;
        if request.pool != self.pool {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::PoolUnavailable,
                PcuMemoryDisposition::Reject,
            ));
        }
        request.validate().map_err(|error| {
            self.error(
                op,
                PcuMemoryProviderFailure::InvalidRequest(error),
                PcuMemoryDisposition::Reject,
            )
        })?;
        if request.require_device_local {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::DeviceLocalRequired,
                PcuMemoryDisposition::Reject,
            ));
        }
        if request.host_access == PcuMemoryHostAccess::Mapping {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::MappingUnavailable,
                PcuMemoryDisposition::Reject,
            ));
        }
        let size = usize::try_from(request.size_bytes).map_err(|_| {
            self.error(
                op,
                PcuMemoryProviderFailure::BackendFailure,
                PcuMemoryDisposition::Reject,
            )
        })?;
        let alignment = usize::try_from(request.alignment_bytes).map_err(|_| {
            self.error(
                op,
                PcuMemoryProviderFailure::InvalidRequest(PcuMemoryRequestError::InvalidAlignment),
                PcuMemoryDisposition::Reject,
            )
        })?;
        let buffer = self
            .runtime
            .allocate(size)
            .map_err(|error| hip_failure(self, op, error))?;
        if (buffer.allocation.pointer as usize) % alignment != 0 {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::BackendFailure,
                PcuMemoryDisposition::Reject,
            ));
        }
        Ok(RocmMemoryResource {
            pool: self.pool,
            buffer,
            alignment: request.alignment_bytes,
            access: request.access,
        })
    }

    fn import(
        &mut self,
        _descriptor: Self::ImportDescriptor,
    ) -> Result<Self::Resource, PcuMemoryProviderError> {
        Err(self.error(
            PcuMemoryProviderOperation::Import,
            PcuMemoryProviderFailure::Unsupported,
            PcuMemoryDisposition::Reject,
        ))
    }

    fn map<'a>(
        &'a mut self,
        resource: &'a mut Self::Resource,
        range: PcuMemoryRange,
    ) -> Result<Self::Mapping<'a>, PcuMemoryProviderError> {
        self.validate_resource(resource, PcuMemoryProviderOperation::Map)?;
        if !range_fits(resource.buffer.len() as u64, range) {
            return Err(self.error(
                PcuMemoryProviderOperation::Map,
                PcuMemoryProviderFailure::RangeOutOfBounds,
                PcuMemoryDisposition::Reject,
            ));
        }
        Err(self.error(
            PcuMemoryProviderOperation::Map,
            PcuMemoryProviderFailure::MappingUnavailable,
            PcuMemoryDisposition::Reject,
        ))
    }

    fn transfer_to(
        &mut self,
        resource: &mut Self::Resource,
        offset_bytes: u64,
        bytes: &[u8],
    ) -> Result<(), PcuMemoryProviderError> {
        let op = PcuMemoryProviderOperation::TransferTo;
        self.validate_resource(resource, op)?;
        if !matches!(
            resource.access,
            PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::AccessDenied,
                PcuMemoryDisposition::Reject,
            ));
        }
        if !range_fits(
            resource.buffer.len() as u64,
            PcuMemoryRange {
                offset_bytes,
                size_bytes: bytes.len() as u64,
            },
        ) {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::RangeOutOfBounds,
                PcuMemoryDisposition::Reject,
            ));
        }
        let offset = usize::try_from(offset_bytes).map_err(|_| {
            self.error(
                op,
                PcuMemoryProviderFailure::RangeOutOfBounds,
                PcuMemoryDisposition::Reject,
            )
        })?;
        resource
            .buffer
            .copy_from_at(offset, bytes)
            .map_err(|error| hip_failure(self, op, error))
    }

    fn transfer_from(
        &mut self,
        resource: &Self::Resource,
        offset_bytes: u64,
        bytes: &mut [u8],
    ) -> Result<(), PcuMemoryProviderError> {
        let op = PcuMemoryProviderOperation::TransferFrom;
        self.validate_resource(resource, op)?;
        if !matches!(
            resource.access,
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ) {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::AccessDenied,
                PcuMemoryDisposition::Reject,
            ));
        }
        if !range_fits(
            resource.buffer.len() as u64,
            PcuMemoryRange {
                offset_bytes,
                size_bytes: bytes.len() as u64,
            },
        ) {
            return Err(self.error(
                op,
                PcuMemoryProviderFailure::RangeOutOfBounds,
                PcuMemoryDisposition::Reject,
            ));
        }
        let offset = usize::try_from(offset_bytes).map_err(|_| {
            self.error(
                op,
                PcuMemoryProviderFailure::RangeOutOfBounds,
                PcuMemoryDisposition::Reject,
            )
        })?;
        resource
            .buffer
            .copy_to_at(offset, bytes)
            .map_err(|error| hip_failure(self, op, error))
    }
}

fn range_fits(capacity: u64, range: PcuMemoryRange) -> bool {
    range.checked_end().is_some_and(|end| end <= capacity)
}

fn hip_failure(
    provider: &RocmMemoryProvider,
    operation: PcuMemoryProviderOperation,
    error: HipError,
) -> PcuMemoryProviderError {
    let (failure, disposition) = match error {
        HipError::Busy => (PcuMemoryProviderFailure::Busy, PcuMemoryDisposition::Defer),
        HipError::BufferTooSmall { .. } => (
            PcuMemoryProviderFailure::RangeOutOfBounds,
            PcuMemoryDisposition::Reject,
        ),
        _ => (
            PcuMemoryProviderFailure::BackendFailure,
            PcuMemoryDisposition::Reject,
        ),
    };
    provider.error(operation, failure, disposition)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn range_validation_rejects_overflow_and_past_end_without_gpu() {
        assert!(range_fits(
            16,
            PcuMemoryRange {
                offset_bytes: 8,
                size_bytes: 8
            }
        ));
        assert!(!range_fits(
            16,
            PcuMemoryRange {
                offset_bytes: 9,
                size_bytes: 8
            }
        ));
        assert!(!range_fits(
            u64::MAX,
            PcuMemoryRange {
                offset_bytes: u64::MAX,
                size_bytes: 1
            }
        ));
    }
}
