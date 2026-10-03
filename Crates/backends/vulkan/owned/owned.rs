//! Typed native carrier ownership. Storage support never grants arithmetic admission.
#[rustfmt::skip]
use core::marker::PhantomData;
#[rustfmt::skip]
use fusion_pcu::PcuScalar;
#[rustfmt::skip]
use crate::{
    ffi::{VulkanOwnedBuffer, VulkanPreparedOwnedCopy},
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
};

#[path = "bytes/bytes.rs"]
pub mod bytes;

/// Exclusive native Vulkan carrier buffer, retaining its exact logical-device session.
///
/// All twenty-two sealed scalar carriers preserve arbitrary raw bits. No host pointer or fake
/// generic device pointer is imported. Reads and deep copies finish before returning; dropping
/// the backend does not invalidate this owner. Unknown completion quarantines the whole session.
pub struct PcuVulkanOwnedBuffer<T: PcuScalar> {
    native: VulkanOwnedBuffer,
    len: usize,
    scalar: PhantomData<T>,
}

/// Cold fixed-extent carrier transfer resources with a reusable host upload allocation.
///
/// Each execution returns one freshly allocated native output buffer. It allocates no Rust
/// workspace, preserves all scalar bits and never authorizes floating or integer arithmetic.
pub struct PcuVulkanPreparedCarrierCopy<T: PcuScalar> {
    copy: VulkanPreparedOwnedCopy,
    upload: VulkanOwnedBuffer,
    len: usize,
    scalar: PhantomData<T>,
}

impl PcuVulkanBackend {
    /// Freezes carrier layout, extent, native upload storage and transfer submission resources.
    ///
    /// # Errors
    /// Rejects unsupported host ABI, extent overflow, quarantine or native preparation failures.
    pub fn prepare_carrier_copy<T: PcuScalar>(
        &self,
        len: usize,
    ) -> Result<PcuVulkanPreparedCarrierCopy<T>, PcuVulkanError> {
        if !cfg!(target_endian = "little") || T::HOST_SIZE != T::ENCODED_SIZE {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        let bytes = len
            .checked_mul(T::HOST_SIZE)
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        Ok(PcuVulkanPreparedCarrierCopy {
            copy: VulkanPreparedOwnedCopy::new(&self.device)?,
            upload: VulkanOwnedBuffer::new(&self.device, bytes)?,
            len,
            scalar: PhantomData,
        })
    }

    /// Uploads exact scalar bytes into a separately owned native Vulkan allocation.
    ///
    /// # Errors
    /// Rejects unsupported host layout, byte-size overflow, quarantine or native allocation errors.
    pub fn upload_owned<T: PcuScalar>(
        &self,
        input: &[T],
    ) -> Result<PcuVulkanOwnedBuffer<T>, PcuVulkanError> {
        if !cfg!(target_endian = "little") || T::HOST_SIZE != T::ENCODED_SIZE {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        let bytes = input
            .len()
            .checked_mul(T::HOST_SIZE)
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        let mut native = VulkanOwnedBuffer::new(&self.device, bytes)?;
        native.write(bytes::read(input))?;
        Ok(PcuVulkanOwnedBuffer {
            native,
            len: input.len(),
            scalar: PhantomData,
        })
    }

    /// Tests exact session affinity, including logical-device identity rather than just GPU UUID.
    #[must_use]
    pub fn owns_buffer<T: PcuScalar>(&self, buffer: &PcuVulkanOwnedBuffer<T>) -> bool {
        buffer.native.same_session(&self.device)
    }
}

impl<T: PcuScalar> PcuVulkanPreparedCarrierCopy<T> {
    /// Transfers a host prefix into a fresh native owner using the cold retained upload storage.
    ///
    /// # Errors
    /// Rejects a short input, quarantine or native allocation/submission errors before publication.
    pub fn copy_host(&mut self, input: &[T]) -> Result<PcuVulkanOwnedBuffer<T>, PcuVulkanError> {
        if input.len() < self.len {
            return Err(PcuVulkanError::BufferTooSmall);
        }
        self.upload.write(bytes::read(&input[..self.len]))?;
        Ok(PcuVulkanOwnedBuffer {
            native: self.copy.copy(&self.upload)?,
            len: self.len,
            scalar: PhantomData,
        })
    }

    /// Transfers an exact-extent owner directly on its logical device, without a host round trip.
    ///
    /// # Errors
    /// Rejects wrong extents or foreign logical sessions before submission, plus native failures.
    pub fn copy_owned(
        &mut self,
        input: &PcuVulkanOwnedBuffer<T>,
    ) -> Result<PcuVulkanOwnedBuffer<T>, PcuVulkanError> {
        if input.len != self.len {
            return Err(PcuVulkanError::InvalidArguments);
        }
        Ok(PcuVulkanOwnedBuffer {
            native: self.copy.copy(&input.native)?,
            len: self.len,
            scalar: PhantomData,
        })
    }
}

impl<T: PcuScalar> PcuVulkanOwnedBuffer<T> {
    pub(crate) const fn native(&self) -> &VulkanOwnedBuffer {
        &self.native
    }

    pub(crate) const fn native_mut(&mut self) -> &mut VulkanOwnedBuffer {
        &mut self.native
    }

    #[cfg(feature = "tensor")]
    pub(crate) fn from_native(
        native: VulkanOwnedBuffer,
        len: usize,
    ) -> Result<Self, PcuVulkanError> {
        if len.checked_mul(T::HOST_SIZE) != Some(native.byte_len()) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        Ok(Self {
            native,
            len,
            scalar: PhantomData,
        })
    }
    /// Checks terminal session availability without transferring or borrowing payload bytes.
    ///
    /// # Errors
    /// Rejects a session quarantined after uncertain native completion.
    pub fn validate_access_available(&self) -> Result<(), PcuVulkanError> {
        self.native.validate_access_available()
    }

    /// Number of logical scalar elements, excluding private word padding.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the logical value is empty; the native allocation may still own private padding.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Tests exact logical-device affinity across scalar representations.
    #[must_use]
    pub fn same_session<U: PcuScalar>(&self, other: &PcuVulkanOwnedBuffer<U>) -> bool {
        self.native.same_buffer_session(&other.native)
    }

    /// Reports the actual allocation memory properties; no device-local placement is inferred.
    #[must_use]
    pub const fn memory_realization(&self) -> PcuVulkanMemoryRealization {
        self.native.realization()
    }

    /// Copies the exact logical prefix into caller RAM, leaving an oversized output tail intact.
    ///
    /// # Errors
    /// Rejects a short output before publication and rejects quarantined sessions.
    pub fn read_into(&self, output: &mut [T]) -> Result<(), PcuVulkanError> {
        if output.len() < self.len {
            return Err(PcuVulkanError::BufferTooSmall);
        }
        self.native.read(bytes::write(&mut output[..self.len]))
    }

    /// Creates a distinct owner through a completed native buffer transfer.
    ///
    /// The result never aliases this allocation. This cold copy allocates native buffer and
    /// submission resources; it is not advertised as a prepared zero-overhead tensor executor.
    ///
    /// # Errors
    /// Returns quarantine, allocation, command-recording or terminal-submission errors.
    pub fn copy_owned(&self) -> Result<Self, PcuVulkanError> {
        Ok(Self {
            native: self.native.copy_owned()?,
            len: self.len,
            scalar: PhantomData,
        })
    }
}
