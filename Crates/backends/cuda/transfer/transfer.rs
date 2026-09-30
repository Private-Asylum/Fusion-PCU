//! Owned pinned host staging with completion-protected CPU access.

#[rustfmt::skip]
use std::{
    any::Any,
    ffi::c_void,
    ptr::NonNull,
    rc::Rc,
};

#[rustfmt::skip]
use crate::{
    CudaBatchCompletion,
    CudaCompletionBatch,
    CudaError,
    CudaRuntime,
    CudaStreamHandle,
    DeviceBuffer,
    ensure_batch_open,
    validate_buffer_range,
    ffi::require_free_host,
    ffi::runtime::{
        CUDA_MEMCPY_DEVICE_TO_HOST,
        CUDA_MEMCPY_HOST_TO_DEVICE,
    },
};

struct PinnedAllocation {
    runtime: CudaRuntime,
    pointer: NonNull<u8>,
    bytes: usize,
}

impl Drop for PinnedAllocation {
    fn drop(&mut self) {
        if self.bytes != 0 {
            // SAFETY: this allocation is exclusively owned here after every retained DMA owner
            // has been released. Runtime ownership keeps the library loaded through this call.
            // An activation/free error deliberately leaks the pinned allocation.
            let _ = unsafe {
                crate::ffi::invoke_cudaFreeHost(&self.runtime, self.pointer.as_ptr().cast())
            };
        }
    }
}

/// Initialized, exclusively accessible pinned host bytes owned by one CUDA runtime.
///
/// Upload and download consume this handle. The underlying allocation is retained by the
/// completion, and CPU access to a download is unavailable until a successful wait. No borrowed
/// host pointer or shared mutable CPU view is admitted.
pub struct CudaPinnedBuffer {
    allocation: Rc<PinnedAllocation>,
}

impl CudaRuntime {
    /// Allocate zero-initialized pinned host staging.
    ///
    /// Empty staging is a validated allocation-free buffer. Nonempty allocations are bounded by
    /// Rust's maximum slice length and use `cudaMallocHost`, with no pageable fallback.
    ///
    /// # Errors
    ///
    /// Returns an invalid length, unavailable pinned allocation/free symbol, or CUDA error.
    pub fn allocate_pinned(&self, bytes: usize) -> Result<CudaPinnedBuffer, CudaError> {
        validate_pinned_length(bytes)?;
        let mut pointer = std::ptr::null_mut::<c_void>();
        if bytes != 0 {
            // Resolve deallocation before allocating storage. This probe has no GPU side effect.
            require_free_host(&self.0.library)?;
            // SAFETY: CUDA initializes the pointer to a live allocation of the requested size.
            unsafe { crate::ffi::invoke_cudaMallocHost(self, &raw mut pointer, bytes) }?;
        }
        let pointer = if bytes == 0 {
            NonNull::dangling()
        } else {
            NonNull::new(pointer.cast::<u8>()).ok_or_else(|| CudaError::Runtime {
                operation: "cudaMallocHost",
                code: -1,
                detail: Some("successful pinned allocation returned a null pointer".into()),
            })?
        };
        // SAFETY: nonempty storage is newly allocated, and the empty pointer is aligned/dangling.
        unsafe { pointer.as_ptr().write_bytes(0, bytes) };
        Ok(CudaPinnedBuffer {
            allocation: Rc::new(PinnedAllocation {
                runtime: self.clone(),
                pointer,
                bytes,
            }),
        })
    }
}

const fn validate_pinned_length(bytes: usize) -> Result<(), CudaError> {
    if bytes > isize::MAX as usize {
        return Err(CudaError::BufferTooSmall {
            allocation: isize::MAX as usize,
            requested: bytes,
        });
    }
    Ok(())
}

impl CudaPinnedBuffer {
    /// Return the initialized bytes before submitting a transfer.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: allocation is initialized and retained, and this unique public owner has not
        // been consumed by a transfer. No asynchronous writer can exist during this borrow.
        unsafe {
            std::slice::from_raw_parts(self.allocation.pointer.as_ptr(), self.allocation.bytes)
        }
    }

    /// Mutably borrow initialized staging before submitting a transfer.
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the public handle is not Clone; completion retention only starts after this
        // handle is consumed, so the mutable borrow cannot coexist with any DMA operation.
        unsafe {
            std::slice::from_raw_parts_mut(self.allocation.pointer.as_ptr(), self.allocation.bytes)
        }
    }

    /// Consume staging and queue its bytes for upload on `stream`.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, destination range/busy error, or CUDA submission error.
    pub fn upload(
        self,
        destination: &DeviceBuffer,
        stream: &CudaStreamHandle,
    ) -> Result<CudaBatchCompletion, CudaError> {
        let mut batch = CudaCompletionBatch::new(stream);
        batch.copy_pinned_host_to_device(destination, 0, self)?;
        batch.finish()
    }

    /// Consume staging for upload while retaining it for reuse after successful completion.
    ///
    /// # Errors
    ///
    /// Returns the same runtime, range, busy, or submission errors as [`Self::upload`].
    pub fn upload_reusable(
        self,
        destination: &DeviceBuffer,
        stream: &CudaStreamHandle,
    ) -> Result<CudaPinnedUpload, CudaError> {
        let staging = Self {
            allocation: Rc::clone(&self.allocation),
        };
        Ok(CudaPinnedUpload {
            completion: staging.upload(destination, stream)?,
            buffer: self,
        })
    }

    /// Consume staging and queue a device download of exactly its length.
    ///
    /// # Errors
    ///
    /// Returns a runtime mismatch, source range/busy error, or CUDA submission error.
    pub fn download(
        self,
        source: &DeviceBuffer,
        stream: &CudaStreamHandle,
    ) -> Result<CudaPinnedDownload, CudaError> {
        self.allocation
            .runtime
            .ensure_same_runtime(&stream.inner.runtime)?;
        stream
            .inner
            .runtime
            .ensure_same_runtime(&source.allocation.runtime)?;
        source.check_range(0, self.allocation.bytes)?;
        let mut batch = CudaCompletionBatch::new(stream);
        if self.allocation.bytes != 0 {
            let owner: Rc<dyn Any> = self.allocation.clone();
            batch.retain_external_operation(&[source], owner)?;
            // SAFETY: exclusive host storage and source lease are retained before submission.
            let enqueue = unsafe {
                crate::ffi::invoke_cudaMemcpyAsync(
                    &stream.inner.runtime,
                    self.allocation.pointer.as_ptr().cast(),
                    source.allocation.pointer,
                    self.allocation.bytes,
                    CUDA_MEMCPY_DEVICE_TO_HOST,
                    stream.inner.raw,
                )
            };
            if let Err(error) = enqueue {
                batch.failed = true;
                batch.release_after_stream_sync();
                return Err(error);
            }
        }
        Ok(CudaPinnedDownload {
            completion: batch.finish()?,
            buffer: self,
            complete: false,
        })
    }
}

impl CudaCompletionBatch {
    /// Consume owned pinned staging and queue an upload at the checked device offset.
    ///
    /// Staging and destination remain retained through the batch's terminal completion. Empty
    /// uploads validate runtime/range but do not acquire device access.
    ///
    /// # Errors
    ///
    /// Returns a poisoned batch, runtime mismatch, destination range/busy or CUDA enqueue error.
    pub fn copy_pinned_host_to_device(
        &mut self,
        destination: &DeviceBuffer,
        offset: usize,
        source: CudaPinnedBuffer,
    ) -> Result<(), CudaError> {
        let CudaPinnedBuffer { allocation } = source;
        ensure_batch_open(self.failed)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&allocation.runtime)?;
        self.stream
            .inner
            .runtime
            .ensure_same_runtime(&destination.allocation.runtime)?;
        validate_buffer_range(destination.allocation.bytes, offset, allocation.bytes)?;
        if allocation.bytes == 0 {
            return Ok(());
        }
        let owner: Rc<dyn Any> = allocation.clone();
        self.retain_external_operation(&[destination], owner)?;
        // SAFETY: source is consumed and retained; the destination lease is held through final
        // completion. Checked ranges and the selected runtime establish pointer validity.
        let enqueue = unsafe {
            crate::ffi::invoke_cudaMemcpyAsync(
                &self.stream.inner.runtime,
                destination
                    .allocation
                    .pointer
                    .cast::<u8>()
                    .wrapping_add(offset)
                    .cast(),
                allocation.pointer.as_ptr().cast(),
                allocation.bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream.inner.raw,
            )
        };
        if let Err(error) = enqueue {
            self.failed = true;
            self.release_after_stream_sync();
            return Err(error);
        }
        Ok(())
    }
}

/// An upload retaining exclusive pinned staging until terminal completion.
///
/// CPU access becomes available only by consuming this owner with [`Self::finish`].
pub struct CudaPinnedUpload {
    completion: CudaBatchCompletion,
    buffer: CudaPinnedBuffer,
}

impl CudaPinnedUpload {
    /// Wait for the complete upload and reclaim its staging for CPU access or another transfer.
    ///
    /// # Errors
    ///
    /// Returns the CUDA wait error. Failed completion quarantines staging before release.
    pub fn finish(mut self) -> Result<CudaPinnedBuffer, CudaError> {
        self.completion.wait()?;
        Ok(self.buffer)
    }
}

/// A pinned download whose CPU bytes are gated by successful terminal completion.
pub struct CudaPinnedDownload {
    // Drop completion before the public buffer. Completion retains a second allocation owner
    // and quarantines that owner if stream synchronization cannot prove quiescence.
    completion: CudaBatchCompletion,
    buffer: CudaPinnedBuffer,
    complete: bool,
}

impl CudaPinnedDownload {
    /// Wait for the download; failed waits keep storage retained and permit a retry.
    ///
    /// # Errors
    ///
    /// Returns the terminal CUDA synchronization error.
    pub fn wait(&mut self) -> Result<(), CudaError> {
        self.completion.wait()?;
        self.complete = true;
        Ok(())
    }

    /// Wait successfully and reclaim downloaded staging for CPU access or another transfer.
    ///
    /// # Errors
    ///
    /// Returns the CUDA wait error. Failed completion retains or quarantines DMA storage.
    pub fn finish(mut self) -> Result<CudaPinnedBuffer, CudaError> {
        self.wait()?;
        Ok(self.buffer)
    }

    /// Borrow initialized downloaded bytes after a successful wait.
    ///
    /// # Errors
    ///
    /// Returns [`CudaError::BatchNotComplete`] until the complete download is proven.
    pub fn as_bytes(&self) -> Result<&[u8], CudaError> {
        require_download_complete(self.complete)?;
        Ok(self.buffer.as_bytes())
    }
}

const fn require_download_complete(complete: bool) -> Result<(), CudaError> {
    if complete {
        Ok(())
    } else {
        Err(CudaError::BatchNotComplete)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_slice_size_is_bounded_before_cuda_allocation() {
        assert!(validate_pinned_length(0).is_ok());
        assert!(validate_pinned_length(isize::MAX as usize).is_ok());
        assert!(matches!(
            validate_pinned_length(usize::MAX),
            Err(CudaError::BufferTooSmall { .. })
        ));
    }

    #[test]
    fn downloaded_bytes_require_a_successful_wait() {
        assert_eq!(
            require_download_complete(false),
            Err(CudaError::BatchNotComplete)
        );
        assert_eq!(require_download_complete(true), Ok(()));
    }

    #[test]
    #[ignore = "requires a CUDA device and pinned host allocation support"]
    fn pinned_upload_download_and_offset_preserve_bytes() {
        let runtime = CudaRuntime::new(0).expect("CUDA runtime");
        let stream = runtime.create_stream().expect("stream");
        let device = runtime.allocate(16).expect("device");
        let mut upload = runtime.allocate_pinned(16).expect("pinned upload");
        upload.as_bytes_mut().copy_from_slice(b"pinned DMA bytes");
        let mut completion = upload.upload(&device, &stream).expect("upload");
        completion.wait().expect("upload completion");
        let mut download = runtime
            .allocate_pinned(16)
            .expect("pinned download")
            .download(&device, &stream)
            .expect("download");
        assert_eq!(download.as_bytes(), Err(CudaError::BatchNotComplete));
        download.wait().expect("download completion");
        assert_eq!(
            download.as_bytes().expect("completed bytes"),
            b"pinned DMA bytes"
        );
        let mut reusable = runtime.allocate_pinned(16).expect("reused staging");
        for marker in [13_u8, 241] {
            reusable.as_bytes_mut().fill(marker);
            reusable = reusable
                .upload_reusable(&device, &stream)
                .expect("reused upload")
                .finish()
                .expect("reclaim upload staging");
            reusable = reusable
                .download(&device, &stream)
                .expect("reused download")
                .finish()
                .expect("reclaim downloaded staging");
            assert_eq!(reusable.as_bytes(), &[marker; 16]);
        }
        let mut replacement = runtime.allocate_pinned(3).expect("replacement");
        replacement.as_bytes_mut().copy_from_slice(b"XYZ");
        let mut batch = CudaCompletionBatch::new(&stream);
        batch
            .copy_pinned_host_to_device(&device, 7, replacement)
            .expect("offset upload");
        batch
            .finish()
            .expect("finish")
            .wait()
            .expect("offset completion");
        let mut download = runtime
            .allocate_pinned(16)
            .expect("readback")
            .download(&device, &stream)
            .expect("offset download");
        download.wait().expect("wait");
        assert_eq!(&download.as_bytes().expect("bytes")[7..10], b"XYZ");
        let empty = runtime.allocate_pinned(0).expect("empty staging");
        assert!(empty.as_bytes().is_empty());
        empty
            .upload(&device, &stream)
            .expect("empty upload")
            .wait()
            .expect("empty wait");
        assert!(
            runtime
                .allocate_pinned(17)
                .expect("oversize staging")
                .download(&device, &stream)
                .is_err()
        );
    }
}
