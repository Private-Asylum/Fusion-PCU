//! Private retained orchestration storage for bounded fixed guarded execution.
#[rustfmt::skip]
use super::{
    CudaCompletionBatch,
    CudaReadbackId,
    CudaStreamHandle,
    CudaError,
    DeviceBuffer,
    EventInner,
    Rc,
    Any,
    c_void,
    CUDA_MEMCPY_HOST_TO_DEVICE,
    validate_buffer_range,
};

pub struct GuardedBatch {
    batch: CudaCompletionBatch,
    readback: CudaReadbackId,
    event: Option<EventInner>,
}

impl GuardedBatch {
    pub(crate) fn new(
        stream: &CudaStreamHandle,
        stages: usize,
        bytes: usize,
    ) -> Result<Self, CudaError> {
        let mut batch = CudaCompletionBatch::new(stream);
        batch.resources.reserve_exact(stages + 2);
        let readback = batch.allocate_readback(bytes)?;
        let event = Some(stream.inner.runtime.create_owned_event()?);
        Ok(Self {
            batch,
            readback,
            event,
        })
    }

    pub(crate) const fn batch(&mut self) -> &mut CudaCompletionBatch {
        &mut self.batch
    }

    pub(crate) fn uses_stream(&self, stream: &CudaStreamHandle) -> bool {
        Rc::ptr_eq(&self.batch.stream.inner, &stream.inner)
    }

    pub(crate) fn initialize(
        &mut self,
        destination: &DeviceBuffer,
        source: Rc<Vec<u8>>,
    ) -> Result<(), CudaError> {
        super::ensure_batch_open(self.batch.failed)?;
        self.batch
            .stream
            .inner
            .runtime
            .ensure_same_runtime(&destination.allocation.runtime)?;
        validate_buffer_range(destination.len(), 0, source.len())?;
        let bytes = source.len();
        let pointer = source.as_ptr().cast::<c_void>();
        let owner: Rc<dyn Any> = source;
        self.batch
            .retain_external_operation(&[destination], owner)?;
        // SAFETY: batch owns immutable source storage and destination lease before enqueue.
        let result = unsafe {
            crate::ffi::invoke_cudaMemcpyAsync(
                &self.batch.stream.inner.runtime,
                destination.allocation.pointer,
                pointer,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.batch.stream.inner.raw,
            )
        };
        if result.is_err() {
            self.batch.failed = true;
        }
        result
    }

    pub(crate) fn queue_readback(
        &mut self,
        source: &DeviceBuffer,
        bytes: usize,
    ) -> Result<(), CudaError> {
        self.batch
            .copy_device_to_host(source, &self.readback, bytes)
    }

    pub(crate) fn complete(&mut self) -> Result<&[u8], CudaError> {
        super::ensure_batch_open(self.batch.failed)?;
        let event = self.event.as_ref().ok_or(CudaError::BatchPoisoned)?;
        event.record(&self.batch.stream)?;
        event.synchronize()?;
        self.batch.resources.clear();
        self.batch.launch_events.clear();
        self.batch.dependencies.clear();
        self.batch.queue_markers.clear();
        self.batch.readbacks.read(&self.readback)
    }

    /// End one terminal attempt without reallocating its private endpoints or lease capacity.
    pub(crate) fn reset_after_terminal(&mut self) -> Result<(), CudaError> {
        if !self.batch.resources.is_empty() {
            if let Err(error) = self.batch.stream.synchronize() {
                self.batch.failed = true;
                self.batch.quarantine_and_forget();
                if let Some(event) = self.event.take() {
                    std::mem::forget(event);
                }
                return Err(error);
            }
            self.batch.resources.clear();
        }
        if self.event.is_none() {
            return Err(CudaError::BatchPoisoned);
        }
        self.batch.failed = false;
        self.batch.readbacks.buffer_mut(&self.readback)?.queued = false;
        Ok(())
    }
}
