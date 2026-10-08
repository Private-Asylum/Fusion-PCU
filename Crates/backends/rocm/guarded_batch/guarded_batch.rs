//! Private retained orchestration storage for bounded fixed guarded execution.
#[rustfmt::skip]
use super::{
    HipCompletionBatch,
    HipReadbackId,
    HipStreamHandle,
    HipError,
    DeviceBuffer,
    EventInner,
    Rc,
    Any,
    c_void,
    HIP_MEMCPY_HOST_TO_DEVICE,
    validate_buffer_range,
};

pub struct GuardedBatch {
    batch: HipCompletionBatch,
    readback: HipReadbackId,
    event: Option<EventInner>,
}

impl GuardedBatch {
    pub(crate) fn new(
        stream: &HipStreamHandle,
        stages: usize,
        bytes: usize,
    ) -> Result<Self, HipError> {
        let mut batch = HipCompletionBatch::new(stream);
        batch.resources.reserve_exact(stages + 2);
        let readback = batch.allocate_readback(bytes)?;
        let event = Some(stream.inner.runtime.create_owned_event()?);
        Ok(Self {
            batch,
            readback,
            event,
        })
    }

    #[cfg(test)]
    pub(crate) fn retained_readback_bytes(&self) -> &[u8] {
        &self
            .batch
            .readbacks
            .buffer(&self.readback)
            .expect("retained endpoint")
            .bytes
    }

    pub(crate) const fn batch(&mut self) -> &mut HipCompletionBatch {
        &mut self.batch
    }

    pub(crate) fn uses_stream(&self, stream: &HipStreamHandle) -> bool {
        Rc::ptr_eq(&self.batch.stream.inner, &stream.inner)
    }

    pub(crate) fn initialize(
        &mut self,
        destination: &DeviceBuffer,
        source: Rc<Vec<u8>>,
    ) -> Result<(), HipError> {
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
            crate::ffi::invoke_hipMemcpyAsync(
                &self.batch.stream.inner.runtime,
                destination.allocation.pointer,
                pointer,
                bytes,
                HIP_MEMCPY_HOST_TO_DEVICE,
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
    ) -> Result<(), HipError> {
        self.batch
            .copy_device_to_host(source, &self.readback, bytes)
    }

    pub(crate) fn complete(&mut self) -> Result<&[u8], HipError> {
        super::ensure_batch_open(self.batch.failed)?;
        let event = self.event.as_ref().ok_or(HipError::BatchPoisoned)?;
        event.record(&self.batch.stream)?;
        event.synchronize()?;
        self.batch.resources.clear();
        self.batch.launch_events.clear();
        self.batch.dependencies.clear();
        self.batch.queue_markers.clear();
        self.batch.readbacks.read(&self.readback)
    }

    /// End one terminal attempt without reallocating its private endpoints or lease capacity.
    pub(crate) fn reset_after_terminal(&mut self) -> Result<(), HipError> {
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
            return Err(HipError::BatchPoisoned);
        }
        self.batch.failed = false;
        self.batch.readbacks.buffer_mut(&self.readback)?.queued = false;
        Ok(())
    }
}
