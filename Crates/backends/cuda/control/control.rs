//! Opt-in CUDA stream control and readiness observations.
//!
//! Readiness describes queued physical work. It never observes checked arithmetic status,
//! releases allocation leases, or substitutes for an owned completion's terminal protocol.

#[rustfmt::skip]
use std::{
    ptr,
    rc::Rc,
};
#[rustfmt::skip]
use crate::{
    CudaError,
    CudaEventHandle,
    CudaRuntime,
    CudaStreamHandle,
    CudaTimingEventHandle,
    StreamInner,
    ffi::runtime::{
            CUDA_ERROR_NOT_READY,
            CUDA_STREAM_DEFAULT,
            CUDA_STREAM_NON_BLOCKING,
            CUDA_SUCCESS,
            CudaResult,
        },
};

/// CUDA's implicit synchronization relationship with the legacy default stream.
/// Neither mode changes a host call into an allocation/completion ownership barrier.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CudaStreamMode {
    /// Preserve the implicit relationship used by ordinary `cudaStreamCreate`.
    #[default]
    Blocking,
    /// Permit work independent of the legacy default stream's implicit ordering.
    NonBlocking,
}

impl CudaStreamMode {
    const fn flags(self) -> u32 {
        match self {
            Self::Blocking => CUDA_STREAM_DEFAULT,
            Self::NonBlocking => CUDA_STREAM_NON_BLOCKING,
        }
    }

    fn from_flags(flags: u32) -> Result<Self, CudaError> {
        match flags {
            CUDA_STREAM_DEFAULT => Ok(Self::Blocking),
            CUDA_STREAM_NON_BLOCKING => Ok(Self::NonBlocking),
            _ => Err(CudaError::Runtime {
                operation: "cudaStreamGetFlags",
                code: -1,
                detail: Some(format!("unsupported CUDA stream flags {flags}")),
            }),
        }
    }
}

/// Explicit creation options. CUDA clamps priority to the device's meaningful range.
/// Lower numeric values mean higher priority; priority is a scheduling hint, not preemption.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CudaStreamOptions {
    pub mode: CudaStreamMode,
    pub priority: i32,
}

/// Device scheduling priority endpoints; both may be zero when priorities are unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaStreamPriorityRange {
    pub least: i32,
    pub greatest: i32,
}

/// Snapshot of physical queued-work readiness, with no arithmetic-success implication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaReadiness {
    Pending,
    Ready,
}

pub const fn classify_readiness(status: CudaResult) -> Option<CudaReadiness> {
    match status {
        CUDA_SUCCESS => Some(CudaReadiness::Ready),
        CUDA_ERROR_NOT_READY => Some(CudaReadiness::Pending),
        _ => None,
    }
}

impl CudaRuntime {
    /// Create an explicitly configured stream, preserving existing default creation behavior.
    /// CUDA clamps out-of-range priority requests; query [`CudaStreamHandle::options`] for the
    /// effective settings. Nonblocking describes default-stream ordering, not CPU completion.
    ///
    /// # Errors
    /// Returns a missing stable symbol or CUDA stream-creation error.
    pub fn create_stream_with_options(
        &self,
        options: CudaStreamOptions,
    ) -> Result<CudaStreamHandle, CudaError> {
        let mut stream = ptr::null_mut();
        unsafe {
            crate::ffi::invoke_cudaStreamCreateWithPriority(
                self,
                &raw mut stream,
                options.mode.flags(),
                options.priority,
            )
        }?;
        Ok(CudaStreamHandle {
            inner: Rc::new(StreamInner {
                runtime: self.clone(),
                raw: stream,
            }),
        })
    }

    /// Query the current device's meaningful stream-priority endpoints on explicit request.
    ///
    /// # Errors
    /// Returns a missing symbol or CUDA device-priority query error.
    pub fn stream_priority_range(&self) -> Result<CudaStreamPriorityRange, CudaError> {
        let mut range = CudaStreamPriorityRange {
            least: 0,
            greatest: 0,
        };
        unsafe {
            crate::ffi::invoke_cudaDeviceGetStreamPriorityRange(
                self,
                &raw mut range.least,
                &raw mut range.greatest,
            )
        }?;
        Ok(range)
    }
}

impl CudaStreamHandle {
    /// Query actual flags and CUDA-clamped priority without changing the stream.
    ///
    /// # Errors
    /// Returns a missing symbol, unknown flag set, or CUDA query error.
    pub fn options(&self) -> Result<CudaStreamOptions, CudaError> {
        let runtime = &self.inner.runtime;
        let mut flags = 0;
        unsafe { crate::ffi::invoke_cudaStreamGetFlags(runtime, self.inner.raw, &raw mut flags) }?;
        let mut priority = 0;
        unsafe {
            crate::ffi::invoke_cudaStreamGetPriority(runtime, self.inner.raw, &raw mut priority)
        }?;
        Ok(CudaStreamOptions {
            mode: CudaStreamMode::from_flags(flags)?,
            priority,
        })
    }

    /// Observe all work currently queued on this stream without releasing any retained owners.
    /// A ready stream does not prove checked arithmetic success or expose pending readback bytes.
    ///
    /// # Errors
    /// Returns a missing query symbol or CUDA error other than ordinary not-ready status.
    pub fn query(&self) -> Result<CudaReadiness, CudaError> {
        unsafe { crate::ffi::invoke_cudaStreamQuery(&self.inner.runtime, self.inner.raw) }
    }
}

impl CudaEventHandle {
    /// Observe the event's most recently recorded work. An unrecorded event is CUDA-ready for an
    /// empty work set. This observation never releases completion owners or reads fault status.
    ///
    /// # Errors
    /// Returns a missing query symbol or CUDA error other than ordinary not-ready status.
    pub fn query(&self) -> Result<CudaReadiness, CudaError> {
        unsafe { crate::ffi::invoke_cudaEventQuery(&self.inner.runtime, self.inner.raw) }
    }
}

impl CudaTimingEventHandle {
    /// Observe recorded timing work without changing allocation/completion ownership.
    /// An unrecorded event reports readiness for an empty work set.
    ///
    /// # Errors
    /// Returns a missing query symbol or CUDA error other than ordinary not-ready status.
    pub fn query(&self) -> Result<CudaReadiness, CudaError> {
        unsafe { crate::ffi::invoke_cudaEventQuery(&self.inner.runtime, self.inner.raw) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_options_preserve_blocking_default_priority() {
        assert_eq!(
            CudaStreamOptions::default(),
            CudaStreamOptions {
                mode: CudaStreamMode::Blocking,
                priority: 0,
            }
        );
        assert_eq!(CudaStreamMode::Blocking.flags(), 0);
        assert_eq!(CudaStreamMode::NonBlocking.flags(), 1);
        assert_eq!(
            CudaStreamMode::from_flags(1).unwrap(),
            CudaStreamMode::NonBlocking
        );
        assert!(CudaStreamMode::from_flags(2).is_err());
    }

    #[test]
    fn readiness_classifies_pending_without_promoting_cuda_failures() {
        assert_eq!(classify_readiness(0), Some(CudaReadiness::Ready));
        assert_eq!(classify_readiness(600), Some(CudaReadiness::Pending));
        assert_eq!(classify_readiness(700), None);
        assert_eq!(classify_readiness(-1), None);
    }

    #[test]
    #[ignore = "requires CUDA device for stream options and event readiness"]
    fn configured_streams_query_clamped_priority_and_preserve_completion_leases() {
        let runtime = CudaRuntime::new(0).unwrap();
        let range = runtime.stream_priority_range().unwrap();
        assert!(range.greatest <= range.least);
        assert_eq!(
            runtime.create_stream().unwrap().options().unwrap(),
            CudaStreamOptions::default()
        );
        for (priority, expected) in [(i32::MIN, range.greatest), (i32::MAX, range.least)] {
            let stream = runtime
                .create_stream_with_options(CudaStreamOptions {
                    mode: CudaStreamMode::NonBlocking,
                    priority,
                })
                .unwrap();
            assert_eq!(
                stream.options().unwrap(),
                CudaStreamOptions {
                    mode: CudaStreamMode::NonBlocking,
                    priority: expected,
                }
            );
            let event = runtime.create_event().unwrap();
            assert_eq!(event.query().unwrap(), CudaReadiness::Ready);
            let device = runtime.allocate(8).unwrap();
            let mut batch = crate::CudaCompletionBatch::new(&stream);
            batch
                .copy_host_to_device(&device, std::sync::Arc::from(&b"controls"[..]))
                .unwrap();
            stream.record(&event).unwrap();
            let mut completion = batch.finish().unwrap();
            event.synchronize().unwrap();
            assert_eq!(event.query().unwrap(), CudaReadiness::Ready);
            stream.synchronize().unwrap();
            assert_eq!(stream.query().unwrap(), CudaReadiness::Ready);
            let mut actual = [0; 8];
            assert_eq!(device.copy_to(&mut actual), Err(CudaError::Busy));
            completion.wait().unwrap();
            device.copy_to(&mut actual).unwrap();
            assert_eq!(&actual, b"controls");
            let timing = runtime.create_timing_event().unwrap();
            stream.record_timing(&timing).unwrap();
            timing.synchronize().unwrap();
            assert_eq!(timing.query().unwrap(), CudaReadiness::Ready);
        }
    }
}
