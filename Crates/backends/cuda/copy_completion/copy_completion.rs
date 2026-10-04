//! Host-visible completion for synchronous API copies queued on the legacy default stream.
#[rustfmt::skip]
use crate::{
    CudaError,
    CudaRuntime,
    DeviceAccessLease,
};
use std::ptr;

fn await_default_copy(runtime: &CudaRuntime) -> Result<(), CudaError> {
    // SAFETY: null denotes the valid legacy default stream of the retained selected runtime.
    // Pageable H2D may return after staging, and D2D may return without host synchronization.
    // Fence this queue before releasing either allocation lease: NonBlocking consumers have
    // no implicit dependency on it. Successful D2H needs no extra fence; only its error path
    // reaches this helper. No device-wide synchronization or fresh symbol lookup is needed.
    unsafe { crate::ffi::invoke_cudaStreamSynchronize(runtime, ptr::null_mut()) }
}

fn finish_copy<const N: usize>(
    leases: [DeviceAccessLease; N],
    copy: Result<(), CudaError>,
    completion: Result<(), CudaError>,
) -> Result<(), CudaError> {
    if completion.is_err() {
        for lease in leases {
            lease.retain_after_unknown_completion();
        }
    }
    // Preserve the original native copy error when both copy and completion fail. A successful
    // copy with uncertain completion returns the wait error and retains every native endpoint.
    copy.and(completion)
}

impl DeviceAccessLease {
    pub(super) fn finish_synchronous(self, copy: Result<(), CudaError>) -> Result<(), CudaError> {
        let completion = await_default_copy(&self.allocation.runtime);
        finish_copy([self], copy, completion)
    }

    pub(super) fn finish_synchronous_with(
        self,
        other: Self,
        copy: Result<(), CudaError>,
    ) -> Result<(), CudaError> {
        let completion = await_default_copy(&self.allocation.runtime);
        finish_copy([self, other], copy, completion)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
