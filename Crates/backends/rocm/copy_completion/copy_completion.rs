//! Host-visible terminal readiness and retained endpoints for public HIP copies.
#[rustfmt::skip]
use crate::{
    DeviceAccessLease,
    HipError,
    HipRuntime,
};
use std::ptr;

fn await_default_copy(runtime: &HipRuntime) -> Result<(), HipError> {
    // SAFETY: Exported `hipMemcpy` uses the retained runtime's legacy default stream;
    // null is its valid stream handle. Failed copies may still have enqueued work.
    unsafe { crate::ffi::invoke_hipStreamSynchronize(runtime, ptr::null_mut()) }
}

fn finish_copy<const N: usize>(
    leases: [DeviceAccessLease; N],
    copy: Result<(), HipError>,
    completion: Result<(), HipError>,
) -> Result<(), HipError> {
    if completion.is_err() {
        for lease in leases {
            // Keep the native allocation, owned host endpoint, access gate and runtime.
            // Uncertain completion must never permit reuse or native destruction.
            lease.retain_after_unknown_completion();
        }
    }
    // Preserve the original copy failure when both enqueue and completion fail.
    copy.and(completion)
}

impl DeviceAccessLease {
    pub(super) fn finish_synchronous(self, copy: Result<(), HipError>) -> Result<(), HipError> {
        // Unlike CUDA's pageable H2D staging behavior, HIP's unpinned host transfers
        // complete in the lower copy layer. Independent NonBlocking native witnesses
        // qualify that distinction. Successful D2H is likewise host-terminal.
        if copy.is_ok() {
            return copy;
        }
        let completion = await_default_copy(&self.allocation.runtime);
        finish_copy([self], copy, completion)
    }

    pub(super) fn finish_synchronous_with(
        self,
        other: Self,
        copy: Result<(), HipError>,
    ) -> Result<(), HipError> {
        // HIP's same-device D2D `hipMemcpy` can return before device work completes:
        // ROCm 7.2.4 `clr/hipamd/src/hip_memory.cpp`, `ihipMemcpy`, promotes device
        // copies to host-asynchronous work and orders completion onto the null stream.
        // https://github.com/ROCm/clr/blob/rocm-7.2.4/hipamd/src/hip_memory.cpp
        // Keep both allocation gates until this relevant queue is terminal. This
        // closes independent-consumer ordering without a whole-device success wait.
        let completion = await_default_copy(&self.allocation.runtime);
        finish_copy([self, other], copy, completion)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
