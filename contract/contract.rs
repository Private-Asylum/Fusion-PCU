//! Canonical PCU contract vocabulary.
//!
//! `fusion-pcu` owns the pure architectural model:
//! - capability and executor vocabulary
//! - generic invocation and IR vocabulary
//! - backend-neutral unsupported placeholder types
//!
//! It intentionally does not own:
//! - transport/channel protocol glue
//! - platform/provider selection
//! - backend lowering or dispatch policy

pub use crate::core::*;
pub use crate::validation::*;

pub mod caps;
pub mod device;
pub mod error;
pub mod invocation;
pub mod ir;
pub mod unsupported;

pub use caps::*;
pub use device::*;
pub use error::*;
pub use invocation::*;
pub use ir::*;
pub use unsupported::*;

/// Capability trait for generic PCU backends.
pub trait PcuBaseContract {
    /// Reports the truthful generic PCU surface for this backend.
    fn support(&self) -> PcuSupport;

    /// Returns the surfaced PCU execution substrates.
    #[must_use]
    fn executors(&self) -> &'static [PcuExecutorDescriptor];

    /// Returns one surfaced execution substrate descriptor by id.
    #[must_use]
    fn executor(&self, executor: PcuExecutorId) -> Option<PcuExecutorDescriptor> {
        self.executors()
            .iter()
            .copied()
            .find(|descriptor| descriptor.id == executor)
    }

    /// Returns whether one surfaced execution substrate can run the supplied kernel directly.
    #[must_use]
    fn executor_supports_kernel_direct(
        &self,
        executor: PcuExecutorId,
        kernel: PcuKernel<'_>,
    ) -> bool {
        self.executor(executor)
            .is_some_and(|descriptor| descriptor.support.supports_kernel_direct(kernel))
    }

    /// Returns whether any surfaced execution substrate can run the supplied kernel directly.
    #[must_use]
    fn any_executor_supports_kernel_direct(&self, kernel: PcuKernel<'_>) -> bool {
        self.executors()
            .iter()
            .copied()
            .any(|descriptor| descriptor.supports_kernel_direct(kernel))
    }

    /// Returns whether the backend can run the supplied kernel through CPU fallback.
    #[must_use]
    fn supports_kernel_cpu_fallback(&self, kernel: PcuKernel<'_>) -> bool {
        self.support().supports_kernel_cpu_fallback(kernel)
    }
}

/// Control contract for generic PCU backends.
pub trait PcuControlContract: PcuBaseContract {
    /// Claims one PCU execution substrate exclusively.
    ///
    /// # Errors
    ///
    /// Returns an honest error when the executor is unknown, unsupported, or already claimed.
    fn claim_executor(&self, executor: PcuExecutorId) -> Result<PcuExecutorClaim, PcuError>;

    /// Releases one previously claimed execution substrate.
    ///
    /// # Errors
    ///
    /// Returns an honest error when the claim no longer matches backend state.
    fn release_executor(&self, claim: PcuExecutorClaim) -> Result<(), PcuError>;
}
