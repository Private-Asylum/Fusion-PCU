//! Fusion coprocessor library.
//!
//! `fusion-pcu` is the dedicated semantic home for Fusion's coprocessor model:
//! - generic PCU contract and IR surface re-exported from PAL truth
//! - generic planning, preparation, dispatch, and stream-building helpers
//! - no channel/fiber service glue
//! - no hardware-specific provider implementation ownership

#![cfg_attr(not(feature = "std"), no_std)]

#[path = "contract/contract.rs"]
pub mod contract;
mod dispatch;
mod ir;
mod stream;
mod system;

use core::num::NonZeroU32;

pub use fusion_pal::sys::pcu::{
    PcuAttachmentTableHandle,
    PcuBase,
    PcuByteStreamBindings,
    PcuCaps,
    PcuControl,
    PcuDeviceClaim,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuDeviceId,
    PcuError,
    PcuErrorKind,
    PcuExecutorClaim,
    PcuExecutorClass,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuExecutorMetadataMessage,
    PcuExecutorMetadataProtocol,
    PcuExecutorOrigin,
    PcuHalfWordStreamBindings,
    PcuImplementationKind,
    PcuInvocation,
    PcuInvocationBindings,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuParameter,
    PcuParameterBinding,
    PcuParameterSlot,
    PcuParameterTableHandle,
    PcuParameterValue,
    PcuPortTableHandle,
    PcuSubmissionId,
    PcuSubmissionRequest,
    PcuSubmissionStatusMessage,
    PcuSubmissionStatusProtocol,
    PcuSubmissionWriteProtocol,
    PcuSupport,
    PcuWordStreamBindings,
};

pub use dispatch::*;
pub use ir::*;
pub use stream::*;
pub use system::*;

/// Reusable dispatch profile for one family of PCU submissions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchProfile {
    threads: NonZeroU32,
    policy: PcuDispatchPolicy,
}

impl PcuDispatchProfile {
    /// Creates the default dispatch profile.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            threads: match NonZeroU32::new(1) {
                Some(value) => value,
                None => unreachable!(),
            },
            policy: PcuDispatchPolicy::PreferHardwareAllowCpuFallback,
        }
    }

    /// Returns the requested logical thread count.
    #[must_use]
    pub const fn thread_count(self) -> NonZeroU32 {
        self.threads
    }

    /// Returns the selected dispatch policy.
    #[must_use]
    pub const fn policy(self) -> PcuDispatchPolicy {
        self.policy
    }

    /// Replaces the requested logical thread count with one checked scalar value.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` when `threads == 0`.
    pub fn with_thread_count(mut self, threads: u32) -> Result<Self, PcuError> {
        self.threads = NonZeroU32::new(threads).ok_or_else(PcuError::invalid)?;
        Ok(self)
    }

    /// Replaces the requested logical thread count with one already validated non-zero value.
    #[must_use]
    pub const fn threads(mut self, threads: NonZeroU32) -> Self {
        self.threads = threads;
        self
    }

    /// Replaces the dispatch policy.
    #[must_use]
    pub const fn with_policy(mut self, policy: PcuDispatchPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Forces CPU fallback execution only.
    #[must_use]
    pub const fn cpu_only(self) -> Self {
        self.with_policy(PcuDispatchPolicy::CpuOnly)
    }

    /// Requires one specific backend.
    #[must_use]
    pub const fn require_backend(self, backend: PcuBackendKind) -> Self {
        self.with_policy(PcuDispatchPolicy::Require(backend))
    }

    /// Requires Cortex-M PIO execution.
    #[must_use]
    pub const fn require_pio(self) -> Self {
        self.require_backend(PcuBackendKind::CortexMPio)
    }

    /// Prefers one specific backend and falls back to another supported executor when needed.
    #[must_use]
    pub const fn prefer_backend(self, backend: PcuBackendKind) -> Self {
        self.with_policy(PcuDispatchPolicy::Prefer(backend))
    }

    /// Prefers hardware execution and allows CPU fallback.
    #[must_use]
    pub const fn prefer_hardware(self) -> Self {
        self.with_policy(PcuDispatchPolicy::PreferHardwareAllowCpuFallback)
    }
}

impl Default for PcuDispatchProfile {
    fn default() -> Self {
        Self::new()
    }
}

/// `fusion-pcu` facade plus one reusable dispatch profile.
#[derive(Debug, Clone, Copy)]
pub struct ProfiledPcu<'a> {
    system: &'a PcuSystem,
    profile: PcuDispatchProfile,
}

/// Back-compat alias for the current system-facing PCU facade.
pub type Pcu = PcuSystem;

impl PcuSystem {
    /// Returns one profiled view over this PCU facade.
    #[must_use]
    pub const fn with_profile(&self, profile: PcuDispatchProfile) -> ProfiledPcu<'_> {
        ProfiledPcu {
            system: self,
            profile,
        }
    }

    /// Returns one default dispatch profile over this PCU facade.
    #[must_use]
    pub const fn profile(&self) -> ProfiledPcu<'_> {
        self.with_profile(PcuDispatchProfile::new())
    }

    /// Returns one single-lane PIO-profiled view over this PCU facade.
    #[must_use]
    pub const fn pio(&self) -> ProfiledPcu<'_> {
        self.with_profile(PcuDispatchProfile::new().require_pio())
    }

    /// Returns one CPU-only profiled view over this PCU facade.
    #[must_use]
    pub const fn cpu(&self) -> ProfiledPcu<'_> {
        self.with_profile(PcuDispatchProfile::new().cpu_only())
    }

    /// Returns one PIO-profiled view with an explicit thread count.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` when `threads == 0`.
    pub fn pio_threads(&self, threads: u32) -> Result<ProfiledPcu<'_>, PcuError> {
        Ok(self.with_profile(
            PcuDispatchProfile::new()
                .with_thread_count(threads)?
                .require_pio(),
        ))
    }

    /// Returns one CPU-only profiled view with an explicit thread count.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` when `threads == 0`.
    pub fn cpu_threads(&self, threads: u32) -> Result<ProfiledPcu<'_>, PcuError> {
        Ok(self.with_profile(
            PcuDispatchProfile::new()
                .with_thread_count(threads)?
                .cpu_only(),
        ))
    }

    /// Starts one byte-stream transform builder.
    #[must_use]
    pub fn stream_bytes<'a>(
        &'a self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        PcuStreamDispatchBuilder::new(
            self,
            PcuKernelId(kernel_id),
            entry_point,
            PcuStreamValueType::U8,
        )
    }

    /// Starts one byte-stream transform builder.
    #[must_use]
    pub fn bytes<'a>(
        &'a self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        self.stream_bytes(kernel_id, entry_point)
    }

    /// Starts one half-word stream transform builder.
    #[must_use]
    pub fn stream_half_words<'a>(
        &'a self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        PcuStreamDispatchBuilder::new(
            self,
            PcuKernelId(kernel_id),
            entry_point,
            PcuStreamValueType::U16,
        )
    }

    /// Starts one half-word stream transform builder.
    #[must_use]
    pub fn half_words<'a>(
        &'a self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        self.stream_half_words(kernel_id, entry_point)
    }

    /// Starts one word-stream transform builder.
    #[must_use]
    pub fn stream_words<'a>(
        &'a self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        PcuStreamDispatchBuilder::new(
            self,
            PcuKernelId(kernel_id),
            entry_point,
            PcuStreamValueType::U32,
        )
    }

    /// Starts one word-stream transform builder.
    #[must_use]
    pub fn words<'a>(
        &'a self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        self.stream_words(kernel_id, entry_point)
    }
}

impl ProfiledPcu<'_> {
    /// Returns the selected generic PCU executor system.
    #[must_use]
    pub const fn system(&self) -> &PcuSystem {
        self.system
    }

    /// Returns the active dispatch profile.
    #[must_use]
    pub const fn profile(&self) -> PcuDispatchProfile {
        self.profile
    }
}

impl<'a> ProfiledPcu<'a> {
    /// Returns the dispatch profile carried by this profiled facade.
    #[must_use]
    pub const fn settings(&self) -> PcuDispatchProfile {
        self.profile
    }

    /// Replaces the requested logical thread count carried by this profiled facade.
    ///
    /// # Errors
    ///
    /// Returns `Invalid` when `threads == 0`.
    pub fn with_thread_count(self, threads: u32) -> Result<Self, PcuError> {
        Ok(Self {
            system: self.system,
            profile: self.profile.with_thread_count(threads)?,
        })
    }

    /// Replaces the dispatch policy carried by this profiled facade.
    #[must_use]
    pub const fn with_policy(self, policy: PcuDispatchPolicy) -> Self {
        Self {
            system: self.system,
            profile: self.profile.with_policy(policy),
        }
    }

    /// Forces CPU fallback execution for builders created through this profiled facade.
    #[must_use]
    pub const fn cpu_only(self) -> Self {
        self.with_policy(PcuDispatchPolicy::CpuOnly)
    }

    /// Requires one specific backend for builders created through this profiled facade.
    #[must_use]
    pub const fn require_backend(self, backend: PcuBackendKind) -> Self {
        self.with_policy(PcuDispatchPolicy::Require(backend))
    }

    /// Requires Cortex-M PIO execution for builders created through this profiled facade.
    #[must_use]
    pub const fn require_pio(self) -> Self {
        self.require_backend(PcuBackendKind::CortexMPio)
    }

    /// Starts one byte-stream transform builder with the profiled dispatch settings already
    /// applied.
    #[must_use]
    pub fn stream_bytes(
        &self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        self.system
            .stream_bytes(kernel_id, entry_point)
            .threads(self.profile.thread_count())
            .with_policy(self.profile.policy())
    }

    /// Starts one byte-stream transform builder with the profiled dispatch settings already
    /// applied.
    #[must_use]
    pub fn bytes(&self, kernel_id: u32, entry_point: &'a str) -> PcuStreamDispatchBuilder<'a> {
        self.stream_bytes(kernel_id, entry_point)
    }

    /// Starts one half-word stream transform builder with the profiled dispatch settings already
    /// applied.
    #[must_use]
    pub fn stream_half_words(
        &self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        self.system
            .stream_half_words(kernel_id, entry_point)
            .threads(self.profile.thread_count())
            .with_policy(self.profile.policy())
    }

    /// Starts one half-word stream transform builder with the profiled dispatch settings already
    /// applied.
    #[must_use]
    pub fn half_words(&self, kernel_id: u32, entry_point: &'a str) -> PcuStreamDispatchBuilder<'a> {
        self.stream_half_words(kernel_id, entry_point)
    }

    /// Starts one word-stream transform builder with the profiled dispatch settings already
    /// applied.
    #[must_use]
    pub fn stream_words(
        &self,
        kernel_id: u32,
        entry_point: &'a str,
    ) -> PcuStreamDispatchBuilder<'a> {
        self.system
            .stream_words(kernel_id, entry_point)
            .threads(self.profile.thread_count())
            .with_policy(self.profile.policy())
    }

    /// Starts one word-stream transform builder with the profiled dispatch settings already
    /// applied.
    #[must_use]
    pub fn words(&self, kernel_id: u32, entry_point: &'a str) -> PcuStreamDispatchBuilder<'a> {
        self.stream_words(kernel_id, entry_point)
    }
}

/// Returns the public `fusion-pcu` facade for the selected backend.
#[must_use]
pub const fn system_pcu() -> PcuSystem {
    PcuSystem::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_profile_applies_threads_and_policy_to_stream_builder() {
        let profile = PcuDispatchProfile::new()
            .with_thread_count(4)
            .expect("non-zero thread count should be valid")
            .require_pio();
        let system = PcuSystem::new();
        let builder = system
            .with_profile(profile)
            .stream_words(0x210, "increment");

        assert_eq!(builder.thread_count().get(), 4);
        assert_eq!(
            builder.policy(),
            PcuDispatchPolicy::Require(PcuBackendKind::CortexMPio)
        );
    }

    #[test]
    fn pio_helper_applies_single_lane_pio_defaults() {
        let system = PcuSystem::new();
        let builder = system.pio().words(0x301, "increment");

        assert_eq!(builder.thread_count().get(), 1);
        assert_eq!(
            builder.policy(),
            PcuDispatchPolicy::Require(PcuBackendKind::CortexMPio)
        );
    }

    #[test]
    fn pio_threads_helper_applies_requested_threads() {
        let system = PcuSystem::new();
        let builder = system
            .pio_threads(4)
            .expect("non-zero thread count should be valid")
            .words(0x302, "bit_reverse");

        assert_eq!(builder.thread_count().get(), 4);
        assert_eq!(
            builder.policy(),
            PcuDispatchPolicy::Require(PcuBackendKind::CortexMPio)
        );
    }
}
