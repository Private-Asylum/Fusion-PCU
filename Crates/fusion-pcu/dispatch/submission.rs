//! Submission descriptors and execution handles.

use crate::contract::{
    PcuCommandKernelIr,
    PcuDispatchKernelIr,
    PcuError,
    PcuInvocationShape,
    PcuSignalKernelIr,
    PcuStreamKernelIr,
    PcuTransactionKernelIr,
};

/// Finite execution state for one submitted kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuFiniteState {
    Pending,
    Running,
    Complete,
}

/// Persistent execution state for one installed kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuPersistentState {
    Dormant,
    Active,
    Stopped,
}

/// Borrowed dispatch submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchSubmission<'a> {
    pub kernel: &'a PcuDispatchKernelIr<'a>,
    pub shape: PcuInvocationShape,
}

/// Borrowed command submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCommandSubmission<'a> {
    pub kernel: &'a PcuCommandKernelIr<'a>,
}

/// Borrowed transaction submission descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuTransactionSubmission<'a> {
    pub kernel: &'a PcuTransactionKernelIr<'a>,
}

/// Borrowed stream installation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuStreamInstallation<'a> {
    pub kernel: &'a PcuStreamKernelIr<'a>,
}

/// Borrowed signal installation descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSignalInstallation<'a> {
    pub kernel: &'a PcuSignalKernelIr<'a>,
}

/// Handle for one finite PCU submission.
pub trait PcuFiniteHandle {
    /// Returns the current finite execution state.
    ///
    /// # Errors
    ///
    /// Returns any honest state-query failure.
    fn state(&self) -> Result<PcuFiniteState, PcuError>;

    /// Waits synchronously for completion.
    ///
    /// # Errors
    ///
    /// Returns any honest completion failure.
    fn wait(self) -> Result<(), PcuError>;
}

/// Handle for one persistent installed PCU kernel.
pub trait PcuPersistentHandle {
    /// Returns the current persistent execution state.
    ///
    /// # Errors
    ///
    /// Returns any honest state-query failure.
    fn state(&self) -> Result<PcuPersistentState, PcuError>;

    /// Starts one installed persistent kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest start failure.
    fn start(&mut self) -> Result<(), PcuError>;

    /// Stops one installed persistent kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest stop failure.
    fn stop(&mut self) -> Result<(), PcuError>;

    /// Uninstalls one persistent kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest uninstall failure.
    fn uninstall(self) -> Result<(), PcuError>;
}
