//! Routing and scheduling contracts for PCU execution models.
//!
//! This module defines how model-local kernels reach an execution substrate.
//! It intentionally does not define:
//! - backend preference policy
//! - provider selection
//! - platform lowering
//! - CPU fallback doctrine

use core::num::NonZeroU32;

use crate::contract::{
    PcuBaseContract,
    PcuBinding,
    PcuBindingRef,
    PcuCommandKernelIr,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuError,
    PcuExecutorId,
    PcuInvocationBindings,
    PcuInvocationTarget,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuInvocationTopology,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelSignature,
    PcuPort,
    PcuSignalKernelIr,
    PcuStreamKernelIr,
    PcuTransactionKernelIr,
};
use crate::validation::validate_command_kernel;

/// Logical invocation context surfaced to one dispatch-style kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchContext {
    global_invocation_id: u32,
    invocation_count: NonZeroU32,
}

impl PcuDispatchContext {
    /// Creates one logical invocation context, rejecting padded physical lanes.
    #[must_use]
    pub const fn new(global_invocation_id: u32, invocation_count: NonZeroU32) -> Option<Self> {
        if global_invocation_id >= invocation_count.get() {
            return None;
        }
        Some(Self {
            global_invocation_id,
            invocation_count,
        })
    }

    /// Returns this invocation's global logical identifier.
    #[must_use]
    pub const fn global_invocation_id(self) -> u32 {
        self.global_invocation_id
    }

    /// Returns the requested logical invocation count, independent of physical launch padding.
    #[must_use]
    pub const fn invocation_count(self) -> NonZeroU32 {
        self.invocation_count
    }

    /// Returns the indices this logical invocation covers in a grid-stride loop.
    ///
    /// The stride is the logical invocation count, even if a backend launches more physical
    /// lanes to fill its final workgroup. Callers may use this as a serial CPU reference for the
    /// portable one-dimensional grid-stride contract.
    #[must_use]
    pub const fn grid_stride_indices(self, extent: u64) -> PcuGridStrideIndices {
        let first = self.global_invocation_id as u64;
        PcuGridStrideIndices {
            next: if first < extent { Some(first) } else { None },
            stride: self.invocation_count,
            extent,
        }
    }
}

/// Overflow-safe serial reference for one logical invocation's grid-stride indices.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PcuGridStrideIndices {
    next: Option<u64>,
    stride: NonZeroU32,
    extent: u64,
}

impl Iterator for PcuGridStrideIndices {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.next?;
        self.next = current
            .checked_add(u64::from(self.stride.get()))
            .filter(|next| *next < self.extent);
        Some(current)
    }
}

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

/// Routing contract for the full PCU model family.
pub trait PcuDispatchContract {
    type DispatchHandle: PcuFiniteHandle;
    type CommandHandle: PcuFiniteHandle;
    type TransactionHandle: PcuFiniteHandle;
    type StreamHandle: PcuPersistentHandle;
    type SignalHandle: PcuPersistentHandle;

    /// Submits one finite logical-dispatch kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError>;

    /// Submits one finite sequential command kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_command(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError>;

    /// Submits one finite opaque transaction kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_transaction(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError>;

    /// Installs one persistent stream kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest installation failure.
    fn install_stream(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError>;

    /// Installs one persistent signal kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest installation failure.
    fn install_signal(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError>;
}

/// Direct-execution backend contract for PCU model routing.
///
/// This is the foundation-level adapter surface: `fusion-pcu` owns structural admission checks and
/// direct-support law, while concrete backends own actual executor choice and execution.
///
/// It intentionally does not define:
/// - CPU fallback selection
/// - fiber/channel choreography
/// - pipelining strategy
/// - backend preference policy
pub trait PcuDirectDispatchBackend: PcuBaseContract {
    type DispatchHandle: PcuFiniteHandle;
    type CommandHandle: PcuFiniteHandle;
    type TransactionHandle: PcuFiniteHandle;
    type StreamHandle: PcuPersistentHandle;
    type SignalHandle: PcuPersistentHandle;

    /// Submits one already-validated direct dispatch kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_dispatch_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError>;

    /// Submits one already-validated direct command kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_command_direct(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError>;

    /// Submits one already-validated direct transaction kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_transaction_direct(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError>;

    /// Installs one already-validated direct stream kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend installation failure.
    fn install_stream_direct(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError>;

    /// Installs one already-validated direct signal kernel.
    ///
    /// # Errors
    ///
    /// Returns any honest backend installation failure.
    fn install_signal_direct(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError>;
}

/// Stream-family execution surface for a consumer-owned scheduler or courier.
pub trait PcuStreamBackend: PcuBaseContract {
    type StreamHandle: PcuPersistentHandle;

    /// Installs one persistent Stream program.
    ///
    /// # Errors
    ///
    /// Returns an admission or backend installation error.
    fn install_stream(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError>;
}

/// Direct Stream-only backend contract.
///
/// This permits a PIO-like backend to implement the Stream family without placeholder handles
/// or methods for unrelated PCU families. The broader legacy backend trait remains available
/// while family-specific contracts are introduced incrementally.
pub trait PcuDirectStreamBackend: PcuBaseContract {
    type StreamHandle: PcuPersistentHandle;

    /// Installs a Stream program after common structural and direct-support checks.
    ///
    /// # Errors
    ///
    /// Returns an honest support, binding, parameter, or backend installation error.
    fn install_stream(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Stream(*installation.kernel))?;
        validate_parameters(installation.kernel.signature(), parameters)?;
        validate_invocation_bindings(installation.kernel.signature(), bindings)?;
        self.install_stream_direct(installation, bindings, parameters)
    }

    /// Installs a previously checked Stream program on this backend.
    ///
    /// # Errors
    ///
    /// Returns an honest backend installation error.
    fn install_stream_direct(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError>;
}

impl<T> PcuStreamBackend for T
where
    T: PcuDirectStreamBackend,
{
    type StreamHandle = T::StreamHandle;

    fn install_stream(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError> {
        PcuDirectStreamBackend::install_stream(self, installation, bindings, parameters)
    }
}

/// Stream backend that grants an exclusive, backend-bound executor lease.
///
/// The associated lease is an opaque backend-owned token by contract. Implementations should make
/// it non-`Copy`, bind it to the originating backend and executor generation, and keep the claim
/// alive for as long as installed work can use the executor. Rust cannot enforce those properties
/// for an associated type. `install_stream_on_lease` performs common admission against the
/// executor named by the lease; the backend method must still reject forged, foreign, or stale
/// leases before touching the executor.
pub trait PcuExclusiveStreamBackend: PcuBaseContract {
    type Lease;
    type StreamHandle: PcuPersistentHandle;

    /// Claims one executor for exclusive Stream installation.
    ///
    /// # Errors
    ///
    /// Returns an honest invalid, unsupported, or busy error.
    fn claim_stream_executor(&self, executor: PcuExecutorId) -> Result<Self::Lease, PcuError>;

    /// Returns the executor identity carried by a lease.
    fn lease_executor(&self, lease: &Self::Lease) -> PcuExecutorId;

    /// Installs a Stream program using the selected executor lease after common admission checks.
    ///
    /// # Errors
    ///
    /// Returns an honest support, parameter, binding, lease, or installation error.
    fn install_stream_on_lease(
        &self,
        lease: &mut Self::Lease,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError> {
        let executor = self.lease_executor(lease);
        if !self.executor_supports_kernel_direct(executor, PcuKernel::Stream(*installation.kernel))
        {
            return Err(PcuError::unsupported());
        }
        validate_parameters(installation.kernel.signature(), parameters)?;
        validate_invocation_bindings(installation.kernel.signature(), bindings)?;
        self.install_stream_on_lease_direct(lease, installation, bindings, parameters)
    }

    /// Installs one already-admitted Stream program and validates that the lease is live and
    /// belongs to this backend before using the selected executor.
    ///
    /// # Errors
    ///
    /// Returns an error if the lease or backend installation is invalid.
    fn install_stream_on_lease_direct(
        &self,
        lease: &mut Self::Lease,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError>;
}

/// Command-family execution surface.
///
/// Backends that execute only sequential Command programs can implement this trait without
/// providing placeholder handles or methods for Dispatch, Transaction, Stream, or Signal.
pub trait PcuCommandBackend: PcuBaseContract {
    type CommandHandle: PcuFiniteHandle;

    /// Submits one finite sequential command program.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_command(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError>;
}

/// Direct Command-family backend implementation contract.
///
/// The default method applies the common direct-support and parameter checks before delegating to
/// the backend implementation. It has no associated types or operations for unrelated families.
pub trait PcuDirectCommandBackend: PcuBaseContract {
    type CommandHandle: PcuFiniteHandle;

    /// Submits one Command program after common admission checks.
    ///
    /// # Errors
    ///
    /// Returns an honest support, parameter, or backend execution error.
    fn submit_command(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Command(*submission.kernel))?;
        validate_command_kernel(submission.kernel).map_err(|_| PcuError::invalid())?;
        validate_parameters(submission.kernel.signature(), parameters)?;
        self.submit_command_direct(submission, parameters)
    }

    /// Submits one already-validated direct Command program.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or execution failure.
    fn submit_command_direct(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError>;
}

impl<T> PcuCommandBackend for T
where
    T: PcuDirectCommandBackend,
{
    type CommandHandle = T::CommandHandle;

    fn submit_command(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError> {
        PcuDirectCommandBackend::submit_command(self, submission, parameters)
    }
}

/// Signal-family execution surface.
///
/// Backends that install only triggered Signal programs can implement this trait without
/// providing placeholder handles or methods for Dispatch, Command, Transaction, or Stream.
pub trait PcuSignalBackend: PcuBaseContract {
    type SignalHandle: PcuPersistentHandle;

    /// Installs one persistent Signal program.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or installation failure.
    fn install_signal(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError>;
}

/// Direct Signal-family backend implementation contract.
///
/// The default method applies the common direct-support and parameter checks before delegating to
/// the backend implementation. It defines no operations for unrelated families.
pub trait PcuDirectSignalBackend: PcuBaseContract {
    type SignalHandle: PcuPersistentHandle;

    /// Installs one Signal program after common admission checks.
    ///
    /// # Errors
    ///
    /// Returns an honest support, parameter, or backend installation error.
    fn install_signal(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Signal(*installation.kernel))?;
        validate_parameters(installation.kernel.signature(), parameters)?;
        self.install_signal_direct(installation, parameters)
    }

    /// Installs one already-validated direct Signal program.
    ///
    /// # Errors
    ///
    /// Returns any honest backend admission or installation failure.
    fn install_signal_direct(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError>;
}

impl<T> PcuSignalBackend for T
where
    T: PcuDirectSignalBackend,
{
    type SignalHandle = T::SignalHandle;

    fn install_signal(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError> {
        PcuDirectSignalBackend::install_signal(self, installation, parameters)
    }
}

/// Dispatch-family execution surface.
///
/// The historical `PcuDirectDispatchBackend` remains the aggregate five-family adapter. This
/// family-specific surface lets a backend implement indexed Dispatch alone.
pub trait PcuDispatchBackend: PcuBaseContract {
    type DispatchHandle: PcuFiniteHandle;

    /// Submits one finite logical-dispatch program.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError>;
}

/// Direct Dispatch-family backend implementation contract.
pub trait PcuDirectDispatchFamilyBackend: PcuBaseContract {
    type DispatchHandle: PcuFiniteHandle;

    /// Submits one Dispatch program after common admission checks.
    ///
    /// # Errors
    ///
    /// Returns an honest support, shape, binding, parameter, or backend execution error.
    fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Dispatch(*submission.kernel))?;
        validate_dispatch_submission(submission)?;
        validate_parameters(submission.kernel.signature(), parameters)?;
        validate_invocation_bindings(submission.kernel.signature(), bindings)?;
        self.submit_dispatch_direct(submission, bindings, parameters)
    }

    /// Submits one already-validated direct Dispatch program.
    ///
    /// # Errors
    ///
    /// Returns a backend submission error without claiming completion.
    fn submit_dispatch_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError>;
}

impl<T> PcuDispatchBackend for T
where
    T: PcuDirectDispatchFamilyBackend,
{
    type DispatchHandle = T::DispatchHandle;

    fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError> {
        PcuDirectDispatchFamilyBackend::submit_dispatch(self, submission, bindings, parameters)
    }
}

/// Transaction-family execution surface.
pub trait PcuTransactionBackend: PcuBaseContract {
    type TransactionHandle: PcuFiniteHandle;

    /// Submits one finite opaque transaction program.
    ///
    /// # Errors
    ///
    /// Returns any honest admission, scheduling, or execution-substrate failure.
    fn submit_transaction(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError>;
}

/// Direct Transaction-family backend implementation contract.
pub trait PcuDirectTransactionBackend: PcuBaseContract {
    type TransactionHandle: PcuFiniteHandle;

    /// Submits one Transaction program after common admission checks.
    ///
    /// # Errors
    ///
    /// Returns an honest support, binding, parameter, or backend execution error.
    fn submit_transaction(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Transaction(*submission.kernel))?;
        validate_parameters(submission.kernel.signature(), parameters)?;
        validate_invocation_bindings(submission.kernel.signature(), bindings)?;
        self.submit_transaction_direct(submission, bindings, parameters)
    }

    /// Submits one already-validated direct Transaction program.
    ///
    /// # Errors
    ///
    /// Returns a backend submission error without claiming completion.
    fn submit_transaction_direct(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError>;
}

impl<T> PcuTransactionBackend for T
where
    T: PcuDirectTransactionBackend,
{
    type TransactionHandle = T::TransactionHandle;

    fn submit_transaction(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError> {
        PcuDirectTransactionBackend::submit_transaction(self, submission, bindings, parameters)
    }
}

impl<T> PcuDispatchContract for T
where
    T: PcuDirectDispatchBackend,
{
    type DispatchHandle = T::DispatchHandle;
    type CommandHandle = T::CommandHandle;
    type TransactionHandle = T::TransactionHandle;
    type StreamHandle = T::StreamHandle;
    type SignalHandle = T::SignalHandle;

    fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::DispatchHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Dispatch(*submission.kernel))?;
        validate_dispatch_submission(submission)?;
        validate_parameters(submission.kernel.signature(), parameters)?;
        validate_invocation_bindings(submission.kernel.signature(), bindings)?;
        self.submit_dispatch_direct(submission, bindings, parameters)
    }

    fn submit_command(
        &self,
        submission: PcuCommandSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::CommandHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Command(*submission.kernel))?;
        validate_command_kernel(submission.kernel).map_err(|_| PcuError::invalid())?;
        validate_parameters(submission.kernel.signature(), parameters)?;
        self.submit_command_direct(submission, parameters)
    }

    fn submit_transaction(
        &self,
        submission: PcuTransactionSubmission<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::TransactionHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Transaction(*submission.kernel))?;
        validate_parameters(submission.kernel.signature(), parameters)?;
        validate_invocation_bindings(submission.kernel.signature(), bindings)?;
        self.submit_transaction_direct(submission, bindings, parameters)
    }

    fn install_stream(
        &self,
        installation: PcuStreamInstallation<'_>,
        bindings: PcuInvocationBindings<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::StreamHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Stream(*installation.kernel))?;
        validate_parameters(installation.kernel.signature(), parameters)?;
        validate_invocation_bindings(installation.kernel.signature(), bindings)?;
        self.install_stream_direct(installation, bindings, parameters)
    }

    fn install_signal(
        &self,
        installation: PcuSignalInstallation<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::SignalHandle, PcuError> {
        validate_direct_kernel_support(self, PcuKernel::Signal(*installation.kernel))?;
        validate_parameters(installation.kernel.signature(), parameters)?;
        self.install_signal_direct(installation, parameters)
    }
}

fn validate_direct_kernel_support<B: PcuBaseContract + ?Sized>(
    backend: &B,
    kernel: PcuKernel<'_>,
) -> Result<(), PcuError> {
    let kernel_supported = backend.any_executor_supports_kernel_direct(kernel);

    if let PcuKernel::Dispatch(dispatch) = kernel
        && !kernel_supported
    {
        let any_structural = backend.executors().iter().copied().any(|descriptor| {
            descriptor
                .support
                .supports_dispatch_direct_structure(dispatch)
        });
        if any_structural {
            let any_typed = backend.executors().iter().copied().any(|descriptor| {
                descriptor
                    .support
                    .supports_dispatch_direct_structure(dispatch)
                    && descriptor
                        .support
                        .supports_value_types_direct(dispatch.required_type_support())
            });
            if any_typed {
                return Err(PcuError::unsupported_feature_support());
            }
            return Err(PcuError::unsupported_type_support());
        }
    }

    if kernel_supported {
        return Ok(());
    }
    Err(PcuError::unsupported())
}

/// Checks that invocation parameters match the declared slots and types.
///
/// # Errors
///
/// Returns `Invalid` for missing, extra, duplicate, or mistyped parameters.
pub fn validate_parameters(
    signature: PcuKernelSignature<'_>,
    parameters: PcuInvocationParameters<'_>,
) -> Result<(), PcuError> {
    if parameters.validate_against(signature.parameters) {
        Ok(())
    } else {
        Err(PcuError::invalid())
    }
}

/// Checks that each supplied binding target exists and appears only once.
///
/// This is a partial structural check: backends must also verify required targets, access,
/// layout, size, and residency before execution.
///
/// # Errors
///
/// Returns `Invalid` for duplicate or unknown targets.
pub fn validate_invocation_bindings(
    signature: PcuKernelSignature<'_>,
    bindings: PcuInvocationBindings<'_>,
) -> Result<(), PcuError> {
    for (index, binding) in bindings.bindings.iter().enumerate() {
        if bindings.bindings[..index]
            .iter()
            .any(|existing| existing.target == binding.target)
        {
            return Err(PcuError::invalid());
        }

        let target_exists = match binding.target {
            PcuInvocationTarget::Binding(reference) => {
                binding_exists(signature.bindings, reference)
            }
            PcuInvocationTarget::Port(name) => port_exists(signature.ports, name),
        };

        if !target_exists {
            return Err(PcuError::invalid());
        }
    }

    Ok(())
}

/// Checks that a dispatch submission's invocation count matches its logical shape.
///
/// # Errors
///
/// Returns `Invalid` for an unsupported topology, zero/overflowed extent, or mismatched count.
pub fn validate_dispatch_submission(submission: PcuDispatchSubmission<'_>) -> Result<(), PcuError> {
    for (index, binding) in submission.kernel.bindings.iter().enumerate() {
        if submission.kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            // A binding address names exactly one declared resource. Accepting duplicates makes
            // lookup-based admission depend on declaration order, so the requested access/type
            // contract could disagree with the resource a backend actually selects.
            return Err(PcuError::invalid());
        }
    }
    if submission
        .kernel
        .ops
        .iter()
        .any(|op| matches!(op, PcuDispatchOp::GridStrideLoop { extent: 0, .. }))
    {
        return Err(PcuError::invalid());
    }
    let PcuInvocationTopology::Indexed { logical_shape } =
        submission.kernel.signature().invocation.topology
    else {
        return Err(PcuError::invalid());
    };

    let expected_invocations = logical_shape
        .into_iter()
        .try_fold(1_u64, |product, axis| product.checked_mul(u64::from(axis)))
        .ok_or_else(PcuError::invalid)?;

    if expected_invocations == 0
        || expected_invocations != u64::from(submission.shape.invocation_count().get())
    {
        return Err(PcuError::invalid());
    }

    Ok(())
}

fn binding_exists(bindings: &[PcuBinding<'_>], reference: PcuBindingRef) -> bool {
    bindings
        .iter()
        .copied()
        .any(|binding| binding.reference() == reference)
}

fn port_exists(ports: &[PcuPort<'_>], name: &str) -> bool {
    ports.iter().any(|port| port.name == Some(name))
}

#[cfg(test)]
mod tests {
    use core::num::NonZeroU32;

    #[test]
    fn grid_stride_reference_covers_each_element_once_without_padded_lanes() {
        let count = NonZeroU32::new(250).expect("nonzero");
        let mut seen = std::vec![0_u8; 2048];
        for physical_lane in 0..256 {
            let Some(context) = super::PcuDispatchContext::new(physical_lane, count) else {
                assert!(physical_lane >= count.get());
                continue;
            };
            for index in context.grid_stride_indices(2048) {
                seen[usize::try_from(index).expect("small index")] += 1;
            }
        }
        assert!(seen.iter().all(|visits| *visits == 1));
    }

    #[test]
    fn grid_stride_reference_stops_without_wrapping() {
        let count = NonZeroU32::new(2).expect("nonzero");
        let mut indices = super::PcuGridStrideIndices {
            next: Some(u64::MAX - 1),
            stride: count,
            extent: u64::MAX,
        };
        assert_eq!(indices.next(), Some(u64::MAX - 1));
        assert_eq!(indices.next(), None);

        let context = super::PcuDispatchContext::new(1, count).expect("active lane");
        assert_eq!(context.grid_stride_indices(1).next(), None);
        assert!(super::PcuDispatchContext::new(2, count).is_none());
    }

    use super::{
        PcuCommandSubmission,
        PcuCommandBackend,
        PcuDirectCommandBackend,
        PcuDirectDispatchFamilyBackend,
        PcuDirectDispatchBackend,
        PcuDirectSignalBackend,
        PcuDirectStreamBackend,
        PcuDirectTransactionBackend,
        PcuDispatchBackend,
        PcuDispatchContract,
        PcuDispatchSubmission,
        PcuFiniteHandle,
        PcuFiniteState,
        PcuPersistentHandle,
        PcuPersistentState,
        PcuSignalBackend,
        PcuSignalInstallation,
        PcuExclusiveStreamBackend,
        PcuStreamInstallation,
        PcuTransactionBackend,
    };
    use crate::{
        PcuBaseContract,
        PcuCaps,
        PcuError,
        PcuCommandOp,
        PcuDispatchOpCaps,
        PcuDispatchPolicyCaps,
        PcuDispatchSupport,
        PcuExecutorClass,
        PcuExecutorDescriptor,
        PcuExecutorId,
        PcuExecutorOrigin,
        PcuExecutorSupport,
        PcuFeatureSupport,
        PcuInvocationBindings,
        PcuInvocationBuffer,
        PcuInvocationBinding,
        PcuInvocationParameters,
        PcuInvocationShape,
        PcuInvocationTarget,
        PcuParameter,
        PcuParameterSlot,
        PcuParameterValue,
        PcuPrimitiveCaps,
        PcuPrimitiveSupport,
        PcuSupport,
        PcuStreamCapabilities,
        PcuValueType,
        model::{
            PcuCommandKernelBuilder,
            PcuDispatchAluOp,
            PcuDispatchKernelBuilder,
            PcuStreamKernelBuilder,
        },
    };

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestFiniteHandle;

    impl PcuFiniteHandle for TestFiniteHandle {
        fn state(&self) -> Result<PcuFiniteState, PcuError> {
            Ok(PcuFiniteState::Complete)
        }

        fn wait(self) -> Result<(), PcuError> {
            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct TestPersistentHandle;

    impl PcuPersistentHandle for TestPersistentHandle {
        fn state(&self) -> Result<PcuPersistentState, PcuError> {
            Ok(PcuPersistentState::Dormant)
        }

        fn start(&mut self) -> Result<(), PcuError> {
            Ok(())
        }

        fn stop(&mut self) -> Result<(), PcuError> {
            Ok(())
        }

        fn uninstall(self) -> Result<(), PcuError> {
            Ok(())
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct TestBackend {
        support: PcuSupport,
        executors: &'static [PcuExecutorDescriptor],
    }

    impl PcuBaseContract for TestBackend {
        fn support(&self) -> PcuSupport {
            self.support
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.executors
        }
    }

    impl PcuDirectDispatchBackend for TestBackend {
        type DispatchHandle = TestFiniteHandle;
        type CommandHandle = TestFiniteHandle;
        type TransactionHandle = TestFiniteHandle;
        type StreamHandle = TestPersistentHandle;
        type SignalHandle = TestPersistentHandle;

        fn submit_dispatch_direct(
            &self,
            _submission: PcuDispatchSubmission<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::DispatchHandle, PcuError> {
            Ok(TestFiniteHandle)
        }

        fn submit_command_direct(
            &self,
            _submission: PcuCommandSubmission<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::CommandHandle, PcuError> {
            Ok(TestFiniteHandle)
        }

        fn submit_transaction_direct(
            &self,
            _submission: super::PcuTransactionSubmission<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::TransactionHandle, PcuError> {
            Ok(TestFiniteHandle)
        }

        fn install_stream_direct(
            &self,
            _installation: PcuStreamInstallation<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::StreamHandle, PcuError> {
            Ok(TestPersistentHandle)
        }

        fn install_signal_direct(
            &self,
            _installation: super::PcuSignalInstallation<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::SignalHandle, PcuError> {
            Ok(TestPersistentHandle)
        }
    }

    struct StreamOnlyBackend(TestBackend);

    impl PcuBaseContract for StreamOnlyBackend {
        fn support(&self) -> PcuSupport {
            self.0.support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.0.executors()
        }
    }

    impl PcuDirectStreamBackend for StreamOnlyBackend {
        type StreamHandle = TestPersistentHandle;

        fn install_stream_direct(
            &self,
            _installation: PcuStreamInstallation<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::StreamHandle, PcuError> {
            Ok(TestPersistentHandle)
        }
    }

    struct CommandOnlyBackend(TestBackend);

    impl PcuBaseContract for CommandOnlyBackend {
        fn support(&self) -> PcuSupport {
            self.0.support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.0.executors()
        }
    }

    impl PcuDirectCommandBackend for CommandOnlyBackend {
        type CommandHandle = TestFiniteHandle;

        fn submit_command_direct(
            &self,
            _submission: PcuCommandSubmission<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::CommandHandle, PcuError> {
            Ok(TestFiniteHandle)
        }
    }

    struct SignalOnlyBackend(TestBackend);

    impl PcuBaseContract for SignalOnlyBackend {
        fn support(&self) -> PcuSupport {
            self.0.support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.0.executors()
        }
    }

    impl PcuDirectSignalBackend for SignalOnlyBackend {
        type SignalHandle = TestPersistentHandle;

        fn install_signal_direct(
            &self,
            _installation: PcuSignalInstallation<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::SignalHandle, PcuError> {
            Ok(TestPersistentHandle)
        }
    }

    struct TestStreamLease {
        owner: u8,
        generation: u32,
        executor: crate::PcuExecutorId,
    }

    #[derive(Debug)]
    struct SelectedStreamHandle(crate::PcuExecutorId);

    impl PcuPersistentHandle for SelectedStreamHandle {
        fn state(&self) -> Result<PcuPersistentState, PcuError> {
            Ok(PcuPersistentState::Dormant)
        }

        fn start(&mut self) -> Result<(), PcuError> {
            Ok(())
        }

        fn stop(&mut self) -> Result<(), PcuError> {
            Ok(())
        }

        fn uninstall(self) -> Result<(), PcuError> {
            Ok(())
        }
    }

    struct ExclusiveStreamTestBackend {
        owner: u8,
        generation: u32,
    }

    const SELECTABLE_EXECUTORS: [PcuExecutorDescriptor; 2] = [
        PcuExecutorDescriptor {
            id: PcuExecutorId(1),
            ..DIRECT_EXECUTOR[0]
        },
        PcuExecutorDescriptor {
            id: PcuExecutorId(2),
            ..DIRECT_EXECUTOR[0]
        },
    ];

    impl PcuBaseContract for ExclusiveStreamTestBackend {
        fn support(&self) -> PcuSupport {
            direct_backend().support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            &SELECTABLE_EXECUTORS
        }
    }

    impl PcuExclusiveStreamBackend for ExclusiveStreamTestBackend {
        type Lease = TestStreamLease;
        type StreamHandle = SelectedStreamHandle;

        fn claim_stream_executor(
            &self,
            executor: crate::PcuExecutorId,
        ) -> Result<Self::Lease, PcuError> {
            if self.executor(executor).is_none() {
                return Err(PcuError::invalid());
            }
            Ok(TestStreamLease {
                owner: self.owner,
                generation: self.generation,
                executor,
            })
        }

        fn lease_executor(&self, lease: &Self::Lease) -> crate::PcuExecutorId {
            lease.executor
        }

        fn install_stream_on_lease_direct(
            &self,
            lease: &mut Self::Lease,
            _installation: PcuStreamInstallation<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::StreamHandle, PcuError> {
            if lease.owner != self.owner
                || lease.generation != self.generation
                || self.executor(lease.executor).is_none()
            {
                return Err(PcuError::state_conflict());
            }
            Ok(SelectedStreamHandle(lease.executor))
        }
    }

    struct DispatchOnlyBackend(TestBackend);

    impl PcuBaseContract for DispatchOnlyBackend {
        fn support(&self) -> PcuSupport {
            self.0.support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.0.executors()
        }
    }

    impl PcuDirectDispatchFamilyBackend for DispatchOnlyBackend {
        type DispatchHandle = TestFiniteHandle;

        fn submit_dispatch_direct(
            &self,
            _submission: PcuDispatchSubmission<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::DispatchHandle, PcuError> {
            Ok(TestFiniteHandle)
        }
    }

    struct TransactionOnlyBackend(TestBackend);

    impl PcuBaseContract for TransactionOnlyBackend {
        fn support(&self) -> PcuSupport {
            self.0.support()
        }

        fn executors(&self) -> &'static [PcuExecutorDescriptor] {
            self.0.executors()
        }
    }

    impl PcuDirectTransactionBackend for TransactionOnlyBackend {
        type TransactionHandle = TestFiniteHandle;

        fn submit_transaction_direct(
            &self,
            _submission: super::PcuTransactionSubmission<'_>,
            _bindings: PcuInvocationBindings<'_>,
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<Self::TransactionHandle, PcuError> {
            Ok(TestFiniteHandle)
        }
    }

    const DIRECT_EXECUTOR: [PcuExecutorDescriptor; 1] = [PcuExecutorDescriptor {
        id: PcuExecutorId(1),
        name: "cpu",
        class: PcuExecutorClass::Cpu,
        origin: PcuExecutorOrigin::Synthetic,
        support: PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::all(),
            dispatch_policy: PcuDispatchPolicyCaps::SERIAL
                .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION)
                .union(PcuDispatchPolicyCaps::PERSISTENT_INSTALL),
            value_types: crate::PcuValueTypeCaps::UINT32
                .union(crate::PcuValueTypeCaps::SCALAR_VALUES),
            dispatch_instructions: PcuDispatchOpCaps::ALU_ADD,
            dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
            stream_instructions: PcuStreamCapabilities::FIFO_INPUT
                .union(PcuStreamCapabilities::FIFO_OUTPUT)
                .union(PcuStreamCapabilities::BIT_INVERT),
            command_instructions: crate::PcuCommandOpCaps::WRITE,
            transaction_features: crate::PcuTransactionFeatureCaps::empty(),
            signal_instructions: crate::PcuSignalOpCaps::ACK,
        },
    }];

    fn direct_backend() -> TestBackend {
        let mut support = PcuSupport::unsupported();
        support.caps = PcuCaps::DISPATCH;
        support.executor_count = 1;
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(PcuPrimitiveCaps::all(), PcuPrimitiveCaps::all()),
        };
        support.value_type_support = PcuFeatureSupport::new(
            crate::PcuValueTypeCaps::UINT32.union(crate::PcuValueTypeCaps::SCALAR_VALUES),
            crate::PcuValueTypeCaps::empty(),
        );
        support.dispatch_support = PcuDispatchSupport {
            flags: PcuDispatchPolicyCaps::SERIAL
                .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION)
                .union(PcuDispatchPolicyCaps::PERSISTENT_INSTALL),
            instructions: PcuFeatureSupport::new(
                PcuDispatchOpCaps::ALU_ADD,
                PcuDispatchOpCaps::empty(),
            ),
            features: PcuFeatureSupport::new(
                crate::PcuDispatchFeatureCaps::empty(),
                crate::PcuDispatchFeatureCaps::empty(),
            ),
        };
        support.stream_support = crate::PcuStreamSupport {
            instructions: PcuFeatureSupport::new(
                PcuStreamCapabilities::FIFO_INPUT
                    .union(PcuStreamCapabilities::FIFO_OUTPUT)
                    .union(PcuStreamCapabilities::BIT_INVERT),
                PcuStreamCapabilities::empty(),
            ),
        };
        support.command_support = crate::PcuCommandSupport {
            instructions: PcuFeatureSupport::new(
                crate::PcuCommandOpCaps::WRITE,
                crate::PcuCommandOpCaps::WRITE,
            ),
        };
        support.transaction_support = crate::PcuTransactionSupport {
            features: PcuFeatureSupport::new(
                crate::PcuTransactionFeatureCaps::empty(),
                crate::PcuTransactionFeatureCaps::empty(),
            ),
        };
        support.signal_support = crate::PcuSignalSupport {
            instructions: PcuFeatureSupport::new(
                crate::PcuSignalOpCaps::ACK,
                crate::PcuSignalOpCaps::empty(),
            ),
        };
        TestBackend {
            support,
            executors: &DIRECT_EXECUTOR,
        }
    }

    fn cpu_fallback_only_backend() -> TestBackend {
        let mut backend = direct_backend();
        backend.support.command_support = crate::PcuCommandSupport {
            instructions: PcuFeatureSupport::new(
                crate::PcuCommandOpCaps::empty(),
                crate::PcuCommandOpCaps::WRITE,
            ),
        };
        backend.executors = &[];
        backend
    }

    #[test]
    fn blanket_impl_requires_direct_support_instead_of_cpu_fallback() {
        let backend = cpu_fallback_only_backend();
        let builder = PcuCommandKernelBuilder::<2>::new(1, "write")
            .with_step(
                Some("write"),
                PcuCommandOp::Write {
                    target: crate::PcuTarget::Named("reg"),
                    value: crate::PcuOperand::Immediate(PcuParameterValue::U32(7)),
                },
            )
            .expect("builder should accept one command step");
        let kernel = builder.ir();

        let result = backend.submit_command(
            PcuCommandSubmission { kernel: &kernel },
            PcuInvocationParameters::empty(),
        );

        assert_eq!(
            result
                .expect_err("fallback-only support must not route directly")
                .kind(),
            crate::PcuErrorKind::Unsupported
        );
    }

    #[test]
    fn blanket_impl_validates_dispatch_shape_before_backend_execution() {
        let backend = direct_backend();
        let builder = PcuDispatchKernelBuilder::<2>::new(7, "main", [4, 1, 1])
            .with_type_caps(
                crate::PcuValueTypeCaps::UINT32 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            )
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("builder should accept one dispatch op");
        let kernel = builder.ir();

        let result = backend.submit_dispatch(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(3).expect("nonzero")),
            },
            PcuInvocationBindings::empty(),
            PcuInvocationParameters::empty(),
        );

        assert_eq!(
            result.expect_err("shape mismatch must be rejected").kind(),
            crate::PcuErrorKind::Invalid
        );
    }

    #[test]
    fn dispatch_submission_rejects_aliased_binding_addresses() {
        let bindings = [
            crate::PcuBinding::value(
                Some("input"),
                0,
                0,
                crate::PcuBindingStorageClass::Storage,
                crate::PcuBindingAccess::ReadOnly,
                PcuValueType::u32(),
            ),
            crate::PcuBinding::value(
                Some("output"),
                0,
                0,
                crate::PcuBindingStorageClass::Storage,
                crate::PcuBindingAccess::WriteOnly,
                PcuValueType::u32(),
            ),
        ];
        let builder =
            PcuDispatchKernelBuilder::<0>::new(8, "main", [1, 1, 1]).with_bindings(&bindings);
        let kernel = builder.ir();
        let submission = PcuDispatchSubmission {
            kernel: &kernel,
            shape: PcuInvocationShape::invocations(NonZeroU32::new(1).expect("nonzero")),
        };

        assert_eq!(
            super::validate_dispatch_submission(submission),
            Err(PcuError::invalid())
        );
    }

    #[test]
    fn blanket_impl_reports_unsupported_dispatch_type_floor() {
        let backend = direct_backend();
        let builder = PcuDispatchKernelBuilder::<2>::new(8, "main", [1, 1, 1])
            .with_type_caps(
                crate::PcuValueTypeCaps::FLOAT64 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            )
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("builder should accept one dispatch op");
        let kernel = builder.ir();

        let result = backend.submit_dispatch(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(1).expect("nonzero")),
            },
            PcuInvocationBindings::empty(),
            PcuInvocationParameters::empty(),
        );

        assert_eq!(
            result
                .expect_err("unsupported dispatch types must be rejected")
                .kind(),
            crate::PcuErrorKind::UnsupportedTypeSupport
        );
    }

    #[test]
    fn blanket_impl_reports_unsupported_dispatch_feature_floor() {
        let backend = direct_backend();
        let parameters = [crate::PcuParameter {
            slot: crate::PcuParameterSlot(0),
            name: Some("scale"),
            value_type: crate::PcuValueType::u32(),
        }];
        let builder = PcuDispatchKernelBuilder::<2>::new(9, "main", [1, 1, 1])
            .with_parameters(&parameters)
            .with_type_caps(
                crate::PcuValueTypeCaps::UINT32 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            )
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("builder should accept one dispatch op");
        let kernel = builder.ir();

        let result = backend.submit_dispatch(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(1).expect("nonzero")),
            },
            PcuInvocationBindings::empty(),
            PcuInvocationParameters::empty(),
        );

        assert_eq!(
            result
                .expect_err("unsupported dispatch features must be rejected")
                .kind(),
            crate::PcuErrorKind::UnsupportedFeatureSupport
        );
    }

    #[test]
    fn blanket_impl_validates_stream_binding_targets() {
        let backend = direct_backend();
        let builder = PcuStreamKernelBuilder::<2>::words(9, "stream")
            .bit_invert()
            .expect("builder should accept one stream pattern");
        let kernel = builder.ir();
        let mut output = [0_u32; 4];
        let bindings = [PcuInvocationBinding {
            target: PcuInvocationTarget::Port("missing"),
            buffer: PcuInvocationBuffer::WordsOut(&mut output),
        }];

        let result = backend.install_stream(
            PcuStreamInstallation { kernel: &kernel },
            PcuInvocationBindings {
                bindings: &bindings,
            },
            PcuInvocationParameters::empty(),
        );

        assert_eq!(
            result
                .expect_err("unknown binding target must be rejected")
                .kind(),
            crate::PcuErrorKind::Invalid
        );
    }

    #[test]
    fn stream_only_backend_needs_no_unrelated_family_methods() {
        let backend = StreamOnlyBackend(direct_backend());
        let builder = PcuStreamKernelBuilder::<1>::words(12, "stream")
            .bit_invert()
            .expect("builder should accept one stream pattern");
        let kernel = builder.ir();

        let handle = PcuDirectStreamBackend::install_stream(
            &backend,
            PcuStreamInstallation { kernel: &kernel },
            PcuInvocationBindings::empty(),
            PcuInvocationParameters::empty(),
        )
        .expect("stream-only backend should install a supported stream");
        assert_eq!(
            handle.state().expect("state is available"),
            PcuPersistentState::Dormant
        );
    }

    #[test]
    fn exclusive_stream_lease_selects_executor_and_rejects_foreign_or_stale_leases() {
        let backend = ExclusiveStreamTestBackend {
            owner: 1,
            generation: 9,
        };
        let builder = PcuStreamKernelBuilder::<1>::words(17, "leased")
            .bit_invert()
            .expect("builder should accept one pattern");
        let kernel = builder.ir();
        let mut selected = backend
            .claim_stream_executor(PcuExecutorId(2))
            .expect("executor 2 should be claimable");
        let handle = backend
            .install_stream_on_lease(
                &mut selected,
                PcuStreamInstallation { kernel: &kernel },
                PcuInvocationBindings::empty(),
                PcuInvocationParameters::empty(),
            )
            .expect("selected executor should install stream");
        assert_eq!(handle.0, PcuExecutorId(2));

        let foreign_backend = ExclusiveStreamTestBackend {
            owner: 2,
            generation: 9,
        };
        let mut foreign = foreign_backend
            .claim_stream_executor(PcuExecutorId(2))
            .expect("executor should be claimable on second provider");
        assert_eq!(
            backend
                .install_stream_on_lease(
                    &mut foreign,
                    PcuStreamInstallation { kernel: &kernel },
                    PcuInvocationBindings::empty(),
                    PcuInvocationParameters::empty(),
                )
                .expect_err("foreign lease must be rejected")
                .kind(),
            crate::PcuErrorKind::StateConflict
        );

        let mut stale = backend
            .claim_stream_executor(PcuExecutorId(1))
            .expect("executor should be claimable");
        stale.generation = 8;
        assert_eq!(
            backend
                .install_stream_on_lease(
                    &mut stale,
                    PcuStreamInstallation { kernel: &kernel },
                    PcuInvocationBindings::empty(),
                    PcuInvocationParameters::empty(),
                )
                .expect_err("stale lease must be rejected")
                .kind(),
            crate::PcuErrorKind::StateConflict
        );
    }

    #[test]
    fn command_only_backend_needs_no_unrelated_family_methods() {
        let backend = CommandOnlyBackend(direct_backend());
        let builder = PcuCommandKernelBuilder::<1>::new(13, "write")
            .with_step(
                Some("write"),
                PcuCommandOp::Write {
                    target: crate::PcuTarget::Named("reg"),
                    value: crate::PcuOperand::Immediate(PcuParameterValue::U32(7)),
                },
            )
            .expect("builder should accept one command step");
        let kernel = builder.ir();

        let handle = PcuCommandBackend::submit_command(
            &backend,
            PcuCommandSubmission { kernel: &kernel },
            PcuInvocationParameters::empty(),
        )
        .expect("command-only backend should accept a supported command");
        assert_eq!(
            handle.state().expect("state is available"),
            PcuFiniteState::Complete
        );
    }

    #[test]
    fn command_admission_rejects_undefined_typed_result() {
        let backend = CommandOnlyBackend(direct_backend());
        let builder = PcuCommandKernelBuilder::<1>::new(19, "bad-result")
            .with_step(
                Some("write"),
                PcuCommandOp::Write {
                    target: crate::PcuTarget::Named("reg"),
                    value: crate::PcuOperand::Result(crate::model::PcuCommandResultId(3)),
                },
            )
            .expect("builder should accept one command step");
        let kernel = builder.ir();
        let error = PcuCommandBackend::submit_command(
            &backend,
            PcuCommandSubmission { kernel: &kernel },
            PcuInvocationParameters::empty(),
        )
        .expect_err("undefined result must be rejected before backend execution");
        assert_eq!(error.kind(), crate::PcuErrorKind::Invalid);
    }

    #[test]
    fn signal_only_backend_needs_no_unrelated_family_methods() {
        let backend = SignalOnlyBackend(direct_backend());
        let builder = crate::model::PcuSignalKernelBuilder::<1>::new(
            14,
            "ack",
            crate::PcuSignalTriggerKind::Software,
        )
        .with_op(crate::PcuSignalOp::Ack)
        .expect("builder should accept one signal op");
        let kernel = builder.ir();

        let handle = PcuSignalBackend::install_signal(
            &backend,
            PcuSignalInstallation { kernel: &kernel },
            PcuInvocationParameters::empty(),
        )
        .expect("signal-only backend should accept a supported signal");
        assert_eq!(
            handle.state().expect("state is available"),
            PcuPersistentState::Dormant
        );
    }

    #[test]
    fn dispatch_only_backend_needs_no_unrelated_family_methods() {
        let backend = DispatchOnlyBackend(direct_backend());
        let builder = PcuDispatchKernelBuilder::<1>::new(15, "dispatch", [1, 1, 1])
            .with_type_caps(
                crate::PcuValueTypeCaps::UINT32 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            )
            .with_arithmetic_op(PcuDispatchAluOp::Add)
            .expect("builder should accept one dispatch operation");
        let kernel = builder.ir();

        let handle = PcuDispatchBackend::submit_dispatch(
            &backend,
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::invocations(NonZeroU32::new(1).expect("nonzero")),
            },
            PcuInvocationBindings::empty(),
            PcuInvocationParameters::empty(),
        )
        .expect("dispatch-only backend should accept a supported dispatch");
        assert_eq!(
            handle.state().expect("state is available"),
            PcuFiniteState::Complete
        );
    }

    #[test]
    fn transaction_only_backend_needs_no_unrelated_family_methods() {
        let backend = TransactionOnlyBackend(direct_backend());
        let builder = crate::model::PcuTransactionKernelBuilder::new(16, "transaction");
        let kernel = builder.ir();

        let handle = PcuTransactionBackend::submit_transaction(
            &backend,
            super::PcuTransactionSubmission { kernel: &kernel },
            PcuInvocationBindings::empty(),
            PcuInvocationParameters::empty(),
        )
        .expect("transaction-only backend should accept a supported transaction");
        assert_eq!(
            handle.state().expect("state is available"),
            PcuFiniteState::Complete
        );
    }

    #[test]
    fn blanket_impl_validates_command_parameters_before_backend_execution() {
        let backend = direct_backend();
        let parameter = PcuParameter::named(PcuParameterSlot(0), "value", PcuValueType::u32());
        let parameters = [parameter];
        let builder = PcuCommandKernelBuilder::<2>::new(11, "write")
            .with_parameters(&parameters)
            .with_step(
                Some("write"),
                PcuCommandOp::Write {
                    target: crate::PcuTarget::Named("reg"),
                    value: crate::PcuOperand::Parameter(PcuParameterSlot(0)),
                },
            )
            .expect("builder should accept one command step");
        let kernel = builder.ir();

        let result = backend.submit_command(
            PcuCommandSubmission { kernel: &kernel },
            PcuInvocationParameters::empty(),
        );

        assert_eq!(
            result
                .expect_err("missing required parameter must be rejected")
                .kind(),
            crate::PcuErrorKind::Invalid
        );
    }
}
