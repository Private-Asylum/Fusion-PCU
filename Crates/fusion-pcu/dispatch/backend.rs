//! Backend contracts for the PCU execution families.

use crate::contract::PcuKernelIrContract;
use crate::contract::{
    PcuBaseContract,
    PcuError,
    PcuExecutorId,
    PcuInvocationBindings,
    PcuInvocationParameters,
    PcuKernel,
};
use crate::validation::validate_command_kernel;
use super::{
    validate_direct_kernel_support,
    validate_dispatch_submission,
    validate_invocation_bindings,
    validate_parameters,
    PcuCommandSubmission,
    PcuDispatchSubmission,
    PcuFiniteHandle,
    PcuPersistentHandle,
    PcuSignalInstallation,
    PcuStreamInstallation,
    PcuTransactionSubmission,
};

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
