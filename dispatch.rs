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
    PcuError,
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

/// Logical invocation context surfaced to one dispatch-style kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDispatchContext {
    pub thread_id: u32,
    pub thread_count: NonZeroU32,
}

impl PcuDispatchContext {
    /// Creates one checked logical invocation context.
    #[must_use]
    pub const fn new(thread_id: u32, thread_count: NonZeroU32) -> Self {
        Self {
            thread_id,
            thread_count,
        }
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

fn validate_direct_kernel_support(
    backend: &impl PcuBaseContract,
    kernel: PcuKernel<'_>,
) -> Result<(), PcuError> {
    let kernel_supported = backend.any_executor_supports_kernel_direct(kernel);

    if let PcuKernel::Dispatch(dispatch) = kernel {
        if !kernel_supported {
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
                            .supports_dispatch_types_direct(dispatch.required_type_support())
                });
                if any_typed {
                    return Err(PcuError::unsupported_feature_support());
                }
                return Err(PcuError::unsupported_type_support());
            }
        }
    }

    if kernel_supported {
        return Ok(());
    }
    Err(PcuError::unsupported())
}

fn validate_parameters(
    signature: PcuKernelSignature<'_>,
    parameters: PcuInvocationParameters<'_>,
) -> Result<(), PcuError> {
    if parameters.validate_against(signature.parameters) {
        Ok(())
    } else {
        Err(PcuError::invalid())
    }
}

fn validate_invocation_bindings(
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

fn validate_dispatch_submission(submission: PcuDispatchSubmission<'_>) -> Result<(), PcuError> {
    let PcuInvocationTopology::Indexed { logical_shape } =
        submission.kernel.signature().invocation.topology
    else {
        return Err(PcuError::invalid());
    };

    let expected_threads = logical_shape
        .into_iter()
        .try_fold(1_u64, |product, axis| product.checked_mul(u64::from(axis)))
        .ok_or_else(PcuError::invalid)?;

    if expected_threads == 0 || expected_threads != u64::from(submission.shape.thread_count().get())
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

    use super::{
        PcuCommandSubmission,
        PcuDirectDispatchBackend,
        PcuDispatchContract,
        PcuDispatchSubmission,
        PcuFiniteHandle,
        PcuFiniteState,
        PcuPersistentHandle,
        PcuPersistentState,
        PcuStreamInstallation,
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
            dispatch_instructions: PcuDispatchOpCaps::ALU_ADD,
            dispatch_types: crate::PcuValueTypeCaps::UINT32
                .union(crate::PcuValueTypeCaps::SCALAR_VALUES),
            dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
            stream_instructions: PcuStreamCapabilities::FIFO_INPUT
                .union(PcuStreamCapabilities::FIFO_OUTPUT)
                .union(PcuStreamCapabilities::BIT_INVERT),
            command_instructions: crate::PcuCommandOpCaps::WRITE,
            transaction_features: crate::PcuTransactionFeatureCaps::empty(),
            signal_instructions: crate::PcuSignalOpCaps::empty(),
        },
    }];

    fn direct_backend() -> TestBackend {
        let mut support = PcuSupport::unsupported();
        support.caps = PcuCaps::DISPATCH;
        support.executor_count = 1;
        support.primitive_support = PcuPrimitiveSupport {
            primitives: PcuFeatureSupport::new(PcuPrimitiveCaps::all(), PcuPrimitiveCaps::all()),
        };
        support.dispatch_support = PcuDispatchSupport {
            flags: PcuDispatchPolicyCaps::SERIAL
                .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION)
                .union(PcuDispatchPolicyCaps::PERSISTENT_INSTALL),
            instructions: PcuFeatureSupport::new(
                PcuDispatchOpCaps::ALU_ADD,
                PcuDispatchOpCaps::empty(),
            ),
            types: PcuFeatureSupport::new(
                crate::PcuValueTypeCaps::UINT32.union(crate::PcuValueTypeCaps::SCALAR_VALUES),
                crate::PcuValueTypeCaps::empty(),
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
                shape: PcuInvocationShape::threads(NonZeroU32::new(3).expect("nonzero")),
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
                shape: PcuInvocationShape::threads(NonZeroU32::new(1).expect("nonzero")),
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
                shape: PcuInvocationShape::threads(NonZeroU32::new(1).expect("nonzero")),
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
