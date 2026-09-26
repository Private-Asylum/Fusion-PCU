//! Routing and scheduling contracts for PCU execution models.
//!
//! This module defines how model-local kernels reach an execution substrate.
//! It intentionally does not define:
//! - backend preference policy
//! - provider selection
//! - platform lowering
//! - CPU fallback doctrine

mod backend;
mod context;
mod submission;
mod validation;

pub use backend::*;
pub use context::*;
pub use submission::*;
pub use validation::{
    validate_dispatch_submission,
    validate_invocation_bindings,
    validate_parameters,
};
use validation::validate_direct_kernel_support;

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
            dispatch_scalar_alu: crate::PcuDispatchScalarAluSupport::empty(),
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
            scalar_alu: PcuFeatureSupport::new(
                crate::PcuDispatchScalarAluSupport::empty(),
                crate::PcuDispatchScalarAluSupport::empty(),
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
