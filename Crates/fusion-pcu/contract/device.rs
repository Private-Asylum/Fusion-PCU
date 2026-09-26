//! Shared generic PCU execution-substrate identifiers and descriptor vocabulary.

use super::caps::{
    PcuCommandOpCaps,
    PcuDispatchOpCaps,
    PcuDispatchPolicyCaps,
    PcuDispatchScalarAluSupport,
    PcuPrimitiveCaps,
    PcuSignalOpCaps,
    PcuTransactionFeatureCaps,
};
use crate::{
    PcuDispatchFeatureCaps,
    PcuKernel,
    PcuValueTypeCaps,
    PcuStreamCapabilities,
};

/// Opaque PCU execution substrate identifier surfaced by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutorId(pub u8);

/// Coarse substrate class for one surfaced execution substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuExecutorClass {
    Cpu,
    Compute,
    Io,
    Signal,
    Neural,
    Media,
    Adapter,
    Remote,
    Other,
}

/// Provenance for one surfaced execution substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuExecutorOrigin {
    Synthetic,
    TopologyBound,
    Projected,
}

/// Static descriptor for one surfaced execution substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutorDescriptor {
    pub id: PcuExecutorId,
    pub name: &'static str,
    pub class: PcuExecutorClass,
    pub origin: PcuExecutorOrigin,
    pub support: PcuExecutorSupport,
}

impl PcuExecutorDescriptor {
    #[must_use]
    pub fn supports_kernel_direct(self, kernel: PcuKernel<'_>) -> bool {
        self.support.supports_kernel_direct(kernel)
    }
}

/// Direct support truth for one surfaced execution substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutorSupport {
    pub primitives: PcuPrimitiveCaps,
    pub dispatch_policy: PcuDispatchPolicyCaps,
    pub value_types: PcuValueTypeCaps,
    pub dispatch_instructions: PcuDispatchOpCaps,
    pub dispatch_scalar_alu: PcuDispatchScalarAluSupport,
    pub dispatch_features: PcuDispatchFeatureCaps,
    pub stream_instructions: PcuStreamCapabilities,
    pub command_instructions: PcuCommandOpCaps,
    pub transaction_features: PcuTransactionFeatureCaps,
    pub signal_instructions: PcuSignalOpCaps,
}

impl PcuExecutorSupport {
    #[must_use]
    pub const fn unsupported() -> Self {
        Self {
            primitives: PcuPrimitiveCaps::empty(),
            dispatch_policy: PcuDispatchPolicyCaps::empty(),
            value_types: PcuValueTypeCaps::empty(),
            dispatch_instructions: PcuDispatchOpCaps::empty(),
            dispatch_scalar_alu: PcuDispatchScalarAluSupport::empty(),
            dispatch_features: PcuDispatchFeatureCaps::empty(),
            stream_instructions: PcuStreamCapabilities::empty(),
            command_instructions: PcuCommandOpCaps::empty(),
            transaction_features: PcuTransactionFeatureCaps::empty(),
            signal_instructions: PcuSignalOpCaps::empty(),
        }
    }

    #[must_use]
    pub const fn supports_value_types_direct(self, required: PcuValueTypeCaps) -> bool {
        self.value_types.contains(required)
    }

    #[must_use]
    pub const fn supports_dispatch_features_direct(self, required: PcuDispatchFeatureCaps) -> bool {
        self.dispatch_features.contains(required)
    }

    #[must_use]
    pub fn supports_dispatch_direct_structure(
        self,
        kernel: crate::PcuDispatchKernelIr<'_>,
    ) -> bool {
        !kernel.has_non_scalar_alu_type()
            && self.primitives.contains(PcuPrimitiveCaps::DISPATCH)
            && self
                .dispatch_policy
                .contains(kernel.required_dispatch_policy())
            && self
                .dispatch_instructions
                .contains(kernel.required_instruction_support())
            && self
                .dispatch_scalar_alu
                .supports(kernel.required_scalar_alu_support())
    }

    #[must_use]
    pub fn supports_kernel_direct(self, kernel: PcuKernel<'_>) -> bool {
        match kernel {
            PcuKernel::Dispatch(kernel) => {
                self.supports_dispatch_direct_structure(kernel)
                    && self.supports_value_types_direct(kernel.required_type_support())
                    && self.supports_dispatch_features_direct(kernel.required_feature_support())
            }
            PcuKernel::Stream(kernel) => {
                self.primitives.contains(PcuPrimitiveCaps::STREAM)
                    && self
                        .dispatch_policy
                        .contains(kernel.required_dispatch_policy())
                    && self
                        .stream_instructions
                        .contains(kernel.required_instruction_support())
            }
            PcuKernel::Command(kernel) => {
                self.primitives.contains(PcuPrimitiveCaps::COMMAND)
                    && self
                        .dispatch_policy
                        .contains(kernel.required_dispatch_policy())
                    && self
                        .command_instructions
                        .contains(kernel.required_instruction_support())
            }
            PcuKernel::Transaction(kernel) => {
                self.primitives.contains(PcuPrimitiveCaps::TRANSACTION)
                    && self
                        .dispatch_policy
                        .contains(kernel.required_dispatch_policy())
                    && self
                        .transaction_features
                        .contains(kernel.required_features())
            }
            PcuKernel::Signal(kernel) => {
                self.primitives.contains(PcuPrimitiveCaps::SIGNAL)
                    && self
                        .dispatch_policy
                        .contains(kernel.required_dispatch_policy())
                    && self
                        .signal_instructions
                        .contains(kernel.required_instruction_support())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuExecutorSupport,
        PcuExecutorDescriptor,
        PcuExecutorClass,
        PcuExecutorId,
        PcuExecutorOrigin,
    };
    use crate::{
        PcuCommandKernelIr,
        PcuCommandOp,
        PcuCommandOpCaps,
        PcuCommandStep,
        PcuDispatchFeatureCaps,
        PcuDispatchAluOp,
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchOpCaps,
        PcuDispatchScalarAluSupport,
        PcuDispatchValueId,
        PcuDispatchPolicyCaps,
        PcuKernel,
        PcuKernelId,
        PcuPrimitiveCaps,
        PcuTarget,
        PcuValueTypeCaps,
        PcuValueType,
    };
    use crate::model::PcuDispatchKernelBuilder;

    fn command_kernel() -> PcuKernel<'static> {
        PcuKernel::Command(PcuCommandKernelIr {
            id: PcuKernelId(11),
            entry_point: "write",
            bindings: &[],
            ports: &[],
            parameters: &[],
            steps: &[PcuCommandStep {
                name: Some("write"),
                op: PcuCommandOp::Write {
                    target: PcuTarget::Named("register"),
                    value: crate::PcuOperand::PreviousResult,
                },
            }],
        })
    }

    #[test]
    fn executor_requires_type_specific_alu_support() {
        let mut support = PcuExecutorSupport::unsupported();
        support.primitives = PcuPrimitiveCaps::DISPATCH;
        support.dispatch_policy = PcuDispatchPolicyCaps::ORDERED_SUBMISSION;
        support.value_types =
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES;
        support.dispatch_instructions = PcuDispatchOpCaps::ALU_ADD;
        support.dispatch_scalar_alu = PcuDispatchScalarAluSupport::empty()
            .with(crate::PcuScalarType::F32, PcuDispatchOpCaps::ALU_ADD);
        let ops = [PcuDispatchOp::Data(PcuDispatchDataOp::Alu {
            result: PcuDispatchValueId(3),
            op: PcuDispatchAluOp::Add,
            value_type: PcuValueType::u32(),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        })];
        let kernel = PcuDispatchKernelIr {
            id: PcuKernelId(12),
            entry: PcuDispatchEntryPoint {
                name: "u32-add-capability",
                logical_shape: [1, 1, 1],
            },
            bindings: &[],
            ports: &[],
            parameters: &[],
            ops: &ops,
            type_caps: PcuValueTypeCaps::empty(),
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };
        assert!(!support.supports_kernel_direct(PcuKernel::Dispatch(kernel)));
        support.dispatch_scalar_alu = support
            .dispatch_scalar_alu
            .with(crate::PcuScalarType::U32, PcuDispatchOpCaps::ALU_ADD);
        assert!(support.supports_kernel_direct(PcuKernel::Dispatch(kernel)));
    }

    #[test]
    fn executor_support_requires_dispatch_policy() {
        let support = PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::COMMAND,
            dispatch_policy: PcuDispatchPolicyCaps::empty(),
            value_types: crate::PcuValueTypeCaps::empty(),
            dispatch_instructions: crate::PcuDispatchOpCaps::empty(),
            dispatch_scalar_alu: crate::PcuDispatchScalarAluSupport::empty(),
            dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
            stream_instructions: crate::PcuStreamCapabilities::empty(),
            command_instructions: PcuCommandOpCaps::WRITE,
            transaction_features: crate::PcuTransactionFeatureCaps::empty(),
            signal_instructions: crate::PcuSignalOpCaps::empty(),
        };

        assert!(!support.supports_kernel_direct(command_kernel()));
    }

    #[test]
    fn executor_descriptor_reports_direct_command_support() {
        let descriptor = PcuExecutorDescriptor {
            id: PcuExecutorId(3),
            name: "cpu",
            class: PcuExecutorClass::Cpu,
            origin: PcuExecutorOrigin::Synthetic,
            support: PcuExecutorSupport {
                primitives: PcuPrimitiveCaps::COMMAND,
                dispatch_policy: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
                value_types: crate::PcuValueTypeCaps::empty(),
                dispatch_instructions: crate::PcuDispatchOpCaps::empty(),
                dispatch_scalar_alu: crate::PcuDispatchScalarAluSupport::empty(),
                dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
                stream_instructions: crate::PcuStreamCapabilities::empty(),
                command_instructions: PcuCommandOpCaps::WRITE,
                transaction_features: crate::PcuTransactionFeatureCaps::empty(),
                signal_instructions: crate::PcuSignalOpCaps::empty(),
            },
        };

        assert!(descriptor.supports_kernel_direct(command_kernel()));
    }

    #[test]
    fn executor_support_rejects_dispatch_when_types_are_missing() {
        let support = PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::DISPATCH,
            dispatch_policy: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            value_types: crate::PcuValueTypeCaps::UINT32 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            dispatch_instructions: crate::PcuDispatchOpCaps::ALU_ADD,
            dispatch_scalar_alu: crate::PcuDispatchScalarAluSupport::empty(),
            dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
            stream_instructions: crate::PcuStreamCapabilities::empty(),
            command_instructions: PcuCommandOpCaps::empty(),
            transaction_features: crate::PcuTransactionFeatureCaps::empty(),
            signal_instructions: crate::PcuSignalOpCaps::empty(),
        };
        let kernel_builder = PcuDispatchKernelBuilder::<1>::new(5, "main", [1, 1, 1])
            .with_type_caps(PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES)
            .with_arithmetic_op(crate::PcuDispatchAluOp::Add)
            .expect("builder should accept one dispatch op");
        let kernel = kernel_builder.ir();

        assert!(support.supports_kernel_direct(PcuKernel::Dispatch(kernel)));
        assert!(
            !support.supports_value_types_direct(crate::PcuValueTypeCaps::for_value_type(
                PcuValueType::Vector {
                    scalar: crate::PcuScalarType::U32,
                    lanes: 4,
                }
            ),)
        );
    }

    #[test]
    fn executor_support_rejects_dispatch_when_features_are_missing() {
        let support = PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::DISPATCH,
            dispatch_policy: PcuDispatchPolicyCaps::ORDERED_SUBMISSION,
            value_types: crate::PcuValueTypeCaps::UINT32 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            dispatch_instructions: crate::PcuDispatchOpCaps::ALU_ADD,
            dispatch_scalar_alu: crate::PcuDispatchScalarAluSupport::empty(),
            dispatch_features: crate::PcuDispatchFeatureCaps::empty(),
            stream_instructions: crate::PcuStreamCapabilities::empty(),
            command_instructions: PcuCommandOpCaps::empty(),
            transaction_features: crate::PcuTransactionFeatureCaps::empty(),
            signal_instructions: crate::PcuSignalOpCaps::empty(),
        };
        let parameters = [crate::PcuParameter {
            slot: crate::PcuParameterSlot(0),
            name: Some("scale"),
            value_type: PcuValueType::u32(),
        }];
        let builder = PcuDispatchKernelBuilder::<1>::new(6, "main", [1, 1, 1])
            .with_parameters(&parameters)
            .with_type_caps(
                crate::PcuValueTypeCaps::UINT32 | crate::PcuValueTypeCaps::SCALAR_VALUES,
            )
            .with_arithmetic_op(crate::PcuDispatchAluOp::Add)
            .expect("builder should accept one dispatch op");
        let kernel = builder.ir();

        assert!(
            kernel
                .required_feature_support()
                .contains(PcuDispatchFeatureCaps::INLINE_PARAMETERS)
        );
        assert!(!support.supports_kernel_direct(PcuKernel::Dispatch(kernel)));
        assert!(
            !support.supports_dispatch_features_direct(PcuDispatchFeatureCaps::INLINE_PARAMETERS)
        );
    }
}
