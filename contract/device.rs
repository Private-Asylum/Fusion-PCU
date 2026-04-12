//! Shared generic PCU execution-substrate identifiers and descriptor vocabulary.

use super::caps::{
    PcuCommandOpCaps,
    PcuDispatchOpCaps,
    PcuDispatchPolicyCaps,
    PcuPrimitiveCaps,
    PcuSignalOpCaps,
    PcuTransactionFeatureCaps,
};
use crate::{
    PcuKernel,
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
    pub dispatch_instructions: PcuDispatchOpCaps,
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
            dispatch_instructions: PcuDispatchOpCaps::empty(),
            stream_instructions: PcuStreamCapabilities::empty(),
            command_instructions: PcuCommandOpCaps::empty(),
            transaction_features: PcuTransactionFeatureCaps::empty(),
            signal_instructions: PcuSignalOpCaps::empty(),
        }
    }

    #[must_use]
    pub fn supports_kernel_direct(self, kernel: PcuKernel<'_>) -> bool {
        match kernel {
            PcuKernel::Dispatch(kernel) => {
                self.primitives.contains(PcuPrimitiveCaps::DISPATCH)
                    && self
                        .dispatch_policy
                        .contains(kernel.required_dispatch_policy())
                    && self
                        .dispatch_instructions
                        .contains(kernel.required_instruction_support())
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
        PcuDispatchPolicyCaps,
        PcuKernel,
        PcuKernelId,
        PcuPrimitiveCaps,
        PcuTarget,
    };

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
    fn executor_support_requires_dispatch_policy() {
        let support = PcuExecutorSupport {
            primitives: PcuPrimitiveCaps::COMMAND,
            dispatch_policy: PcuDispatchPolicyCaps::empty(),
            dispatch_instructions: crate::PcuDispatchOpCaps::empty(),
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
                dispatch_instructions: crate::PcuDispatchOpCaps::empty(),
                stream_instructions: crate::PcuStreamCapabilities::empty(),
                command_instructions: PcuCommandOpCaps::WRITE,
                transaction_features: crate::PcuTransactionFeatureCaps::empty(),
                signal_instructions: crate::PcuSignalOpCaps::empty(),
            },
        };

        assert!(descriptor.supports_kernel_direct(command_kernel()));
    }
}

/// Exclusive execution-substrate claim returned by one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuExecutorClaim {
    pub(crate) executor: PcuExecutorId,
}

impl PcuExecutorClaim {
    /// Creates one exclusive execution-substrate claim from one validated executor id.
    #[must_use]
    pub const fn new(executor: PcuExecutorId) -> Self {
        Self { executor }
    }

    /// Returns the claimed execution substrate identifier.
    #[must_use]
    pub const fn executor(self) -> PcuExecutorId {
        self.executor
    }
}
