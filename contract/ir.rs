//! Canonical PCU IR contract surface.
//!
//! This module exposes:
//! - shared core vocabulary from `fusion-pcu::core`
//! - shared IR support vocabulary from `fusion-pcu::ir`
//! - model-local IR payloads from `fusion-pcu::model`
//! - backend-neutral validation results from `fusion-pcu::validation`

pub use crate::core::{
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuBuiltinValue,
    PcuImageBindingType,
    PcuImageDimension,
    PcuInvocationModel,
    PcuInvocationOrdering,
    PcuInvocationParallelism,
    PcuInvocationProgress,
    PcuInvocationTopology,
    PcuIrKind,
    PcuKernelId,
    PcuKernelIrContract,
    PcuKernelSignature,
    PcuParameter,
    PcuParameterBinding,
    PcuParameterSlot,
    PcuParameterValue,
    PcuPort,
    PcuPortBackpressure,
    PcuPortBlocking,
    PcuPortDirection,
    PcuPortRate,
    PcuPortReliability,
    PcuSamplerAddressMode,
    PcuSamplerBindingType,
    PcuSamplerCoordinateNormalization,
    PcuSamplerFilter,
    PcuSamplerMipmapMode,
    PcuScalarType,
    PcuValueTypeCaps,
    PcuValueType,
};
pub use crate::ir::{
    PcuSampleLevel,
    PcuSampleOp,
};
pub use crate::model::{
    PcuCommandEffectKind,
    PcuCommandKernelIr,
    PcuCommandModifyOp,
    PcuCommandOp,
    PcuCommandPredicate,
    PcuCommandStep,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchAluOp,
    PcuDispatchControlOp,
    PcuDispatchPortOp,
    PcuDispatchResourceOp,
    PcuDispatchSyncOp,
    PcuDispatchValueOp,
    PcuKernel,
    PcuOperand,
    PcuSignalKernelIr,
    PcuSignalOp,
    PcuSignalTriggerKind,
    PcuStreamCapabilities,
    PcuStreamKernelIr,
    PcuStreamPattern,
    PcuStreamValueType,
    PcuTarget,
    PcuTransactionAtomicity,
    PcuTransactionExclusivity,
    PcuTransactionKernelIr,
    PcuTransactionOrdering,
};
pub use crate::validation::{
    PcuSampleValidationError,
    PcuStreamSimpleTransformValidationError,
};

#[cfg(test)]
mod tests {
    use super::{
        PcuCommandEffectKind,
        PcuCommandKernelIr,
        PcuCommandOp,
        PcuCommandStep,
        PcuIrKind,
        PcuKernel,
        PcuKernelId,
        PcuKernelIrContract,
        PcuOperand,
        PcuParameterSlot,
        PcuSignalKernelIr,
        PcuSignalOp,
        PcuSignalTriggerKind,
        PcuTarget,
        PcuTransactionAtomicity,
        PcuTransactionExclusivity,
        PcuTransactionKernelIr,
        PcuTransactionOrdering,
    };

    #[test]
    fn command_ops_report_effect_kind() {
        let read = PcuCommandOp::Read {
            target: PcuTarget::Named("reg"),
        };
        let await_ready = PcuCommandOp::Await {
            predicate: super::PcuCommandPredicate::Ready(PcuTarget::Named("status")),
        };
        let write = PcuCommandOp::Write {
            target: PcuTarget::Named("reg"),
            value: PcuOperand::Parameter(PcuParameterSlot(0)),
        };

        assert_eq!(read.effect_kind(), PcuCommandEffectKind::Read);
        assert_eq!(await_ready.effect_kind(), PcuCommandEffectKind::Synchronize);
        assert_eq!(write.effect_kind(), PcuCommandEffectKind::Write);
    }

    #[test]
    fn command_transaction_and_signal_kernels_report_their_kinds() {
        let command = PcuKernel::Command(PcuCommandKernelIr {
            id: PcuKernelId(1),
            entry_point: "cmd",
            bindings: &[],
            ports: &[],
            parameters: &[],
            steps: &[PcuCommandStep {
                name: Some("write"),
                op: PcuCommandOp::Write {
                    target: PcuTarget::Named("reg"),
                    value: PcuOperand::Parameter(PcuParameterSlot(0)),
                },
            }],
        });
        let transaction = PcuKernel::Transaction(PcuTransactionKernelIr {
            id: PcuKernelId(2),
            entry_point: "txn",
            bindings: &[],
            ports: &[],
            parameters: &[],
            timeout_ticks: Some(32),
            atomicity: PcuTransactionAtomicity::Atomic,
            exclusivity: PcuTransactionExclusivity::Exclusive,
            ordering: PcuTransactionOrdering::InOrder,
            idempotent: false,
        });
        let signal = PcuKernel::Signal(PcuSignalKernelIr {
            id: PcuKernelId(3),
            entry_point: "irq",
            bindings: &[],
            ports: &[],
            parameters: &[],
            trigger: PcuSignalTriggerKind::Edge,
            ops: &[PcuSignalOp::Ack],
        });

        assert_eq!(command.kind(), PcuIrKind::Command);
        assert_eq!(transaction.kind(), PcuIrKind::Transaction);
        assert_eq!(signal.kind(), PcuIrKind::Signal);
    }
}
