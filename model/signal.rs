//! Signal-model vocabulary and backend-neutral kernel builder.

use crate::{
    PcuBinding,
    PcuDispatchPolicyCaps,
    PcuError,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelId,
    PcuKernelSignature,
    PcuInvocationModel,
    PcuIrKind,
    PcuParameter,
    PcuPort,
    PcuSignalOpCaps,
};

pub use crate::model::command::{
    PcuOperand,
    PcuTarget,
};

/// Trigger source family for one installed signal kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSignalTriggerKind {
    Edge,
    Level,
    Message,
    Timer,
    Software,
    Vendor,
}

/// One bounded signal-handler operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuSignalOp<'a> {
    Ack,
    Read {
        target: PcuTarget<'a>,
    },
    Write {
        target: PcuTarget<'a>,
        value: PcuOperand<'a>,
    },
    Publish {
        port: &'a str,
        value: PcuOperand<'a>,
    },
    Notify {
        target: PcuTarget<'a>,
    },
}

impl PcuSignalOp<'_> {
    #[must_use]
    pub const fn support_flag(self) -> PcuSignalOpCaps {
        match self {
            Self::Ack => PcuSignalOpCaps::ACK,
            Self::Read { .. } => PcuSignalOpCaps::READ,
            Self::Write { .. } => PcuSignalOpCaps::WRITE,
            Self::Publish { .. } => PcuSignalOpCaps::PUBLISH,
            Self::Notify { .. } => PcuSignalOpCaps::NOTIFY,
        }
    }
}

/// Minimal semantic signal-profile IR payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSignalKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub trigger: PcuSignalTriggerKind,
    pub ops: &'a [PcuSignalOp<'a>],
}

impl PcuSignalKernelIr<'_> {
    /// Returns the dispatch-policy flags required to route this signal kernel honestly.
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        PcuDispatchPolicyCaps::PERSISTENT_INSTALL
    }

    /// Returns the per-instruction support flags required to execute this signal kernel.
    #[must_use]
    pub fn required_instruction_support(&self) -> PcuSignalOpCaps {
        let mut flags = PcuSignalOpCaps::empty();
        for op in self.ops.iter().copied() {
            flags = flags.union(op.support_flag());
        }
        flags
    }
}

impl PcuKernelIrContract for PcuSignalKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Signal
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::triggered(),
        }
    }
}

const DEFAULT_OP_CAPACITY: usize = 16;

/// Builder for one backend-neutral signal kernel.
#[derive(Debug, Clone, Copy)]
pub struct PcuSignalKernelBuilder<'a, const MAX_OPS: usize = DEFAULT_OP_CAPACITY> {
    kernel_id: PcuKernelId,
    entry_point: &'a str,
    bindings: &'a [PcuBinding<'a>],
    ports: &'a [PcuPort<'a>],
    parameters: &'a [PcuParameter<'a>],
    trigger: PcuSignalTriggerKind,
    ops: [PcuSignalOp<'a>; MAX_OPS],
    op_len: usize,
}

impl<'a, const MAX_OPS: usize> PcuSignalKernelBuilder<'a, MAX_OPS> {
    /// Creates one signal-kernel builder.
    #[must_use]
    pub fn new(kernel_id: u32, entry_point: &'a str, trigger: PcuSignalTriggerKind) -> Self {
        Self {
            kernel_id: PcuKernelId(kernel_id),
            entry_point,
            bindings: &[],
            ports: &[],
            parameters: &[],
            trigger,
            ops: [PcuSignalOp::Ack; MAX_OPS],
            op_len: 0,
        }
    }

    /// Replaces the binding slice.
    #[must_use]
    pub const fn with_bindings(mut self, bindings: &'a [PcuBinding<'a>]) -> Self {
        self.bindings = bindings;
        self
    }

    /// Replaces the port slice.
    #[must_use]
    pub const fn with_ports(mut self, ports: &'a [PcuPort<'a>]) -> Self {
        self.ports = ports;
        self
    }

    /// Replaces the parameter slice.
    #[must_use]
    pub const fn with_parameters(mut self, parameters: &'a [PcuParameter<'a>]) -> Self {
        self.parameters = parameters;
        self
    }

    /// Appends one signal-handler op.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_op(mut self, op: PcuSignalOp<'a>) -> Result<Self, PcuError> {
        self.push_op(op)?;
        Ok(self)
    }

    /// Appends several signal-handler ops in order.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder op capacity is exhausted.
    pub fn with_ops(mut self, ops: &[PcuSignalOp<'a>]) -> Result<Self, PcuError> {
        for op in ops.iter().copied() {
            self.push_op(op)?;
        }
        Ok(self)
    }

    /// Returns the configured signal ops.
    #[must_use]
    pub fn ops(&self) -> &[PcuSignalOp<'a>] {
        &self.ops[..self.op_len]
    }

    /// Builds the signal-kernel IR payload.
    #[must_use]
    pub fn ir(&self) -> PcuSignalKernelIr<'_> {
        PcuSignalKernelIr {
            id: self.kernel_id,
            entry_point: self.entry_point,
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            trigger: self.trigger,
            ops: &self.ops[..self.op_len],
        }
    }

    /// Builds the generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Signal(self.ir())
    }

    fn push_op(&mut self, op: PcuSignalOp<'a>) -> Result<(), PcuError> {
        if self.op_len == MAX_OPS {
            return Err(PcuError::resource_exhausted());
        }
        self.ops[self.op_len] = op;
        self.op_len += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuSignalKernelBuilder,
        PcuSignalOp,
        PcuSignalTriggerKind,
        PcuTarget,
    };
    use crate::{
        PcuIrKind,
        PcuKernel,
        PcuKernelIrContract,
    };

    #[test]
    fn builder_synthesizes_signal_kernel() {
        let builder = PcuSignalKernelBuilder::<4>::new(0x55, "irq", PcuSignalTriggerKind::Edge)
            .with_op(PcuSignalOp::Notify {
                target: PcuTarget::Named("completion"),
            })
            .expect("builder should accept one signal op");
        let kernel = builder.ir();

        assert_eq!(kernel.id.0, 0x55);
        assert_eq!(kernel.kind(), PcuIrKind::Signal);
        assert_eq!(kernel.trigger, PcuSignalTriggerKind::Edge);
        assert_eq!(kernel.ops.len(), 1);
    }

    #[test]
    fn builder_wraps_generic_signal_kernel() {
        let builder = PcuSignalKernelBuilder::<1>::new(13, "watch", PcuSignalTriggerKind::Timer);
        let kernel = builder.kernel();

        match kernel {
            PcuKernel::Signal(signal) => {
                assert_eq!(signal.kind(), PcuIrKind::Signal);
                assert_eq!(signal.id.0, 13);
            }
            _ => panic!("expected signal kernel"),
        }
    }
}
