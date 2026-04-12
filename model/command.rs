//! Command-model vocabulary and backend-neutral kernel builder.

use crate::{
    PcuBinding,
    PcuCommandOpCaps,
    PcuDispatchPolicyCaps,
    PcuError,
    PcuKernel,
    PcuKernelIrContract,
    PcuKernelId,
    PcuKernelSignature,
    PcuInvocationModel,
    PcuIrKind,
    PcuParameter,
    PcuParameterSlot,
    PcuParameterValue,
    PcuPort,
    PcuBindingRef,
};

/// Coarse effect category for one command step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCommandEffectKind {
    Read,
    Write,
    Transfer,
    Transform,
    Control,
    Synchronize,
    Host,
    Vendor,
}

/// One abstract named target surfaced through model-local IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuTarget<'a> {
    Binding(PcuBindingRef),
    Port(&'a str),
    Named(&'a str),
    Intrinsic(&'a str),
}

/// One abstract operand consumed by model-local IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuOperand<'a> {
    Immediate(PcuParameterValue),
    Parameter(PcuParameterSlot),
    Target(PcuTarget<'a>),
    PreviousResult,
}

/// Mutation operation applied by one command step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCommandModifyOp {
    Assign,
    Add,
    Sub,
    And,
    Or,
    Xor,
    ShiftLeft,
    ShiftRight,
}

/// Predicate awaited by one command step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCommandPredicate<'a> {
    Ready(PcuTarget<'a>),
    Equals {
        left: PcuOperand<'a>,
        right: PcuOperand<'a>,
    },
    NonZero(PcuOperand<'a>),
    Named(&'a str),
}

/// One typed command operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuCommandOp<'a> {
    Read {
        target: PcuTarget<'a>,
    },
    Write {
        target: PcuTarget<'a>,
        value: PcuOperand<'a>,
    },
    Modify {
        target: PcuTarget<'a>,
        op: PcuCommandModifyOp,
        value: PcuOperand<'a>,
    },
    Copy {
        source: PcuTarget<'a>,
        target: PcuTarget<'a>,
    },
    Invoke {
        target: PcuTarget<'a>,
        args: &'a [PcuOperand<'a>],
    },
    Await {
        predicate: PcuCommandPredicate<'a>,
    },
    Stall {
        ticks: u32,
    },
    Sleep {
        ticks: u32,
    },
    Barrier,
    Return {
        value: Option<PcuOperand<'a>>,
    },
}

impl PcuCommandOp<'_> {
    #[must_use]
    pub const fn effect_kind(self) -> PcuCommandEffectKind {
        match self {
            Self::Read { .. } => PcuCommandEffectKind::Read,
            Self::Write { .. } | Self::Copy { .. } => PcuCommandEffectKind::Write,
            Self::Modify { .. } => PcuCommandEffectKind::Transform,
            Self::Invoke { .. } => PcuCommandEffectKind::Control,
            Self::Await { .. } | Self::Barrier => PcuCommandEffectKind::Synchronize,
            Self::Stall { .. } | Self::Sleep { .. } => PcuCommandEffectKind::Host,
            Self::Return { .. } => PcuCommandEffectKind::Control,
        }
    }

    #[must_use]
    pub const fn support_flag(self) -> PcuCommandOpCaps {
        match self {
            Self::Read { .. } => PcuCommandOpCaps::READ,
            Self::Write { .. } => PcuCommandOpCaps::WRITE,
            Self::Modify { .. } => PcuCommandOpCaps::MODIFY,
            Self::Copy { .. } => PcuCommandOpCaps::COPY,
            Self::Invoke { .. } => PcuCommandOpCaps::INVOKE,
            Self::Await { .. } => PcuCommandOpCaps::AWAIT,
            Self::Stall { .. } => PcuCommandOpCaps::STALL,
            Self::Sleep { .. } => PcuCommandOpCaps::SLEEP,
            Self::Barrier => PcuCommandOpCaps::BARRIER,
            Self::Return { .. } => PcuCommandOpCaps::RETURN,
        }
    }
}

/// One ordered command step in one command kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCommandStep<'a> {
    pub name: Option<&'a str>,
    pub op: PcuCommandOp<'a>,
}

/// Minimal semantic command-profile IR payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCommandKernelIr<'a> {
    pub id: PcuKernelId,
    pub entry_point: &'a str,
    pub bindings: &'a [PcuBinding<'a>],
    pub ports: &'a [PcuPort<'a>],
    pub parameters: &'a [PcuParameter<'a>],
    pub steps: &'a [PcuCommandStep<'a>],
}

impl PcuCommandKernelIr<'_> {
    /// Returns the dispatch-policy flags required to route this command kernel honestly.
    #[must_use]
    pub const fn required_dispatch_policy(&self) -> PcuDispatchPolicyCaps {
        PcuDispatchPolicyCaps::ORDERED_SUBMISSION
    }

    /// Returns the per-instruction support flags required to execute this command kernel.
    #[must_use]
    pub fn required_instruction_support(&self) -> PcuCommandOpCaps {
        let mut flags = PcuCommandOpCaps::empty();
        for step in self.steps.iter().copied() {
            flags = flags.union(step.op.support_flag());
        }
        flags
    }
}

impl PcuKernelIrContract for PcuCommandKernelIr<'_> {
    fn id(&self) -> PcuKernelId {
        self.id
    }

    fn kind(&self) -> PcuIrKind {
        PcuIrKind::Command
    }

    fn entry_point(&self) -> &str {
        self.entry_point
    }

    fn signature(&self) -> PcuKernelSignature<'_> {
        PcuKernelSignature {
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            invocation: PcuInvocationModel::command(),
        }
    }
}

const DEFAULT_STEP_CAPACITY: usize = 32;

/// Builder for one backend-neutral command kernel.
#[derive(Debug, Clone, Copy)]
pub struct PcuCommandKernelBuilder<'a, const MAX_STEPS: usize = DEFAULT_STEP_CAPACITY> {
    kernel_id: PcuKernelId,
    entry_point: &'a str,
    bindings: &'a [PcuBinding<'a>],
    ports: &'a [PcuPort<'a>],
    parameters: &'a [PcuParameter<'a>],
    steps: [PcuCommandStep<'a>; MAX_STEPS],
    step_len: usize,
}

impl<'a, const MAX_STEPS: usize> PcuCommandKernelBuilder<'a, MAX_STEPS> {
    /// Creates one command-kernel builder.
    #[must_use]
    pub fn new(kernel_id: u32, entry_point: &'a str) -> Self {
        Self {
            kernel_id: PcuKernelId(kernel_id),
            entry_point,
            bindings: &[],
            ports: &[],
            parameters: &[],
            steps: [PcuCommandStep {
                name: None,
                op: PcuCommandOp::Barrier,
            }; MAX_STEPS],
            step_len: 0,
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

    /// Appends one named command step.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder step capacity is exhausted.
    pub fn with_step(
        mut self,
        name: Option<&'a str>,
        op: PcuCommandOp<'a>,
    ) -> Result<Self, PcuError> {
        self.push_step(PcuCommandStep { name, op })?;
        Ok(self)
    }

    /// Appends several command steps in order.
    ///
    /// # Errors
    ///
    /// Returns `ResourceExhausted` when the builder step capacity is exhausted.
    pub fn with_steps(mut self, steps: &[PcuCommandStep<'a>]) -> Result<Self, PcuError> {
        for step in steps.iter().copied() {
            self.push_step(step)?;
        }
        Ok(self)
    }

    /// Returns the configured command steps.
    #[must_use]
    pub fn steps(&self) -> &[PcuCommandStep<'a>] {
        &self.steps[..self.step_len]
    }

    /// Builds the command-kernel IR payload.
    #[must_use]
    pub fn ir(&self) -> PcuCommandKernelIr<'_> {
        PcuCommandKernelIr {
            id: self.kernel_id,
            entry_point: self.entry_point,
            bindings: self.bindings,
            ports: self.ports,
            parameters: self.parameters,
            steps: &self.steps[..self.step_len],
        }
    }

    /// Builds the generic kernel wrapper.
    #[must_use]
    pub fn kernel(&self) -> PcuKernel<'_> {
        PcuKernel::Command(self.ir())
    }

    fn push_step(&mut self, step: PcuCommandStep<'a>) -> Result<(), PcuError> {
        if self.step_len == MAX_STEPS {
            return Err(PcuError::resource_exhausted());
        }
        self.steps[self.step_len] = step;
        self.step_len += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        PcuCommandKernelBuilder,
        PcuCommandOp,
        PcuTarget,
    };
    use crate::{
        PcuIrKind,
        PcuKernel,
        PcuKernelIrContract,
    };

    #[test]
    fn builder_synthesizes_command_kernel() {
        let builder = PcuCommandKernelBuilder::<4>::new(0x33, "init")
            .with_step(
                Some("read-status"),
                PcuCommandOp::Read {
                    target: PcuTarget::Named("status"),
                },
            )
            .expect("builder should accept one step");
        let kernel = builder.ir();

        assert_eq!(kernel.id.0, 0x33);
        assert_eq!(kernel.kind(), PcuIrKind::Command);
        assert_eq!(kernel.entry_point, "init");
        assert_eq!(kernel.steps.len(), 1);
    }

    #[test]
    fn builder_wraps_generic_command_kernel() {
        let builder = PcuCommandKernelBuilder::<1>::new(9, "apply");
        let kernel = builder.kernel();

        match kernel {
            PcuKernel::Command(command) => {
                assert_eq!(command.kind(), PcuIrKind::Command);
                assert_eq!(command.id.0, 9);
            }
            _ => panic!("expected command kernel"),
        }
    }
}
