//! Actual unique read resources, independent scalar indices and joint private quotient/remainder.
#[path = "host_mixed/host_mixed.rs"]
mod host_mixed;
#[path = "mixed/mixed.rs"]
mod mixed;
#[path = "plan/plan.rs"]
mod plan;
pub use plan::MetalDivRemRolePlan;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalHostKernelError,
    MetalPreparedDivRemControl,
    MetalSession,
};

/// A separate exact-session factory; canonical preparation retains its previous profile.
pub struct MetalDivRemRoleHostBackend {
    session: MetalSession,
}
/// Detached actual-read metadata and a retained private packed-output integer control.
pub struct MetalPreparedDivRemRoleHostKernel {
    control: MetalPreparedDivRemControl,
    plan: MetalDivRemRolePlan,
    publication: crate::runtime::div_rem::publication::Publication,
    readback: [Vec<u8>; 2],
    may_have_written: bool,
}
impl MetalSession {
    #[must_use]
    pub fn checked_div_rem_role_backend(&self) -> MetalDivRemRoleHostBackend {
        MetalDivRemRoleHostBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MetalDivRemRoleHostBackend {
    type Prepared = MetalPreparedDivRemRoleHostKernel;
    type Error = MetalHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let plan = MetalDivRemRolePlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        let roles = plan.operand_inputs();
        let counts = plan.input_element_counts();
        let control = self
            .session
            .prepare_checked_div_rem_role_control(
                plan.scalar_type(),
                plan.element_count(),
                roles.map(|slot| counts[slot]),
                plan.operand_broadcast(),
            )
            .map_err(PcuHostDispatchError::Backend)?;
        Ok(MetalPreparedDivRemRoleHostKernel {
            publication: crate::runtime::div_rem::publication::Publication::prepare(
                &self.session,
                plan.byte_len(),
                plan.element_count(),
            )
            .map_err(PcuHostDispatchError::Backend)?,
            readback: [vec![0; plan.byte_len()], vec![0; plan.byte_len()]],
            control,
            plan,
            may_have_written: false,
        })
    }
}
impl MetalPreparedDivRemRoleHostKernel {
    #[must_use]
    pub fn input_bindings(&self) -> &[PcuBindingRef] {
        self.plan.input_bindings()
    }
    #[must_use]
    pub const fn output_bindings(&self) -> [PcuBindingRef; 2] {
        self.plan.output_bindings()
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        if self.plan.input_element_counts()[1] == 0 {
            3
        } else {
            4
        }
    }
    #[must_use]
    pub const fn scalar_type(&self) -> fusion_pcu::PcuScalarType {
        self.plan.scalar_type()
    }
    #[must_use]
    pub const fn element_count(&self) -> usize {
        self.plan.element_count()
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.control.session().ensure_quiescent().is_err()
    }
    pub(crate) fn is_unread_declaration(&self, target: PcuBindingRef) -> bool {
        self.plan.is_unread_declaration(target)
    }
    pub(crate) fn execute_joint(
        &self,
        inputs: &[&MetalBuffer],
        outputs: [&MetalBuffer; 2],
    ) -> Result<(), MetalError> {
        if inputs.len() != self.input_bindings().len() {
            return Err(MetalError::InvalidExtent);
        }
        for (slot, buffer) in inputs.iter().enumerate() {
            if !self.control.session().same_session(buffer.session()) {
                return Err(MetalError::ForeignSession);
            }
            if buffer.byte_len() < self.plan.input_byte_lengths()[slot] {
                return Err(MetalError::InvalidExtent);
            }
        }
        let operands = self.plan.operand_inputs();
        let private = self
            .control
            .execute([inputs[operands[0]], inputs[operands[1]]])?;
        self.publication.execute(&private, outputs)
    }
    fn validate(&self, arguments: &[PcuHostArgument<'_>]) -> Result<(), MetalHostKernelError> {
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            if argument.scalar() != self.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let input = self
                .input_bindings()
                .iter()
                .position(|&binding| binding == target);
            let output = self.output_bindings().contains(&target);
            let access = if output {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            };
            if argument.access() != access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let bytes = if let Some(slot) = input {
                self.plan.input_byte_lengths()[slot]
            } else if output {
                self.plan.byte_len()
            } else if self.plan.is_unread_declaration(target) {
                continue;
            } else {
                return Err(PcuHostDispatchError::Unexpected(target));
            };
            if argument.bytes().len() < bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        for binding in self
            .input_bindings()
            .iter()
            .chain(self.output_bindings().iter())
        {
            if !arguments
                .iter()
                .any(|argument| argument.target() == *binding)
            {
                return Err(PcuHostDispatchError::Missing(*binding));
            }
        }
        Ok(())
    }
    fn stage_inputs(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[Option<MetalBuffer>; 2], MetalHostKernelError> {
        let mut inputs = [None, None];
        for (slot, &target) in self.input_bindings().iter().enumerate() {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == target)
                .ok_or(PcuHostDispatchError::Missing(target))?;
            inputs[slot] = Some(
                self.control
                    .session()
                    .upload_bytes(&argument.bytes()[..self.plan.input_byte_lengths()[slot]])
                    .map_err(PcuHostDispatchError::Backend)?,
            );
        }
        Ok(inputs)
    }
}
impl PcuPreparedHostKernel for MetalPreparedDivRemRoleHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        self.validate(arguments)?;
        let inputs = self.stage_inputs(arguments)?;
        let operands = self.plan.operand_inputs();
        let get = |slot: usize| {
            inputs[slot]
                .as_ref()
                .ok_or(PcuHostDispatchError::Backend(MetalError::Unsupported))
        };
        let private = self
            .control
            .execute([get(operands[0])?, get(operands[1])?])
            .map_err(PcuHostDispatchError::Backend)?;
        let targets = self.output_bindings();
        let mut outputs = [None, None];
        for argument in arguments {
            if let Some(slot) = targets
                .iter()
                .position(|&target| target == argument.target())
            {
                let target = argument.target();
                outputs[slot] = Some(
                    argument
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(target))?,
                );
            }
        }
        let [Some(q), Some(r)] = outputs else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        private
            .read_pair_bytes(
                &mut q[..self.plan.byte_len()],
                &mut r[..self.plan.byte_len()],
            )
            .map_err(PcuHostDispatchError::Backend)?;
        self.may_have_written = true;
        Ok(())
    }
}
