//! Exact actual-read leases and private readback before two-destination publication.
#[rustfmt::skip]
use std::{
    cell::RefCell,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuHostDispatchError,
    PcuMemoryAccess,
    PcuMemoryResource,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalHostKernelError,
    MetalMixedHostArgument,
};
use super::MetalPreparedDivRemRoleHostKernel;

type Lease = Rc<RefCell<MetalBuffer>>;

impl MetalPreparedDivRemRoleHostKernel {
    /// Executes actual unique host/resident inputs and jointly publishes both checked results.
    ///
    /// Unread declarations do not acquire leases, inspect affinity or stage payloads. Resident
    /// prefixes may be larger than the admitted read/write spans; their tails remain untouched.
    /// Both private host reads complete before a potentially resident-writing command starts.
    ///
    /// # Errors
    /// Returns schema, extent, alias, affinity, arithmetic or terminal native failures. Fatal
    /// private arithmetic preserves existing outputs; uncertain publication quarantines the
    /// retained session and reports that existing destinations may have been written.
    pub fn call_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.may_have_written = false;
        let resident = self.preflight_mixed(arguments)?;
        if arguments.iter().all(|argument| {
            matches!(argument, MetalMixedHostArgument::Host(_))
                || self.plan.is_unread_declaration(argument.descriptor().0)
        }) {
            return self.call_host_mixed(arguments);
        }
        let mut inputs = [None, None];
        for (slot, &target) in self.input_bindings().iter().enumerate() {
            inputs[slot] = Some(if let Some(lease) = &resident[slot] {
                Rc::clone(lease)
            } else {
                let Some(MetalMixedHostArgument::Host(argument)) = arguments
                    .iter()
                    .find(|argument| argument.descriptor().0 == target)
                else {
                    return Err(PcuHostDispatchError::Missing(target));
                };
                Rc::new(RefCell::new(
                    self.control
                        .session()
                        .upload_bytes(&argument.bytes()[..self.plan.input_byte_lengths()[slot]])
                        .map_err(PcuHostDispatchError::Backend)?,
                ))
            });
        }
        let operands = self.plan.operand_inputs();
        let get = |slot: usize| -> Result<_, MetalHostKernelError> {
            inputs[slot]
                .as_ref()
                .ok_or(PcuHostDispatchError::Backend(MetalError::Unsupported))?
                .try_borrow()
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))
        };
        let lhs = get(operands[0])?;
        let rhs = get(operands[1])?;
        let private = self
            .control
            .execute([&lhs, &rhs])
            .map_err(PcuHostDispatchError::Backend)?;
        let outputs = self.output_bindings();
        let mut host = [None, None];
        for argument in arguments {
            if let MetalMixedHostArgument::Host(argument) = argument
                && let Some(slot) = outputs
                    .iter()
                    .position(|&target| target == argument.target())
            {
                let target = argument.target();
                host[slot] = Some(
                    argument
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(target))?,
                );
            }
        }
        if host.iter().any(Option::is_some) {
            let [q, r] = &mut self.readback;
            private
                .read_pair_bytes(q, r)
                .map_err(PcuHostDispatchError::Backend)?;
        }
        if resident[2].is_some() || resident[3].is_some() {
            let destination = |slot: usize| -> Result<Lease, MetalHostKernelError> {
                if let Some(lease) = &resident[slot] {
                    return Ok(Rc::clone(lease));
                }
                self.control
                    .session()
                    .allocate_zeroed_bytes(self.plan.byte_len())
                    .map(|buffer| Rc::new(RefCell::new(buffer)))
                    .map_err(PcuHostDispatchError::Backend)
            };
            let q = destination(2)?;
            let r = destination(3)?;
            let q = q
                .try_borrow()
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
            let r = r
                .try_borrow()
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
            self.may_have_written = true;
            self.publication
                .execute(&private, [&q, &r])
                .map_err(PcuHostDispatchError::Backend)?;
        }
        // Every fallible acquisition/read/submission is finished; host prefix copies cannot fail.
        for (slot, output) in host.into_iter().enumerate() {
            if let Some(output) = output {
                output[..self.plan.byte_len()].copy_from_slice(&self.readback[slot]);
            }
        }
        self.may_have_written = true;
        Ok(())
    }

    fn preflight_mixed(
        &self,
        arguments: &[MetalMixedHostArgument<'_>],
    ) -> Result<[Option<Lease>; 4], MetalHostKernelError> {
        self.control
            .session()
            .ensure_quiescent()
            .map_err(PcuHostDispatchError::Backend)?;
        let mut resident = [None, None, None, None];
        for (index, argument) in arguments.iter().enumerate() {
            let (target, scalar, access, bytes) = argument.descriptor();
            if arguments[..index]
                .iter()
                .any(|prior| prior.descriptor().0 == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            if scalar != self.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let input = self
                .input_bindings()
                .iter()
                .position(|&binding| binding == target);
            let output = self
                .output_bindings()
                .iter()
                .position(|&binding| binding == target);
            let expected = if output.is_some() {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            };
            if access != expected {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let (slot, required) = if let Some(slot) = input {
                (slot, self.plan.input_byte_lengths()[slot])
            } else if let Some(slot) = output {
                (slot + 2, self.plan.byte_len())
            } else if self.plan.is_unread_declaration(target) {
                continue;
            } else {
                return Err(PcuHostDispatchError::Unexpected(target));
            };
            if bytes < required {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
            if let MetalMixedHostArgument::Resident(argument) = argument {
                let resource = argument.resource();
                let lease = resource.lease();
                {
                    let buffer = lease
                        .try_borrow()
                        .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
                    if !self.control.session().same_session(buffer.session()) {
                        return Err(PcuHostDispatchError::Backend(MetalError::ForeignSession));
                    }
                    if bytes > buffer.byte_len() {
                        return Err(PcuHostDispatchError::BufferTooSmall(target));
                    }
                }
                if (input.is_some() && resource.access() == PcuMemoryAccess::WriteOnly)
                    || (output.is_some() && resource.access() != PcuMemoryAccess::ReadWrite)
                {
                    return Err(PcuHostDispatchError::AccessMismatch(target));
                }
                resident[slot] = Some(lease);
            }
        }
        for &target in self
            .input_bindings()
            .iter()
            .chain(self.output_bindings().iter())
        {
            if !arguments
                .iter()
                .any(|argument| argument.descriptor().0 == target)
            {
                return Err(PcuHostDispatchError::Missing(target));
            }
        }
        for output in 2..4 {
            if let Some(lease) = &resident[output]
                && resident.iter().enumerate().any(|(slot, other)| {
                    slot != output && other.as_ref().is_some_and(|other| Rc::ptr_eq(lease, other))
                })
            {
                return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
            }
        }
        Ok(resident)
    }
}
