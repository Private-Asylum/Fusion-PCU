//! Two-output preflight, private arithmetic and one terminal resident publication command.
#[rustfmt::skip]
use std::{cell::RefCell,rc::Rc};
#[rustfmt::skip]
use fusion_pcu::{PcuBindingAccess,PcuHostDispatchError,PcuMemoryAccess,PcuMemoryResource};
#[rustfmt::skip]
use crate::{MetalBuffer,MetalMixedHostArgument,MetalError,MetalHostKernelError};
use super::MetalPreparedDivRemHostKernel;
impl MetalPreparedDivRemHostKernel {
    /// Whether the latest call may have changed an existing caller destination.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }
    /// Whether the actual retained native session is quarantined.
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.control.session().ensure_quiescent().is_err()
    }
    /// Computes both private outputs, checks every fault, then publishes them jointly.
    /// Host and resident tails beyond the frozen extent remain untouched.
    /// # Errors
    /// Returns binding, affinity, alias, arithmetic or native completion errors. Arithmetic
    /// failure never writes existing outputs; uncertain publication quarantines the session.
    pub fn call_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.may_have_written = false;
        self.validate_mixed(arguments)?;
        if arguments
            .iter()
            .all(|argument| matches!(argument, MetalMixedHostArgument::Host(_)))
        {
            return self.call_host_mixed(arguments);
        }
        let mut resident = [None, None, None, None];
        for argument in arguments.iter() {
            if let MetalMixedHostArgument::Resident(argument) = argument {
                let slot = self
                    .bindings
                    .iter()
                    .position(|&target| target == argument.target())
                    .ok_or_else(|| PcuHostDispatchError::Unexpected(argument.target()))?;
                resident[slot] = Some(argument.resource().lease());
            }
        }
        for output in 2..4 {
            if let Some(lease) = &resident[output] {
                for (index, other) in resident.iter().enumerate() {
                    if index != output
                        && other.as_ref().is_some_and(|other| Rc::ptr_eq(lease, other))
                    {
                        return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
                    }
                }
            }
        }
        let load = |slot: usize| -> Result<Rc<RefCell<MetalBuffer>>, MetalHostKernelError> {
            if let Some(lease) = &resident[slot] {
                return Ok(Rc::clone(lease));
            }
            let argument = arguments
                .iter()
                .find(|argument| argument.descriptor().0 == self.bindings[slot])
                .ok_or(PcuHostDispatchError::Missing(self.bindings[slot]))?;
            let MetalMixedHostArgument::Host(argument) = argument else {
                return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
            };
            self.control
                .session()
                .upload_bytes(&argument.bytes()[..self.bytes])
                .map(|buffer| Rc::new(RefCell::new(buffer)))
                .map_err(PcuHostDispatchError::Backend)
        };
        let first = load(0)?;
        let second = load(1)?;
        let inputs = [&first, &second];
        let private = self
            .control
            .execute([
                &inputs[self.operands[0]].borrow(),
                &inputs[self.operands[1]].borrow(),
            ])
            .map_err(PcuHostDispatchError::Backend)?;
        if resident[2].is_none() && resident[3].is_none() {
            self.publish_host_pair(arguments, &private)?;
            self.may_have_written = true;
            return Ok(());
        }
        let destination = |slot: usize| -> Result<Rc<RefCell<MetalBuffer>>, MetalHostKernelError> {
            if let Some(lease) = &resident[slot] {
                Ok(Rc::clone(lease))
            } else {
                self.control
                    .session()
                    .allocate_zeroed_bytes(self.bytes)
                    .map(|buffer| Rc::new(RefCell::new(buffer)))
                    .map_err(PcuHostDispatchError::Backend)
            }
        };
        let q = destination(2)?;
        let r = destination(3)?;
        self.may_have_written = true;
        self.publication
            .execute(&private, [&q.borrow(), &r.borrow()])
            .map_err(PcuHostDispatchError::Backend)?;
        for argument in arguments {
            if let MetalMixedHostArgument::Host(argument) = argument {
                for (slot, &target) in self.bindings[2..].iter().enumerate() {
                    if argument.target() == target {
                        let bytes = argument
                            .bytes_mut()
                            .ok_or(PcuHostDispatchError::AccessMismatch(target))?;
                        [&q, &r][slot]
                            .borrow()
                            .read_into_bytes(&mut bytes[..self.bytes])
                            .map_err(PcuHostDispatchError::Backend)?;
                        break;
                    }
                }
            }
        }
        Ok(())
    }
    fn call_host_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        let mut inputs = [None, None];
        let mut outputs = [None, None];
        for argument in arguments {
            let MetalMixedHostArgument::Host(argument) = argument else {
                return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
            };
            let target = argument.target();
            let slot = self
                .bindings
                .iter()
                .position(|&binding| binding == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if slot < 2 {
                inputs[slot] = Some(argument.bytes());
            } else {
                outputs[slot - 2] = Some(
                    argument
                        .bytes_mut()
                        .ok_or(PcuHostDispatchError::AccessMismatch(target))?,
                );
            }
        }
        let [Some(q), Some(r)] = outputs else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        self.control
            .call_bytes(
                inputs[self.operands[0]].ok_or(PcuHostDispatchError::Missing(
                    self.bindings[self.operands[0]],
                ))?,
                inputs[self.operands[1]].ok_or(PcuHostDispatchError::Missing(
                    self.bindings[self.operands[1]],
                ))?,
                q,
                r,
            )
            .map_err(PcuHostDispatchError::Backend)?;
        self.may_have_written = true;
        Ok(())
    }
    fn publish_host_pair(
        &self,
        arguments: &mut [MetalMixedHostArgument<'_>],
        private: &MetalBuffer,
    ) -> Result<(), MetalHostKernelError> {
        let mut outputs = [None, None];
        for argument in arguments {
            if let MetalMixedHostArgument::Host(argument) = argument {
                for (slot, &target) in self.bindings[2..].iter().enumerate() {
                    if argument.target() == target {
                        outputs[slot] = Some(
                            argument
                                .bytes_mut()
                                .ok_or(PcuHostDispatchError::AccessMismatch(target))?,
                        );
                        break;
                    }
                }
            }
        }
        let [Some(q), Some(r)] = outputs else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        private
            .read_pair_bytes(&mut q[..self.bytes], &mut r[..self.bytes])
            .map_err(PcuHostDispatchError::Backend)?;
        Ok(())
    }
    fn validate_mixed(
        &self,
        arguments: &[MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.control
            .session()
            .ensure_quiescent()
            .map_err(PcuHostDispatchError::Backend)?;
        for (index, argument) in arguments.iter().enumerate() {
            let (target, scalar, access, bytes) = argument.descriptor();
            if arguments[..index]
                .iter()
                .any(|prior| prior.descriptor().0 == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let slot = self
                .bindings
                .iter()
                .position(|&binding| binding == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if scalar != self.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let expected = if slot < 2 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            };
            if access != expected {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if bytes < self.bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
            if let MetalMixedHostArgument::Resident(argument) = argument {
                let resource = argument.resource();
                let lease = resource.lease();
                if !self
                    .control
                    .session()
                    .same_session(lease.borrow().session())
                {
                    return Err(PcuHostDispatchError::Backend(MetalError::ForeignSession));
                }
                if u64::try_from(bytes).map_or(true, |bytes| bytes > resource.size_bytes()) {
                    return Err(PcuHostDispatchError::BufferTooSmall(target));
                }
                if (slot < 2 && resource.access() == PcuMemoryAccess::WriteOnly)
                    || (slot >= 2 && resource.access() != PcuMemoryAccess::ReadWrite)
                {
                    return Err(PcuHostDispatchError::AccessMismatch(target));
                }
            }
        }
        for &target in &self.bindings {
            if !arguments
                .iter()
                .any(|argument| argument.descriptor().0 == target)
            {
                return Err(PcuHostDispatchError::Missing(target));
            }
        }
        Ok(())
    }
}
