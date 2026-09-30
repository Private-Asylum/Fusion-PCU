//! Checked mixed host/resident calls over the same admitted scalar map.

#[rustfmt::skip]
use std::{
    cell::RefCell,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDeviceArgument,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuMemoryAccess,
    PcuMemoryResource,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalHostKernelError,
    MetalMemoryResource,
};
use super::MetalPreparedHostKernel;

/// One explicit host borrow or typed resident borrow for a synchronous source call.
pub enum MetalMixedHostArgument<'a> {
    Host(PcuHostArgument<'a>),
    Resident(PcuDeviceArgument<'a, MetalMemoryResource>),
}
impl MetalMixedHostArgument<'_> {
    const fn descriptor(&self) -> (PcuBindingRef, PcuScalarType, PcuBindingAccess, usize) {
        match self {
            Self::Host(argument) => (
                argument.target(),
                argument.scalar(),
                argument.access(),
                argument.bytes().len(),
            ),
            Self::Resident(argument) => (
                argument.target(),
                argument.scalar(),
                argument.access(),
                argument.elements().saturating_mul(4),
            ),
        }
    }
}
impl MetalPreparedHostKernel {
    /// Reports whether native completion left the retained session quarantined.
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.session.ensure_quiescent().is_err()
    }
    /// Executes explicit host/resident staging with complete validation before uploads.
    ///
    /// # Errors
    /// Returns schema/extent/affinity errors, native failure or the first checked fault. Host
    /// output remains unchanged on arithmetic failure; resident output may have changed its
    /// prefix, and the caller must treat it according to the neutral resident contract.
    pub fn call_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.validate_mixed(arguments)?;
        let output_ref = self.kernel.output_binding();
        let output = arguments
            .iter()
            .find(|argument| argument.descriptor().0 == output_ref)
            .ok_or(PcuHostDispatchError::Missing(output_ref))?;
        let resident_output = if let MetalMixedHostArgument::Resident(argument) = output {
            Some(argument.resource().lease())
        } else {
            None
        };
        if let Some(output) = &resident_output {
            for argument in arguments.iter() {
                if let MetalMixedHostArgument::Resident(input) = argument
                    && input.target() != output_ref
                    && Rc::ptr_eq(output, &input.resource().lease())
                {
                    return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
                }
            }
        }
        let inputs = self.kernel.input_bindings();
        let load = |target| -> Result<Rc<RefCell<MetalBuffer>>, MetalHostKernelError> {
            match arguments
                .iter()
                .find(|argument| argument.descriptor().0 == target)
                .ok_or(PcuHostDispatchError::Missing(target))?
            {
                MetalMixedHostArgument::Resident(argument) => Ok(argument.resource().lease()),
                MetalMixedHostArgument::Host(argument) => self
                    .session
                    .upload_bytes(&argument.bytes()[..self.required_bytes])
                    .map(|buffer| Rc::new(RefCell::new(buffer)))
                    .map_err(PcuHostDispatchError::Backend),
            }
        };
        let left = load(inputs[0])?;
        let right = if inputs[0] == inputs[1] {
            Rc::clone(&left)
        } else {
            load(inputs[1])?
        };
        if let Some(output) = resident_output {
            return self
                .kernel
                .execute_into([&left.borrow(), &right.borrow()], &output.borrow_mut())
                .map_err(PcuHostDispatchError::Backend);
        }
        let output = self
            .kernel
            .execute_prefix([&left.borrow(), &right.borrow()])
            .map_err(PcuHostDispatchError::Backend)?;
        let Some(MetalMixedHostArgument::Host(destination)) = arguments
            .iter_mut()
            .find(|argument| argument.descriptor().0 == output_ref)
        else {
            return Err(PcuHostDispatchError::Missing(output_ref));
        };
        let bytes = destination
            .bytes_mut()
            .ok_or(PcuHostDispatchError::AccessMismatch(output_ref))?;
        // Host output remains transactional; a resident output follows its distinct contract.
        output
            .read_into_bytes(&mut bytes[..self.required_bytes])
            .map_err(PcuHostDispatchError::Backend)
    }
    fn validate_mixed(
        &self,
        arguments: &[MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.session
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
            if !self.schema.contains(&target) {
                return Err(PcuHostDispatchError::Unexpected(target));
            }
            if scalar != self.scalar {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            let expected = if target == self.kernel.output_binding() {
                PcuBindingAccess::ReadWrite
            } else {
                PcuBindingAccess::ReadOnly
            };
            if access != expected {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if bytes < self.required_bytes {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
            if let MetalMixedHostArgument::Resident(argument) = argument {
                let resource = argument.resource();
                let lease = resource.lease();
                if !self.session.same_session(lease.borrow().session()) {
                    return Err(PcuHostDispatchError::Backend(MetalError::ForeignSession));
                }
                if u64::try_from(bytes).map_or(true, |bytes| bytes > resource.size_bytes()) {
                    return Err(PcuHostDispatchError::BufferTooSmall(target));
                }
                if (expected == PcuBindingAccess::ReadOnly
                    && resource.access() == PcuMemoryAccess::WriteOnly)
                    || (expected == PcuBindingAccess::ReadWrite
                        && resource.access() != PcuMemoryAccess::ReadWrite)
                {
                    return Err(PcuHostDispatchError::AccessMismatch(target));
                }
            }
        }
        for target in &self.schema {
            if !arguments
                .iter()
                .any(|argument| argument.descriptor().0 == *target)
            {
                return Err(PcuHostDispatchError::Missing(*target));
            }
        }
        Ok(())
    }
}
