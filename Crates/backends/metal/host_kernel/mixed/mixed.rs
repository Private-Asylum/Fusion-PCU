//! Checked mixed host/resident calls over the same admitted scalar map.

#[rustfmt::skip]
use std::{
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
    MetalError,
    MetalHostKernelError,
    MetalMemoryResource,
};
use super::MetalPreparedSingleHostKernel;

#[path = "input/input.rs"]
mod input;

/// One explicit host borrow or typed resident borrow for a synchronous source call.
pub enum MetalMixedHostArgument<'a> {
    Host(PcuHostArgument<'a>),
    Resident(PcuDeviceArgument<'a, MetalMemoryResource>),
}
impl MetalMixedHostArgument<'_> {
    pub(crate) const fn descriptor(
        &self,
    ) -> (PcuBindingRef, PcuScalarType, PcuBindingAccess, usize) {
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
                argument
                    .elements()
                    .saturating_mul(argument.scalar().bit_width() as usize / 8),
            ),
        }
    }
}
impl MetalPreparedSingleHostKernel {
    /// Reports whether the latest call reached potentially output-writing device execution.
    ///
    /// Resets before validation/staging and becomes true immediately before execution, including
    /// launch failures. Completion certainty and successful publication are separate facts.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }
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
        self.may_have_written = false;
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
                if self.kernel.binding_bytes(argument.descriptor().0) == 0 {
                    // An unread declaration has no resource or output-alias dependency.
                    continue;
                }
                if let MetalMixedHostArgument::Resident(input) = argument
                    && input.target() != output_ref
                    && Rc::ptr_eq(output, &input.resource().lease())
                {
                    return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
                }
            }
        }
        let inputs = self.kernel.input_bindings();
        let load = |target| -> Result<input::Input, MetalHostKernelError> {
            let position = self
                .schema
                .iter()
                .position(|&binding| binding == target)
                .ok_or(PcuHostDispatchError::Missing(target))?;
            match arguments
                .iter()
                .find(|argument| argument.descriptor().0 == target)
                .ok_or(PcuHostDispatchError::Missing(target))?
            {
                MetalMixedHostArgument::Resident(argument) => {
                    Ok(input::Input::Resident(argument.resource().lease()))
                }
                MetalMixedHostArgument::Host(argument) => self
                    .session
                    .upload_bytes(&argument.bytes()[..self.binding_bytes[position]])
                    .map(input::Input::Uploaded)
                    .map_err(PcuHostDispatchError::Backend),
            }
        };
        let left = load(inputs[0])?;
        let right = if inputs[0] == inputs[1] {
            None
        } else {
            Some(load(inputs[1])?)
        };
        let left_view = left.borrow();
        let right_view = right.as_ref().map(input::Input::borrow);
        let inputs = [
            left_view.buffer(),
            right_view
                .as_ref()
                .map_or_else(|| left_view.buffer(), input::View::buffer),
        ];
        if let Some(output) = resident_output {
            self.may_have_written = true;
            return self
                .kernel
                .execute_into(inputs, &output.borrow_mut())
                .map_err(PcuHostDispatchError::Backend);
        }
        self.may_have_written = true;
        let (output, recovered) = self
            .kernel
            .execute_completed(inputs)
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
            .map_err(PcuHostDispatchError::Backend)?;
        recovered.map_or(Ok(()), |fault| {
            Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
        })
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
            let Some(position) = self.schema.iter().position(|&binding| binding == target) else {
                return Err(PcuHostDispatchError::Unexpected(target));
            };
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
            if bytes < self.binding_bytes[position] {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
            if self.binding_bytes[position] == 0 {
                // Declared-unused readonly metadata is not a resource/affinity dependency.
                continue;
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
