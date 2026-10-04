//! Actual typed mixed resources; host publication and resident possible-write facts differ.
#[rustfmt::skip]
use std::{cell::{Ref,RefCell},rc::Rc};
#[rustfmt::skip]
use fusion_pcu::{PcuBindingAccess,PcuHostDispatchError,PcuMemoryAccess,PcuMemoryResource};
use crate::{MetalBuffer, MetalError, MetalHostKernelError, MetalMixedHostArgument};
use super::MetalPreparedConversionHostKernel;
enum Input {
    Uploaded(MetalBuffer),
    Resident(Rc<RefCell<MetalBuffer>>),
}
enum View<'a> {
    Uploaded(&'a MetalBuffer),
    Resident(Ref<'a, MetalBuffer>),
}
impl Input {
    fn view(&self) -> Result<View<'_>, MetalError> {
        match self {
            Self::Uploaded(buffer) => Ok(View::Uploaded(buffer)),
            Self::Resident(lease) => lease
                .try_borrow()
                .map(View::Resident)
                .map_err(|_| MetalError::Unsupported),
        }
    }
}
impl View<'_> {
    fn buffer(&self) -> &MetalBuffer {
        match self {
            Self::Uploaded(buffer) => buffer,
            Self::Resident(buffer) => buffer,
        }
    }
}
impl MetalPreparedConversionHostKernel {
    /// Validate exact mixed widths/access/session/real allocation before staging or submission.
    /// Fatal arithmetic completes privately and preserves both host and resident destinations.
    /// Successful or clamped payloads publish only after terminal status validation; coherent
    /// Shared resident prefixes copy directly without an intermediate RAM allocation or shader.
    /// # Errors
    /// Rejects arity/type/extent/access/alias/foreign resources or returns native/arithmetic failure.
    pub fn call_mixed(
        &mut self,
        args: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.may_have_written = false;
        self.session
            .ensure_quiescent()
            .map_err(PcuHostDispatchError::Backend)?;
        let input_bytes = self
            .plan
            .bytes(true)
            .map_err(PcuHostDispatchError::Backend)?;
        let output_bytes = self
            .plan
            .bytes(false)
            .map_err(PcuHostDispatchError::Backend)?;
        let positions = self.validate_mixed(args, [input_bytes, output_bytes])?;
        let input = match &args[positions[0]] {
            MetalMixedHostArgument::Host(argument) => Input::Uploaded(
                self.session
                    .upload_bytes(&argument.bytes()[..input_bytes])
                    .map_err(PcuHostDispatchError::Backend)?,
            ),
            MetalMixedHostArgument::Resident(argument) => {
                Input::Resident(argument.resource().lease())
            }
        };
        let view = input.view().map_err(PcuHostDispatchError::Backend)?;
        match &mut args[positions[1]] {
            MetalMixedHostArgument::Resident(argument) => {
                let lease = argument.resource().lease();
                let mut output = lease
                    .try_borrow_mut()
                    .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
                let (completed, notice) = self
                    .map
                    .execute_completed(view.buffer(), self.plan.count)
                    .map_err(PcuHostDispatchError::Backend)?;
                let publication = completed
                    .prepare_shared_prefix_copy(&mut output, output_bytes)
                    .map_err(PcuHostDispatchError::Backend)?;
                self.may_have_written = true;
                publication.publish();
                notice.map_or(Ok(()), |fault| {
                    Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
                })
            }
            MetalMixedHostArgument::Host(argument) => {
                let (completed, notice) = self
                    .map
                    .execute_completed(view.buffer(), self.plan.count)
                    .map_err(PcuHostDispatchError::Backend)?;
                let output = argument
                    .bytes_mut()
                    .ok_or(PcuHostDispatchError::AccessMismatch(self.plan.output))?;
                completed
                    .read_into_bytes(&mut output[..output_bytes])
                    .map_err(PcuHostDispatchError::Backend)?;
                self.may_have_written = true;
                notice.map_or(Ok(()), |fault| {
                    Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
                })
            }
        }
    }
    fn validate_mixed(
        &self,
        args: &[MetalMixedHostArgument<'_>],
        extents: [usize; 2],
    ) -> Result<[usize; 2], MetalHostKernelError> {
        let targets = [self.plan.input, self.plan.output];
        let mut positions = [0; 2];
        if args.len() != 2 {
            return Err(PcuHostDispatchError::Backend(MetalError::InvalidExtent));
        }
        for (slot, target) in targets.into_iter().enumerate() {
            let position = args
                .iter()
                .position(|arg| arg.descriptor().0 == target)
                .ok_or(PcuHostDispatchError::Missing(target))?;
            if args
                .iter()
                .filter(|arg| arg.descriptor().0 == target)
                .count()
                != 1
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let (_, scalar, access, bytes) = args[position].descriptor();
            let expected_scalar = if slot == 0 {
                self.plan.source_scalar()
            } else {
                self.plan.output_scalar()
            };
            let expected_access = if slot == 0 {
                PcuBindingAccess::ReadOnly
            } else {
                PcuBindingAccess::ReadWrite
            };
            let required = extents[slot];
            if scalar != expected_scalar {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if access != expected_access {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            if bytes < required {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
            if let MetalMixedHostArgument::Resident(argument) = &args[position] {
                let resource = argument.resource();
                let lease = resource.lease();
                let buffer = lease
                    .try_borrow()
                    .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
                if !self.session.same_session(buffer.session()) {
                    return Err(PcuHostDispatchError::Backend(MetalError::ForeignSession));
                }
                if bytes > buffer.byte_len()
                    || u64::try_from(bytes).map_or(true, |size| size > resource.size_bytes())
                {
                    return Err(PcuHostDispatchError::BufferTooSmall(target));
                }
                if (slot == 0 && resource.access() == PcuMemoryAccess::WriteOnly)
                    || (slot == 1 && resource.access() != PcuMemoryAccess::ReadWrite)
                {
                    return Err(PcuHostDispatchError::AccessMismatch(target));
                }
            }
            positions[slot] = position;
        }
        if let (MetalMixedHostArgument::Resident(input), MetalMixedHostArgument::Resident(output)) =
            (&args[positions[0]], &args[positions[1]])
            && Rc::ptr_eq(&input.resource().lease(), &output.resource().lease())
        {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        }
        Ok(positions)
    }
}
