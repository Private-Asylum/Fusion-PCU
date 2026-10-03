//! Actual resource leases, private host banks and sibling publication ordering.

#[rustfmt::skip]
use std::{
    cell::{
        Ref,
        RefCell,
        RefMut,
    },
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
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
use super::MetalPreparedCheckedMapHostKernel;

enum Bank {
    Uploaded(MetalBuffer),
    Resident(Rc<RefCell<MetalBuffer>>),
}
enum View<'a> {
    Uploaded(&'a MetalBuffer),
    Read(Ref<'a, MetalBuffer>),
    Write(RefMut<'a, MetalBuffer>),
}
impl Bank {
    fn view(&self, writable: bool) -> Result<View<'_>, MetalHostKernelError> {
        match self {
            Self::Uploaded(buffer) => Ok(View::Uploaded(buffer)),
            Self::Resident(lease) if writable => lease
                .try_borrow_mut()
                .map(View::Write)
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported)),
            Self::Resident(lease) => lease
                .try_borrow()
                .map(View::Read)
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported)),
        }
    }
}
impl View<'_> {
    fn buffer(&self) -> &MetalBuffer {
        match self {
            Self::Uploaded(buffer) => buffer,
            Self::Read(buffer) => buffer,
            Self::Write(buffer) => buffer,
        }
    }
}

impl MetalPreparedCheckedMapHostKernel {
    /// Executes exactly the declared typed schema with only actual resource staging.
    ///
    /// Both destinations and every actual input are checked before any upload or
    /// submission. Unused declarations retain access/type checks but have no native
    /// resource, lease or affinity dependency. Host siblings publish only after all
    /// fallible private reads; resident writes obey possible-write discard/quarantine.
    /// # Errors
    /// Returns schema, extent, alias, affinity, quarantine, native or exact arithmetic errors.
    pub fn call_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.may_have_written = false;
        self.validate_mixed(arguments)?;
        self.execute_mixed(arguments)
    }

    fn execute_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        let banks = self.stage_mixed(arguments)?;
        let count = self.plan().resources().len();
        let mut views: [Option<View<'_>>; 4] = core::array::from_fn(|_| None);
        for (slot, role) in self.plan().resources().iter().enumerate() {
            views[slot] = Some(
                banks[slot]
                    .as_ref()
                    .unwrap()
                    .view(role.minimum_write_elements != 0)?,
            );
        }
        let resources: [&MetalBuffer; 4] =
            core::array::from_fn(|slot| views[slot.min(count - 1)].as_ref().unwrap().buffer());
        let resident_writer = self
            .plan()
            .resources()
            .iter()
            .enumerate()
            .any(|(slot, role)| {
                role.minimum_write_elements != 0 && matches!(banks[slot], Some(Bank::Resident(_)))
            });
        let execution = self.kernel.execute_into(&resources[..count]);
        self.may_have_written = resident_writer && self.kernel.last_call_may_have_written();
        let notice = match execution {
            Ok(()) => None,
            Err(MetalError::Arithmetic(fault)) if fault.recovered => Some(fault),
            Err(error) => return Err(PcuHostDispatchError::Backend(error)),
        };
        for (slot, readback) in self.readback.iter_mut().enumerate() {
            if !readback.is_empty() && matches!(banks[slot], Some(Bank::Uploaded(_))) {
                resources[slot]
                    .read_bytes(0, readback)
                    .map_err(PcuHostDispatchError::Backend)?;
            }
        }
        // No fallible work remains between sibling public host commits.
        drop(views);
        self.publish_host(arguments);
        notice.map_or(Ok(()), |fault| {
            Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
        })
    }

    fn validate_mixed(
        &self,
        arguments: &[MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        self.kernel
            .session
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
            let declaration = self
                .plan()
                .declared_bindings()
                .iter()
                .find(|binding| binding.0 == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if scalar != self.plan().value_type().scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if access != declaration.1 {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let required = self
                .plan()
                .resources()
                .iter()
                .find(|resource| resource.binding == target)
                .map_or(0, |role| {
                    usize::try_from(role.minimum_elements()).unwrap_or(usize::MAX)
                })
                .saturating_mul(super::scalar_width(self.plan()));
            if bytes < required {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
            if required != 0 {
                self.validate_resident(argument, bytes)?;
            }
        }
        for declaration in self.plan().declared_bindings() {
            argument(arguments, declaration.0)?;
        }
        self.validate_aliases(arguments)
    }

    fn validate_resident(
        &self,
        argument: &MetalMixedHostArgument<'_>,
        bytes: usize,
    ) -> Result<(), MetalHostKernelError> {
        let MetalMixedHostArgument::Resident(argument) = argument else {
            return Ok(());
        };
        let resource = argument.resource();
        let lease = resource.lease();
        let buffer = lease
            .try_borrow()
            .map_err(|_| PcuHostDispatchError::Backend(MetalError::Unsupported))?;
        if !self.kernel.session.same_session(buffer.session()) {
            return Err(PcuHostDispatchError::Backend(MetalError::ForeignSession));
        }
        if bytes > buffer.byte_len() {
            return Err(PcuHostDispatchError::BufferTooSmall(argument.target()));
        }
        let valid_access = match argument.access() {
            PcuBindingAccess::ReadOnly => resource.access() != PcuMemoryAccess::WriteOnly,
            PcuBindingAccess::ReadWrite => resource.access() == PcuMemoryAccess::ReadWrite,
            PcuBindingAccess::WriteOnly => resource.access() != PcuMemoryAccess::ReadOnly,
        };
        if !valid_access {
            return Err(PcuHostDispatchError::AccessMismatch(argument.target()));
        }
        Ok(())
    }

    fn validate_aliases(
        &self,
        arguments: &[MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        for (slot, role) in self.plan().resources().iter().enumerate() {
            let MetalMixedHostArgument::Resident(current) = argument(arguments, role.binding)?
            else {
                continue;
            };
            for prior in &self.plan().resources()[..slot] {
                if (role.minimum_write_elements != 0 || prior.minimum_write_elements != 0)
                    && let MetalMixedHostArgument::Resident(other) =
                        argument(arguments, prior.binding)?
                    && Rc::ptr_eq(&current.resource().lease(), &other.resource().lease())
                {
                    return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
                }
            }
        }
        Ok(())
    }

    fn stage_mixed(
        &self,
        arguments: &[MetalMixedHostArgument<'_>],
    ) -> Result<[Option<Bank>; 4], MetalHostKernelError> {
        let mut banks = core::array::from_fn(|_| None);
        for (slot, role) in self.plan().resources().iter().enumerate() {
            banks[slot] = Some(match argument(arguments, role.binding)? {
                MetalMixedHostArgument::Resident(argument) => {
                    Bank::Resident(argument.resource().lease())
                }
                MetalMixedHostArgument::Host(argument) => {
                    let extent = usize::try_from(role.minimum_elements()).unwrap()
                        * super::scalar_width(self.plan());
                    let initial = usize::try_from(role.minimum_initial_read_elements).unwrap()
                        * super::scalar_width(self.plan());
                    let mut buffer = self
                        .kernel
                        .session
                        .allocate_zeroed_bytes(extent)
                        .map_err(PcuHostDispatchError::Backend)?;
                    if initial != 0 {
                        buffer
                            .write_bytes(0, &argument.bytes()[..initial])
                            .map_err(PcuHostDispatchError::Backend)?;
                    }
                    Bank::Uploaded(buffer)
                }
            });
        }
        Ok(banks)
    }

    fn publish_host(&mut self, arguments: &mut [MetalMixedHostArgument<'_>]) {
        for (slot, role) in self.kernel.plan().resources().iter().enumerate() {
            if role.minimum_write_elements != 0
                && let MetalMixedHostArgument::Host(output) = arguments
                    .iter_mut()
                    .find(|argument| argument.descriptor().0 == role.binding)
                    .unwrap()
            {
                self.may_have_written = true;
                output.bytes_mut().unwrap()[..self.readback[slot].len()]
                    .copy_from_slice(&self.readback[slot]);
            }
        }
    }
}

fn argument<'argument, 'data>(
    arguments: &'argument [MetalMixedHostArgument<'data>],
    target: PcuBindingRef,
) -> Result<&'argument MetalMixedHostArgument<'data>, MetalHostKernelError> {
    arguments
        .iter()
        .find(|argument| argument.descriptor().0 == target)
        .ok_or(PcuHostDispatchError::Missing(target))
}
