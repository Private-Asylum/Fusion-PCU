//! Cold full declaration retention, warm unique-role and mutable-prefix alias checks.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuValueType,
};
use fusion_pcu_spirv::PcuSpirvOrderedTransportProfile as Profile;
use crate::PcuVulkanError as Error;
struct Declaration {
    target: PcuBindingRef,
    access: PcuBindingAccess,
}
pub(super) struct Schema {
    declarations: Vec<Declaration>,
}
impl Schema {
    pub(super) const fn argument_count(&self) -> usize {
        self.declarations.len()
    }
    pub(super) fn new(kernel: &PcuDispatchKernelIr<'_>, profile: &Profile) -> Result<Self, Error> {
        let mut declarations = Vec::with_capacity(kernel.bindings.len());
        for binding in kernel.bindings {
            if binding.binding_type != PcuBindingType::Value(PcuValueType::Scalar(profile.scalar()))
                || !matches!(
                    binding.access,
                    PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
                )
                || declarations
                    .iter()
                    .any(|d: &Declaration| d.target == binding.reference())
            {
                return Err(Error::UnsupportedPreparedProfile);
            }
            declarations.push(Declaration {
                target: binding.reference(),
                access: binding.access,
            });
        }
        Ok(Self { declarations })
    }
    pub(super) fn validate(
        &self,
        arguments: &[PcuHostArgument<'_>],
        profile: &Profile,
    ) -> Result<[usize; 4], Error> {
        if arguments.len() != self.declarations.len() {
            return Err(Error::InvalidArguments);
        }
        // Declaration metadata has no native resource cost. Each declared role is required exactly
        // once, including typed empty unused arguments, without allocating a warm projection Vec.
        for declaration in &self.declarations {
            let mut matches = arguments
                .iter()
                .filter(|arg| arg.target() == declaration.target);
            let argument = matches.next().ok_or(Error::InvalidArguments)?;
            if matches.next().is_some()
                || argument.scalar() != profile.scalar()
                || argument.access() != declaration.access
            {
                return Err(Error::InvalidArguments);
            }
        }
        let mut positions = [0; 4];
        let mut prefixes = [(0_usize, 0_usize, false); 4];
        for (slot, resource) in profile.resources().iter().enumerate() {
            let position = arguments
                .iter()
                .position(|arg| arg.target() == resource.binding)
                .ok_or(Error::InvalidArguments)?;
            let argument = &arguments[position];
            let bytes = (resource.read_elements.max(resource.write_elements) as usize)
                .checked_mul(profile.element_bytes())
                .ok_or(Error::InvalidArguments)?;
            if argument.bytes().len() < bytes
                || (resource.write_elements != 0
                    && argument.access() != PcuBindingAccess::ReadWrite)
            {
                return Err(Error::InvalidArguments);
            }
            let start = argument.bytes().as_ptr().addr();
            let end = start.checked_add(bytes).ok_or(Error::InvalidArguments)?;
            let write = resource.write_elements != 0;
            for &(other_start, other_end, other_write) in &prefixes[..slot] {
                if (write || other_write) && start < other_end && other_start < end {
                    return Err(Error::InvalidArguments);
                }
            }
            prefixes[slot] = (start, end, write);
            positions[slot] = position;
        }
        Ok(positions)
    }
}
