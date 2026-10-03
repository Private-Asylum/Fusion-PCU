//! Original declaration access, actual spans and physical aliases checked before submission.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuHostArgument,
};
use fusion_pcu_spirv::PcuSpirvComposedFloatProfile;
use crate::PcuVulkanError;

pub(super) fn validate(
    arguments: &[PcuHostArgument<'_>],
    profile: &PcuSpirvComposedFloatProfile,
) -> Result<[usize; 4], PcuVulkanError> {
    if arguments.len() != profile.declarations().len() {
        return Err(PcuVulkanError::InvalidArguments);
    }
    let mut declared = [0; 4];
    for (slot, target) in profile.declarations().iter().enumerate() {
        let position = arguments
            .iter()
            .position(|argument| argument.target() == *target)
            .ok_or(PcuVulkanError::InvalidArguments)?;
        if declared[..slot].contains(&position) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let argument = &arguments[position];
        if argument.scalar() != profile.scalar()
            || Some(argument.access()) != profile.declaration_access(slot)
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
        declared[slot] = position;
    }
    let mut positions = [0; 4];
    let mut prefixes = [(0_usize, 0_usize, false); 4];
    for (slot, resource) in profile.resources().iter().enumerate() {
        let position = declared[resource.declaration];
        let argument = &arguments[position];
        let bytes = (resource.read_elements.max(resource.write_elements) as usize)
            .checked_mul(profile.element_bytes())
            .ok_or(PcuVulkanError::InvalidArguments)?;
        if argument.bytes().len() < bytes
            || (resource.write_elements != 0 && argument.access() != PcuBindingAccess::ReadWrite)
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let start = argument.bytes().as_ptr().addr();
        let end = start
            .checked_add(bytes)
            .ok_or(PcuVulkanError::InvalidArguments)?;
        let write = resource.write_elements != 0;
        for &(other_start, other_end, other_write) in &prefixes[..slot] {
            if (write || other_write) && start < other_end && other_start < end {
                return Err(PcuVulkanError::InvalidArguments);
            }
        }
        prefixes[slot] = (start, end, write);
        positions[slot] = position;
    }
    Ok(positions)
}
