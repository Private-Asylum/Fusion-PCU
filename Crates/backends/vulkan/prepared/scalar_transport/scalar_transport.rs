//! Frozen twenty-two-carrier dense/broadcast schema; complete validation precedes every native submission.
#[rustfmt::skip]
use fusion_pcu::{PcuBindingAccess,PcuHostArgument,PcuPreparedHostKernel};
use fusion_pcu_spirv::PcuSpirvScalarTransportProfile;
#[rustfmt::skip]
use crate::{ffi::VulkanPreparedScalarTransport,PcuVulkanCallMeasurements,PcuVulkanError,PcuVulkanMemoryRealization};
/// Detached Shader/U32-only raw carrier transport; no native scalar feature is required.
pub struct PcuVulkanPreparedScalarTransport {
    pub(super) native: VulkanPreparedScalarTransport,
    pub(super) profile: PcuSpirvScalarTransportProfile,
}
impl PcuVulkanPreparedScalarTransport {
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvScalarTransportProfile {
        self.profile
    }
    #[must_use]
    pub fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 2]> {
        self.native.memory_realizations()
    }
    /// Measures a complete terminal transfer, publishing the initialized logical prefix.
    /// # Errors
    /// Returns preflight or native failure without publishing an incomplete output.
    pub fn call_profiled(
        &mut self,
        arguments: &mut [PcuHostArgument<'_>],
    ) -> Result<PcuVulkanCallMeasurements, PcuVulkanError> {
        let mut measurements = PcuVulkanCallMeasurements::default();
        self.call_measured(arguments, Some(&mut measurements))?;
        Ok(measurements)
    }
    fn call_measured(
        &mut self,
        arguments: &mut [PcuHostArgument<'_>],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        let positions = validate_arguments(arguments, self.profile)?;
        let (before, rest) = arguments.split_at_mut(positions[1]);
        let (output, after) = rest.split_first_mut().expect("validated output");
        let input = if positions[0] < positions[1] {
            before[positions[0]].bytes()
        } else {
            after[positions[0] - positions[1] - 1].bytes()
        };
        self.native.call(
            input,
            output.bytes_mut().expect("validated writable output"),
            measurements,
        )
    }
}
impl PcuPreparedHostKernel for PcuVulkanPreparedScalarTransport {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}
fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    profile: PcuSpirvScalarTransportProfile,
) -> Result<[usize; 2], PcuVulkanError> {
    if arguments.len() != 2 {
        return Err(PcuVulkanError::InvalidArguments);
    }
    let mut positions = [0; 2];
    for (slot, target) in [profile.input, profile.output].into_iter().enumerate() {
        let position = arguments
            .iter()
            .position(|argument| argument.target() == target)
            .ok_or(PcuVulkanError::InvalidArguments)?;
        let argument = &arguments[position];
        let count = if slot == 0 {
            profile.input_extent()
        } else {
            profile.extent
        };
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(profile.element_bytes()))
            .ok_or(PcuVulkanError::InvalidArguments)?;
        if argument.scalar() != profile.scalar
            || argument.bytes().len() < bytes
            || argument.access()
                != if slot == 0 {
                    PcuBindingAccess::ReadOnly
                } else {
                    PcuBindingAccess::ReadWrite
                }
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
        positions[slot] = position;
    }
    Ok(positions)
}
