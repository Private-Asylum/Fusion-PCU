//! Frozen F32/F64 conversion schema; complete validation precedes every native submission.
#[rustfmt::skip]
use fusion_pcu::{PcuBindingAccess,PcuHostArgument,PcuPreparedHostKernel};
use fusion_pcu_spirv::PcuSpirvCheckedConversionProfile;
#[rustfmt::skip]
use crate::{ffi::VulkanPreparedConversion,PcuVulkanCallMeasurements,PcuVulkanError,PcuVulkanMemoryRealization};
/// Detached Shader/U32-only checked mixed-width conversion owner; no native floating feature is required.
pub struct PcuVulkanPreparedConversion {
    pub(super) native: VulkanPreparedConversion,
    pub(super) profile: PcuSpirvCheckedConversionProfile,
}
impl PcuVulkanPreparedConversion {
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvCheckedConversionProfile {
        self.profile
    }
    #[must_use]
    pub fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.native.memory_realizations()
    }
    /// Measures a complete call with fatal rollback and recovered full publication.
    /// # Errors
    /// Returns preflight, arithmetic or native failure; recovered range faults publish then Err.
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
impl PcuPreparedHostKernel for PcuVulkanPreparedConversion {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}
fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    profile: PcuSpirvCheckedConversionProfile,
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
        let scalar = if slot == 0 {
            profile.source_scalar()
        } else {
            profile.output_scalar()
        };
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(usize::from(scalar.bit_width() / 8)))
            .ok_or(PcuVulkanError::InvalidArguments)?;
        if argument.scalar() != scalar
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
