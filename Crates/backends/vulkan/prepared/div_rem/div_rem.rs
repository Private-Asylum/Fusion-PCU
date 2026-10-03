//! Frozen original declarations and distinct read resources, transactional dual publication.
#[rustfmt::skip]
use fusion_pcu::{PcuBindingAccess,PcuHostArgument,PcuPreparedHostKernel};
use fusion_pcu_spirv::PcuSpirvCheckedDivRemProfile;
#[rustfmt::skip]
use crate::{ffi::VulkanPreparedDivRem,PcuVulkanError,PcuVulkanCallMeasurements,PcuVulkanMemoryRealization};
/// Actual unique read, quotient, remainder and status allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuVulkanDivRemMemoryRealizations {
    RepeatedInput([PcuVulkanMemoryRealization; 4]),
    DistinctInputs([PcuVulkanMemoryRealization; 5]),
}
impl PcuVulkanDivRemMemoryRealizations {
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::RepeatedInput(_) => 4,
            Self::DistinctInputs(_) => 5,
        }
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}
/// Retained U32-only checked quotient/remainder host executor.
pub struct PcuVulkanPreparedDivRem {
    pub(super) native: VulkanPreparedDivRem,
    pub(super) profile: PcuSpirvCheckedDivRemProfile,
}
impl PcuVulkanPreparedDivRem {
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvCheckedDivRemProfile {
        self.profile
    }
    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanDivRemMemoryRealizations> {
        self.native.memory_realizations()
    }
    /// Measures one synchronous validated two-output call.
    /// # Errors
    /// Returns schema/arithmetic/native errors; any fatal preserves both complete caller outputs.
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
        let lo = positions[2].min(positions[3]);
        let hi = positions[2].max(positions[3]);
        let (before, later) = arguments.split_at_mut(lo);
        let (lower, remaining) = later.split_first_mut().expect("validated lower output");
        let (middle, later) = remaining.split_at_mut(hi - lo - 1);
        let (upper, after) = later.split_first_mut().expect("validated upper output");
        let input = |position: usize| {
            if position < lo {
                before[position].bytes()
            } else if position < hi {
                middle[position - lo - 1].bytes()
            } else {
                after[position - hi - 1].bytes()
            }
        };
        let inputs = [input(positions[0]), input(positions[1])];
        let lower = lower.bytes_mut().expect("validated output access");
        let upper = upper.bytes_mut().expect("validated output access");
        let mut outputs = if positions[2] == lo {
            [lower, upper]
        } else {
            [upper, lower]
        };
        self.native.call(&inputs, &mut outputs, measurements)
    }
}
impl PcuPreparedHostKernel for PcuVulkanPreparedDivRem {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}
fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    profile: PcuSpirvCheckedDivRemProfile,
) -> Result<[usize; 4], PcuVulkanError> {
    if arguments.len() != profile.declaration_count {
        return Err(PcuVulkanError::InvalidArguments);
    }
    let mut positions = [0; 4];
    for &target in &profile.declarations[..profile.declaration_count] {
        let position = arguments
            .iter()
            .position(|argument| argument.target() == target)
            .ok_or(PcuVulkanError::InvalidArguments)?;
        let argument = &arguments[position];
        let (count, writable) =
            if let Some(slot) = profile.outputs.iter().position(|output| *output == target) {
                positions[slot + 2] = position;
                (profile.extent, true)
            } else if let Some(slot) = profile.inputs[..profile.input_count]
                .iter()
                .position(|input| *input == target)
            {
                positions[slot] = position;
                (profile.input_extents[slot], false)
            } else {
                (0, false)
            };
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(profile.element_bytes()))
            .ok_or(PcuVulkanError::InvalidArguments)?;
        if argument.scalar() != profile.scalar
            || argument.bytes().len() < bytes
            || argument.access()
                != if writable {
                    PcuBindingAccess::ReadWrite
                } else {
                    PcuBindingAccess::ReadOnly
                }
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
    }
    if profile.input_count == 1 {
        positions[1] = positions[0];
    }
    Ok(positions)
}
