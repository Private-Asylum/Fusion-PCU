//! Frozen exact fourteen-carrier integer host schema and transactional native execution.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
use fusion_pcu_spirv::PcuSpirvCheckedIntegerProfile;
#[rustfmt::skip]
use crate::{
    ffi::VulkanPreparedInteger,
    PcuVulkanCallMeasurements,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
};

/// Actual unique input/output/status allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuVulkanIntegerMemoryRealizations {
    RepeatedInput([PcuVulkanMemoryRealization; 3]),
    DistinctInputs([PcuVulkanMemoryRealization; 4]),
}
impl PcuVulkanIntegerMemoryRealizations {
    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::RepeatedInput(_) => 3,
            Self::DistinctInputs(_) => 4,
        }
    }
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// Reusable U32-synthesized checked integer Add/Sub/Mul owner.
pub struct PcuVulkanPreparedInteger {
    pub(super) native: VulkanPreparedInteger,
    pub(super) profile: PcuSpirvCheckedIntegerProfile,
}

impl PcuVulkanPreparedInteger {
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvCheckedIntegerProfile {
        self.profile
    }

    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanIntegerMemoryRealizations> {
        self.native.memory_realizations()
    }

    /// Measures a synchronous validated invocation including status scan and publication.
    ///
    /// # Errors
    /// Returns schema, arithmetic or native errors; fatal faults preserve caller output; recovered faults publish the complete useful prefix.
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
        let (before, rest) = arguments.split_at_mut(positions[2]);
        let (output, after) = rest.split_first_mut().expect("validated output position");
        let input = |position: usize| {
            if position < positions[2] {
                before[position].bytes()
            } else {
                after[position - positions[2] - 1].bytes()
            }
        };
        self.native.call(
            &[input(positions[0]), input(positions[1])],
            output.bytes_mut().expect("validated writable output"),
            measurements,
        )
    }
}

impl PcuPreparedHostKernel for PcuVulkanPreparedInteger {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}

fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    profile: PcuSpirvCheckedIntegerProfile,
) -> Result<[usize; 3], PcuVulkanError> {
    if arguments.len() != profile.declaration_count {
        return Err(PcuVulkanError::InvalidArguments);
    }
    for (index, argument) in arguments.iter().enumerate() {
        if arguments[..index]
            .iter()
            .any(|prior| prior.target() == argument.target())
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
    }
    for target in &profile.declarations[..profile.declaration_count] {
        let argument = arguments
            .iter()
            .find(|argument| argument.target() == *target)
            .ok_or(PcuVulkanError::InvalidArguments)?;
        let count = if *target == profile.output {
            profile.extent
        } else {
            profile.inputs[..profile.input_count]
                .iter()
                .position(|input| input == target)
                .map_or(0, |slot| profile.input_extent(slot))
        };
        let bytes = usize::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(profile.element_bytes()))
            .ok_or(PcuVulkanError::InvalidArguments)?;
        let access = if *target == profile.output {
            PcuBindingAccess::ReadWrite
        } else {
            PcuBindingAccess::ReadOnly
        };
        if argument.scalar() != profile.scalar
            || argument.access() != access
            || argument.bytes().len() < bytes
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
    }
    let mut positions = [0; 3];
    for (slot, target) in [profile.inputs[0], profile.inputs[1], profile.output]
        .into_iter()
        .enumerate()
    {
        positions[slot] = arguments
            .iter()
            .position(|argument| argument.target() == target)
            .ok_or(PcuVulkanError::InvalidArguments)?;
    }
    Ok(positions)
}
