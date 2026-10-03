//! Frozen exact F32/F64 binary host schema and transactional native execution.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
use fusion_pcu_spirv::PcuSpirvCheckedBinaryProfile;
#[rustfmt::skip]
use crate::{
    ffi::VulkanPreparedBinary,
    PcuVulkanCallMeasurements,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
};

/// Actual retained input/output/status allocations; repeated operands upload one input once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuVulkanBinaryMemoryRealizations {
    RepeatedInput([PcuVulkanMemoryRealization; 3]),
    DistinctInputs([PcuVulkanMemoryRealization; 4]),
}

/// Reusable U32-synthesized six-format checked Add/Sub/Mul/Div owner.
pub struct PcuVulkanPreparedBinary {
    pub(super) native: VulkanPreparedBinary,
    pub(super) profile: PcuSpirvCheckedBinaryProfile,
}

impl PcuVulkanPreparedBinary {
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvCheckedBinaryProfile {
        self.profile
    }

    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanBinaryMemoryRealizations> {
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

impl PcuPreparedHostKernel for PcuVulkanPreparedBinary {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}

fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    profile: PcuSpirvCheckedBinaryProfile,
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn complete_original_schema_is_checked_before_native_submission() {
        let profile = PcuSpirvCheckedBinaryProfile {
            scalar: fusion_pcu::PcuScalarType::F32,
            operation: fusion_pcu::PcuDispatchFloatBinaryOp::Add,
            range: fusion_pcu::PcuRangePolicy::Reject,
            underflow: fusion_pcu::PcuFloatUnderflowPolicy::default(),
            extent: 3,
            local_size: [64, 1, 1],
            operands: [0, 0],
            broadcast: [true, true],
            inputs: PcuSpirvCheckedBinaryProfile::INPUTS,
            input_count: 2,
            input_extents: [1, 3],
            output: PcuSpirvCheckedBinaryProfile::OUTPUT,
            declarations: [
                PcuSpirvCheckedBinaryProfile::INPUTS[0],
                PcuSpirvCheckedBinaryProfile::INPUTS[1],
                PcuSpirvCheckedBinaryProfile::OUTPUT,
            ],
            declaration_count: 3,
        };
        let a = PcuSpirvCheckedBinaryProfile::INPUTS[0];
        let b = PcuSpirvCheckedBinaryProfile::INPUTS[1];
        let c = PcuSpirvCheckedBinaryProfile::OUTPUT;
        let left = [1.0_f32];
        let right = [2.0_f32; 3];
        let mut output = [17.0_f32; 5];
        assert_eq!(
            validate_arguments(
                &[
                    PcuHostArgument::read_write(c, &mut output),
                    PcuHostArgument::read(b, &right),
                    PcuHostArgument::read(a, &left)
                ],
                profile
            )
            .unwrap(),
            [2, 1, 0]
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &left),
                    PcuHostArgument::read(b, &right[..2]),
                    PcuHostArgument::read_write(c, &mut output)
                ],
                profile
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &left),
                    PcuHostArgument::read(b, &right),
                    PcuHostArgument::read_write(c, &mut output[..2])
                ],
                profile
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &left),
                    PcuHostArgument::read(a, &right),
                    PcuHostArgument::read_write(c, &mut output)
                ],
                profile
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &left),
                    PcuHostArgument::read(b, &[2.0_f64; 3]),
                    PcuHostArgument::read_write(c, &mut output)
                ],
                profile
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &left),
                    PcuHostArgument::read(b, &right),
                    PcuHostArgument::read(c, &output)
                ],
                profile
            )
            .is_err()
        );
        assert_eq!(output.map(f32::to_bits), [17.0_f32; 5].map(f32::to_bits));
    }
}
