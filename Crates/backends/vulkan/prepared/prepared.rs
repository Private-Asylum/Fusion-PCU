//! Typed prepared host calls for the exact bit-map Vulkan implementation.

use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_float_bit_map_to_spirv,
    PcuSpirvCapabilityCaps,
    validate_float_bit_map,
    PcuSpirvBitMapProfile,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
    PcuSpirvVersion,
};
#[rustfmt::skip]
use crate::{
    ffi::VulkanPreparedBitMap,
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
    PcuVulkanCallMeasurements,
};

/// Reusable exact F32/F64 Copy/checked Neg executable. Owns native storage and a retained session.
pub struct PcuVulkanPreparedBitMap {
    native: VulkanPreparedBitMap,
    profile: PcuSpirvBitMapProfile,
}

impl PcuVulkanPreparedBitMap {
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvBitMapProfile {
        self.profile
    }

    /// Measures one validated call, with native submission, completion and host publication stages.
    ///
    /// # Errors
    /// Returns the same argument, numerical and native errors as an ordinary prepared call.
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
        let (input, output) =
            validate_arguments(arguments, self.profile.extent, self.profile.scalar)?;
        let (source, destination) = if input < output {
            let (left, right) = arguments.split_at_mut(output);
            (&left[input], &mut right[0])
        } else {
            let (left, right) = arguments.split_at_mut(input);
            (&right[0], &mut left[output])
        };
        self.native.call(
            source.bytes(),
            destination
                .bytes_mut()
                .ok_or(PcuVulkanError::InvalidArguments)?,
            measurements,
        )
    }

    /// Returns actual input/output/status allocation flags, without another driver query.
    #[must_use]
    pub fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.native.memory_realizations()
    }
}

impl PcuHostKernelBackend for PcuVulkanBackend {
    type Prepared = PcuVulkanPreparedBitMap;
    type Error = PcuVulkanError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        if cfg!(target_endian = "big") {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        let admitted = validate_float_bit_map(kernel)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        self.device.validate_bit_map_geometry(admitted)?;
        let mut sink = PcuSpirvFixedSink::<512>::new();
        // Vulkan 1.1 accepts SPIR-V 1.3. Vulkan 1.0 keeps the legacy BufferBlock interface.
        let version = if self.caps().api_version >= (1 << 22) | (1 << 12) {
            PcuSpirvVersion::V1_3
        } else {
            PcuSpirvVersion::V1_0
        };
        let (_, profile) = lower_float_bit_map_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader()
                .with_version(version)
                .with_capabilities(if admitted.scalar == PcuScalarType::F64 {
                    PcuSpirvCapabilityCaps::SHADER.union(PcuSpirvCapabilityCaps::FLOAT64)
                } else {
                    PcuSpirvCapabilityCaps::SHADER
                }),
            &mut sink,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let native = VulkanPreparedBitMap::new(Rc::clone(&self.device), sink.as_slice(), profile)?;
        Ok(PcuVulkanPreparedBitMap { native, profile })
    }
}

impl PcuPreparedHostKernel for PcuVulkanPreparedBitMap {
    type Error = PcuVulkanError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}

fn validate_arguments(
    arguments: &[PcuHostArgument<'_>],
    extent: u32,
    scalar: PcuScalarType,
) -> Result<(usize, usize), PcuVulkanError> {
    if arguments.len() != 2 {
        return Err(PcuVulkanError::InvalidArguments);
    }
    let input = arguments
        .iter()
        .position(|argument| argument.target() == PcuSpirvBitMapProfile::INPUT)
        .ok_or(PcuVulkanError::InvalidArguments)?;
    let output = arguments
        .iter()
        .position(|argument| argument.target() == PcuSpirvBitMapProfile::OUTPUT)
        .ok_or(PcuVulkanError::InvalidArguments)?;
    let bytes = usize::try_from(extent)
        .ok()
        .and_then(|extent| extent.checked_mul(if scalar == PcuScalarType::F64 { 8 } else { 4 }))
        .ok_or(PcuVulkanError::InvalidArguments)?;
    if input == output
        || arguments
            .iter()
            .any(|argument| argument.scalar() != scalar || argument.bytes().len() < bytes)
        || arguments[input].access() != PcuBindingAccess::ReadOnly
        || arguments[output].access() != PcuBindingAccess::ReadWrite
    {
        return Err(PcuVulkanError::InvalidArguments);
    }
    Ok((input, output))
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        validate_arguments,
        PcuHostArgument,
        PcuSpirvBitMapProfile,
        PcuScalarType,
    };

    #[test]
    fn argument_schema_rejects_duplicates_short_types_and_access_before_device_calls() {
        let mut output = [0.0_f32; 3];
        let input = [1.0_f32; 3];
        let a = PcuSpirvBitMapProfile::INPUT;
        let b = PcuSpirvBitMapProfile::OUTPUT;
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &input),
                    PcuHostArgument::read_write(b, &mut output)
                ],
                4,
                PcuScalarType::F32
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &input),
                    PcuHostArgument::read(b, &input)
                ],
                3,
                PcuScalarType::F32
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &input),
                    PcuHostArgument::read_write(a, &mut output)
                ],
                3,
                PcuScalarType::F32
            )
            .is_err()
        );
        assert!(
            validate_arguments(
                &[
                    PcuHostArgument::read(a, &[1_u32; 3]),
                    PcuHostArgument::read_write(b, &mut output)
                ],
                3,
                PcuScalarType::F32
            )
            .is_err()
        );
        assert_eq!(
            validate_arguments(
                &[
                    PcuHostArgument::read_write(b, &mut output),
                    PcuHostArgument::read(a, &input)
                ],
                3,
                PcuScalarType::F32
            )
            .unwrap(),
            (1, 0)
        );
    }
}
