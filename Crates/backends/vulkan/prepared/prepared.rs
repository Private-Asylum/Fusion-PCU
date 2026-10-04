//! Typed prepared host calls for the exact bit-map Vulkan implementation.

use std::rc::Rc;
#[path = "composed/composed.rs"]
mod composed;
pub use composed::PcuVulkanPreparedComposed;
#[path = "ordered_transport/ordered_transport.rs"]
mod ordered_transport;
pub use ordered_transport::PcuVulkanPreparedOrderedTransport;
#[path = "mixed/mixed.rs"]
mod mixed;
pub use mixed::PcuVulkanPreparedMixed;
#[path = "binary/binary.rs"]
mod binary;
#[rustfmt::skip]
pub use binary::{
    PcuVulkanPreparedBinary,
    PcuVulkanBinaryMemoryRealizations,
};
#[path = "div_rem/div_rem.rs"]
mod div_rem;
#[path = "integer/integer.rs"]
mod integer;
#[rustfmt::skip]
pub use div_rem::{
    PcuVulkanPreparedDivRem,
    PcuVulkanDivRemMemoryRealizations,
};
#[rustfmt::skip]
pub use integer::{
    PcuVulkanPreparedInteger,
    PcuVulkanIntegerMemoryRealizations,
};
#[path = "conversion/conversion.rs"]
mod conversion;
#[path = "unary/unary.rs"]
mod unary;
pub use conversion::PcuVulkanPreparedConversion;
#[rustfmt::skip]
pub use unary::{
    PcuVulkanPreparedUnary,
    PcuVulkanPreparedUnaryRoles,
};
#[path = "scalar_transport/scalar_transport.rs"]
mod scalar_transport;
pub use scalar_transport::PcuVulkanPreparedScalarTransport;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarType,
    PcuDispatchOp,
    PcuDispatchDataOp,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_scalar_transport_to_spirv,
    validate_scalar_transport_map,
    lower_float_bit_map_to_spirv,
    validate_checked_float_unary_map,
    validate_checked_float_conversion_map,
    lower_checked_float_conversion_to_spirv,
    lower_checked_float_unary_to_spirv,
    PcuSpirvCapabilityCaps,
    validate_float_bit_map,
    PcuSpirvBitMapProfile,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
    PcuSpirvVersion,
    validate_checked_float_binary_map,
    lower_checked_float_binary_to_spirv,
    validate_checked_integer_map,
    validate_checked_div_rem_map,
    lower_checked_div_rem_to_spirv,
    lower_checked_integer_to_spirv,
};
#[rustfmt::skip]
use crate::{
    ffi::VulkanPreparedBitMap,
    ffi::VulkanPreparedBinary,
    ffi::VulkanPreparedInteger,
    ffi::VulkanPreparedDivRem,
    ffi::VulkanPreparedUnary,
    ffi::VulkanPreparedConversion,
    ffi::VulkanPreparedScalarTransport,
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
    PcuVulkanCallMeasurements,
    PcuVulkanComposedMemoryRealizations,
};

/// Detached prepared host executable with its frozen typed declaration schema.
pub enum PcuVulkanPreparedHost {
    BitMap(PcuVulkanPreparedBitMap),
    Binary(PcuVulkanPreparedBinary),
    Integer(PcuVulkanPreparedInteger),
    DivRem(PcuVulkanPreparedDivRem),
    Unary(PcuVulkanPreparedUnary),
    UnaryRoles(PcuVulkanPreparedUnaryRoles),
    Conversion(PcuVulkanPreparedConversion),
    ScalarTransport(PcuVulkanPreparedScalarTransport),
    // Constructed once cold; keep other prepared variants at their existing stack size.
    Composed(Box<PcuVulkanPreparedComposed>),
    OrderedTransport(Box<PcuVulkanPreparedOrderedTransport>),
}

/// Actual allocation metadata for each private native buffer in schema order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuVulkanPreparedMemoryRealizations {
    BitMap([PcuVulkanMemoryRealization; 3]),
    Binary(PcuVulkanBinaryMemoryRealizations),
    Integer(PcuVulkanIntegerMemoryRealizations),
    DivRem(PcuVulkanDivRemMemoryRealizations),
    Unary([PcuVulkanMemoryRealization; 3]),
    Conversion([PcuVulkanMemoryRealization; 3]),
    ScalarTransport([PcuVulkanMemoryRealization; 2]),
    Composed(PcuVulkanComposedMemoryRealizations),
    OrderedTransport(PcuVulkanComposedMemoryRealizations),
}

impl PcuVulkanPreparedHost {
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        match self {
            Self::BitMap(_) | Self::Unary(_) | Self::Conversion(_) | Self::ScalarTransport(_) => 2,
            Self::UnaryRoles(plan) => plan.argument_count(),
            Self::Binary(plan) => plan.profile.declaration_count,
            Self::Integer(plan) => plan.profile.declaration_count,
            Self::DivRem(plan) => plan.profile.declaration_count,
            Self::Composed(plan) => plan.profile().declarations().len(),
            Self::OrderedTransport(plan) => plan.argument_count(),
        }
    }

    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanPreparedMemoryRealizations> {
        match self {
            Self::BitMap(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::BitMap),
            Self::Unary(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::Unary),
            Self::UnaryRoles(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::Unary),
            Self::Conversion(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::Conversion),
            Self::ScalarTransport(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::ScalarTransport),
            Self::Binary(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::Binary),
            Self::Integer(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::Integer),
            Self::DivRem(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::DivRem),
            Self::Composed(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::Composed),
            Self::OrderedTransport(map) => map
                .memory_realizations()
                .map(PcuVulkanPreparedMemoryRealizations::OrderedTransport),
        }
    }

    /// Measures one complete prepared native call.
    ///
    /// # Errors
    /// Returns argument, numerical or native errors from the frozen executable.
    pub fn call_profiled(
        &mut self,
        arguments: &mut [PcuHostArgument<'_>],
    ) -> Result<PcuVulkanCallMeasurements, PcuVulkanError> {
        match self {
            Self::BitMap(map) => map.call_profiled(arguments),
            Self::Binary(map) => map.call_profiled(arguments),
            Self::Integer(map) => map.call_profiled(arguments),
            Self::DivRem(map) => map.call_profiled(arguments),
            Self::Unary(map) => map.call_profiled(arguments),
            Self::UnaryRoles(map) => map.call_profiled(arguments),
            Self::Conversion(map) => map.call_profiled(arguments),
            Self::ScalarTransport(map) => map.call_profiled(arguments),
            Self::Composed(map) => map.call_profiled(arguments),
            Self::OrderedTransport(map) => map.call_profiled(arguments),
        }
    }
}

impl PcuPreparedHostKernel for PcuVulkanPreparedHost {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        match self {
            Self::BitMap(map) => map.call(arguments),
            Self::Binary(map) => map.call(arguments),
            Self::Integer(map) => map.call(arguments),
            Self::DivRem(map) => map.call(arguments),
            Self::Unary(map) => map.call(arguments),
            Self::UnaryRoles(map) => map.call(arguments),
            Self::Conversion(map) => map.call(arguments),
            Self::ScalarTransport(map) => map.call(arguments),
            Self::Composed(map) => map.call(arguments),
            Self::OrderedTransport(map) => map.call(arguments),
        }
    }
}

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
    type Prepared = PcuVulkanPreparedHost;
    type Error = PcuVulkanError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        if cfg!(target_endian = "big") {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        // Reject unsupported nested/cyclic borrowed regions before any profile's
        // capability scanner, independently of generated-artifact retention.
        crate::shader_artifact::validate_ops(kernel.ops)
            .map_err(|_| PcuVulkanError::UnsupportedPreparedProfile)?;
        let key = match self.device.artifacts.borrow().policy {
            crate::PcuVulkanShaderCachePolicy::Disabled => None,
            _ => match crate::PcuVulkanShaderArtifactKey::for_kernel(kernel, self.caps()) {
                Ok(key) => Some(key),
                Err(crate::PcuVulkanShaderArtifactError::TooLarge)
                    if matches!(
                        self.device.artifacts.borrow().policy,
                        crate::PcuVulkanShaderCachePolicy::MemoryOnly
                    ) =>
                {
                    None
                }
                Err(error) => return Err(PcuVulkanError::ShaderArtifact(error)),
            },
        };
        let _artifact_request =
            crate::shader_artifact::RequestGuard::new(&self.device.artifacts, key);
        let operations = scalar_body(kernel);
        if PcuVulkanPreparedComposed::admitted_profile(kernel).is_some() {
            return PcuVulkanPreparedComposed::prepare(self, kernel)
                .map(|plan| PcuVulkanPreparedHost::Composed(Box::new(plan)));
        }
        if let Some(prepared) = self.prepare_integer_operation(operations, kernel) {
            return prepared;
        }
        if let Some(prepared) = self.prepare_checked_float_operation(operations, kernel) {
            return prepared;
        }
        if let Ok(profile) = validate_scalar_transport_map(kernel) {
            self.device.validate_scalar_transport_geometry(profile)?;
            let mut words = Vec::new();
            let (_, profile) = lower_scalar_transport_to_spirv(
                kernel,
                PcuSpirvLoweringOptions::minimal_shader(),
                &mut words,
            )
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
            let native =
                VulkanPreparedScalarTransport::new(Rc::clone(&self.device), &words, profile)?;
            return Ok(PcuVulkanPreparedHost::ScalarTransport(
                PcuVulkanPreparedScalarTransport { native, profile },
            ));
        }
        if PcuVulkanPreparedOrderedTransport::admitted_profile(kernel).is_some() {
            return PcuVulkanPreparedOrderedTransport::prepare(self, kernel)
                .map(|plan| PcuVulkanPreparedHost::OrderedTransport(Box::new(plan)));
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
        Ok(PcuVulkanPreparedHost::BitMap(PcuVulkanPreparedBitMap {
            native,
            profile,
        }))
    }
}

const fn scalar_body<'a>(kernel: &PcuDispatchKernelIr<'a>) -> &'a [PcuDispatchOp<'a>] {
    if let [PcuDispatchOp::GridStrideLoop { body, .. }, _] = kernel.ops {
        body
    } else {
        kernel.ops
    }
}

impl PcuVulkanBackend {
    fn prepare_integer_operation(
        &self,
        operations: &[PcuDispatchOp<'_>],
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<Result<PcuVulkanPreparedHost, PcuVulkanError>> {
        operations.iter().find_map(|op| match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. }) => {
                Some(self.prepare_div_rem(kernel))
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedIntegerBinary { .. }) => {
                Some(self.prepare_integer(kernel))
            }
            _ => None,
        })
    }
    fn prepare_integer(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuVulkanPreparedHost, PcuVulkanError> {
        let profile = validate_checked_integer_map(kernel)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        self.device.validate_integer_geometry(profile)?;
        let mut words = Vec::new();
        let (_, profile) = lower_checked_integer_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let native = VulkanPreparedInteger::new(Rc::clone(&self.device), &words, profile)?;
        Ok(PcuVulkanPreparedHost::Integer(PcuVulkanPreparedInteger {
            native,
            profile,
        }))
    }

    fn prepare_div_rem(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuVulkanPreparedHost, PcuVulkanError> {
        let profile = validate_checked_div_rem_map(kernel)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        self.device.validate_div_rem_geometry(profile)?;
        let mut words = Vec::new();
        let (_, profile) = lower_checked_div_rem_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let native = VulkanPreparedDivRem::new(Rc::clone(&self.device), &words, profile)?;
        Ok(PcuVulkanPreparedHost::DivRem(PcuVulkanPreparedDivRem {
            native,
            profile,
        }))
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

impl PcuVulkanBackend {
    fn prepare_checked_float_operation(
        &self,
        operations: &[PcuDispatchOp<'_>],
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<Result<PcuVulkanPreparedHost, PcuVulkanError>> {
        operations.iter().find_map(|operation| match operation {
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert { .. }) => {
                Some(self.prepare_conversion(kernel))
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary { .. }) => {
                Some(self.prepare_binary(kernel))
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary { .. }) => {
                Some(self.prepare_unary(kernel))
            }
            _ => None,
        })
    }
    fn prepare_conversion(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuVulkanPreparedHost, PcuVulkanError> {
        let profile = validate_checked_float_conversion_map(kernel)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        self.device.validate_conversion_geometry(profile)?;
        let mut words = Vec::new();
        let (_, profile) = lower_checked_float_conversion_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let native = VulkanPreparedConversion::new(Rc::clone(&self.device), &words, profile)?;
        Ok(PcuVulkanPreparedHost::Conversion(
            PcuVulkanPreparedConversion { native, profile },
        ))
    }
    fn prepare_binary(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuVulkanPreparedHost, PcuVulkanError> {
        let profile = validate_checked_float_binary_map(kernel)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        self.device.validate_binary_geometry(profile)?;
        let mut words = Vec::new();
        let (_, profile) = lower_checked_float_binary_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let native = VulkanPreparedBinary::new(Rc::clone(&self.device), &words, profile)?;
        Ok(PcuVulkanPreparedHost::Binary(PcuVulkanPreparedBinary {
            native,
            profile,
        }))
    }
    fn prepare_unary(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<PcuVulkanPreparedHost, PcuVulkanError> {
        if fusion_pcu_spirv::validate_checked_float_unary_roles_map(kernel).is_ok() {
            return PcuVulkanPreparedUnaryRoles::prepare(Rc::clone(&self.device), kernel)
                .map(PcuVulkanPreparedHost::UnaryRoles);
        }
        let profile = validate_checked_float_unary_map(kernel)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        self.device.validate_unary_geometry(profile)?;
        let mut words = Vec::new();
        let (_, profile) = lower_checked_float_unary_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let native = VulkanPreparedUnary::new(Rc::clone(&self.device), &words, profile)?;
        Ok(PcuVulkanPreparedHost::Unary(PcuVulkanPreparedUnary {
            native,
            profile,
        }))
    }
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
