//! Frozen static native program with private transactional resource shadows.
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_composed_float_to_spirv,
    lower_one_effect_float_to_spirv,
    validate_checked_float_binary_map,
    validate_checked_float_unary_map,
    validate_checked_float_unary_roles_map,
    validate_composed_float_map,
    validate_one_effect_float_map,
    PcuSpirvComposedFloatProfile,
    PcuSpirvLoweringOptions,
};
#[rustfmt::skip]
use crate::{
    ffi::VulkanPreparedComposed,
    PcuVulkanBackend,
    PcuVulkanCallMeasurements,
    PcuVulkanComposedMemoryRealizations,
    PcuVulkanError,
};

/// Frozen static native program with private transactional resource shadows.
pub struct PcuVulkanPreparedComposed {
    native: VulkanPreparedComposed,
    profile: PcuSpirvComposedFloatProfile,
}

impl PcuVulkanPreparedComposed {
    pub(crate) fn admitted_profile(
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<PcuSpirvComposedFloatProfile> {
        validate_composed_float_map(kernel)
            .or_else(|error| {
                // Exact primitive maps keep their frozen executors and identities.
                if validate_checked_float_binary_map(kernel).is_ok()
                    || validate_checked_float_unary_map(kernel).is_ok()
                    || validate_checked_float_unary_roles_map(kernel).is_ok()
                {
                    Err(error)
                } else {
                    validate_one_effect_float_map(kernel)
                }
            })
            .ok()
            .filter(|profile| {
                (0..profile.declarations().len()).all(|index| {
                    matches!(
                        profile.declaration_access(index),
                        Some(PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite)
                    )
                })
            })
    }

    pub(super) fn prepare(
        backend: &PcuVulkanBackend,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self, PcuVulkanError> {
        if cfg!(target_endian = "big") {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        let mut words = Vec::new();
        let admitted =
            Self::admitted_profile(kernel).ok_or(PcuVulkanError::UnsupportedPreparedProfile)?;
        let lower = if admitted.is_one_effect() {
            lower_one_effect_float_to_spirv::<Vec<u32>>
        } else {
            lower_composed_float_to_spirv::<Vec<u32>>
        };
        let (_, profile) = lower(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        if (0..profile.declarations().len()).any(|index| {
            !matches!(
                profile.declaration_access(index),
                Some(PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite)
            )
        }) {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        backend.device.validate_composed_geometry(&profile)?;
        let native = VulkanPreparedComposed::new(Rc::clone(&backend.device), &words, &profile)?;
        Ok(Self { native, profile })
    }
}

impl PcuVulkanPreparedComposed {
    #[must_use]
    pub const fn profile(&self) -> &PcuSpirvComposedFloatProfile {
        &self.profile
    }

    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanComposedMemoryRealizations> {
        self.native.memory_realizations()
    }

    /// Measures synchronous execution including whole-call validation and publication.
    ///
    /// # Errors
    /// Returns preflight, native, protocol or arithmetic failures. Fatal calls preserve
    /// every caller output; completed recovered calls publish useful output and return a fault.
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
        let positions = schema::validate(arguments, &self.profile)?;
        self.native
            .call(&self.profile, arguments, positions, measurements)
    }
}

impl PcuPreparedHostKernel for PcuVulkanPreparedComposed {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}

#[path = "schema/schema.rs"]
mod schema;
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
