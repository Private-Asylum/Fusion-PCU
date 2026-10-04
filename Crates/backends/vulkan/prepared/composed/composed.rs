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
    validate_checked_float_binary_map,
    validate_checked_float_unary_map,
    validate_checked_float_unary_roles_map,
    validate_composed_float_map,
    validate_composed_integer_map,
    validate_one_effect_integer_map,
    validate_checked_integer_map,
    validate_one_effect_float_map,
    PcuSpirvComposedProfile,
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
    profile: PcuSpirvComposedProfile,
}

impl PcuVulkanPreparedComposed {
    pub(crate) fn admitted_profile(
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<PcuSpirvComposedProfile> {
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
                        .or_else(|_| validate_composed_integer_map(kernel))
                        .or_else(|error| {
                            if validate_checked_integer_map(kernel).is_ok() {
                                Err(error)
                            } else {
                                validate_one_effect_integer_map(kernel)
                            }
                        })
                }
            })
            .ok()
            .filter(|profile| {
                profile.is_integer()
                    || (0..profile.declarations().len()).all(|index| {
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
        let family = match admitted.scalar() {
            fusion_pcu::PcuScalarType::F32 => fusion_pcu_spirv::PcuSpirvComposedTemplateFamily::F32,
            fusion_pcu::PcuScalarType::F64 => fusion_pcu_spirv::PcuSpirvComposedTemplateFamily::F64,
            fusion_pcu::PcuScalarType::U256
            | fusion_pcu::PcuScalarType::I256
            | fusion_pcu::PcuScalarType::U512
            | fusion_pcu::PcuScalarType::I512 => {
                fusion_pcu_spirv::PcuSpirvComposedTemplateFamily::IntegerWide
            }
            fusion_pcu::PcuScalarType::U8
            | fusion_pcu::PcuScalarType::I8
            | fusion_pcu::PcuScalarType::U16
            | fusion_pcu::PcuScalarType::I16
            | fusion_pcu::PcuScalarType::U32
            | fusion_pcu::PcuScalarType::I32
            | fusion_pcu::PcuScalarType::U64
            | fusion_pcu::PcuScalarType::I64
            | fusion_pcu::PcuScalarType::U128
            | fusion_pcu::PcuScalarType::I128 => {
                fusion_pcu_spirv::PcuSpirvComposedTemplateFamily::IntegerNarrow
            }
            _ => fusion_pcu_spirv::PcuSpirvComposedTemplateFamily::Low,
        };
        let profile = backend.device.shader_source.borrow_mut().lower(
            kernel,
            family,
            admitted.is_one_effect(),
            &mut words,
        )?;
        if !profile.is_integer()
            && (0..profile.declarations().len()).any(|index| {
                !matches!(
                    profile.declaration_access(index),
                    Some(PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite)
                )
            })
        {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        backend.device.validate_composed_geometry(&profile)?;
        let native = VulkanPreparedComposed::new(Rc::clone(&backend.device), &words, &profile)?;
        Ok(Self { native, profile })
    }
}

impl PcuVulkanPreparedComposed {
    #[must_use]
    pub const fn profile(&self) -> &PcuSpirvComposedProfile {
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
