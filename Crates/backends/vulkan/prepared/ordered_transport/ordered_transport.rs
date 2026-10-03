//! Frozen typed representation transport with private transactional native shadows.
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuPreparedHostKernel,
    PcuValueType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_ordered_scalar_transport_to_spirv,
    validate_ordered_scalar_transport_map,
    PcuSpirvLoweringOptions,
    PcuSpirvOrderedTransportProfile,
};
#[rustfmt::skip]
use crate::{
    ffi::VulkanPreparedOrderedTransport,
    PcuVulkanBackend,
    PcuVulkanCallMeasurements,
    PcuVulkanComposedMemoryRealizations,
    PcuVulkanError,
};

/// Prepared U32-only representation transport with private transactional shadows.
///
/// This adapter admits no arithmetic, Portable reproducibility or resident bindings. It is
/// selected only by its exact original typed load/store profile.
pub struct PcuVulkanPreparedOrderedTransport {
    native: VulkanPreparedOrderedTransport,
    profile: PcuSpirvOrderedTransportProfile,
    schema: schema::Schema,
}
impl PcuVulkanPreparedOrderedTransport {
    pub(crate) fn admitted_profile(
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<PcuSpirvOrderedTransportProfile> {
        let profile = validate_ordered_scalar_transport_map(kernel).ok()?;
        kernel
            .bindings
            .iter()
            .all(|binding| {
                binding.binding_type
                    == PcuBindingType::Value(PcuValueType::Scalar(profile.scalar()))
                    && matches!(
                        binding.access,
                        PcuBindingAccess::ReadOnly | PcuBindingAccess::ReadWrite
                    )
            })
            .then_some(profile)
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.schema.argument_count()
    }
    /// Freezes typed declarations, actual roles and the native load/store program cold.
    ///
    /// # Errors
    /// Refuses unsupported source/profile/access, geometry, lowering or device preparation.
    pub(super) fn prepare(
        backend: &PcuVulkanBackend,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self, PcuVulkanError> {
        if cfg!(target_endian = "big") {
            return Err(PcuVulkanError::UnsupportedPreparedProfile);
        }
        let mut words = Vec::new();
        let (_, profile) = lower_ordered_scalar_transport_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let schema = schema::Schema::new(kernel, &profile)?;
        let native =
            VulkanPreparedOrderedTransport::new(Rc::clone(&backend.device), &words, &profile)?;
        Ok(Self {
            native,
            profile,
            schema,
        })
    }
    #[must_use]
    pub const fn profile(&self) -> &PcuSpirvOrderedTransportProfile {
        &self.profile
    }
    #[must_use]
    pub fn memory_realizations(&self) -> Option<PcuVulkanComposedMemoryRealizations> {
        self.native.memory_realizations()
    }
    /// Measures whole synchronous validation, native work and terminal publication.
    ///
    /// # Errors
    /// Returns preflight, native or zero-only protocol failures before any caller output commit.
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
        let positions = self.schema.validate(arguments, &self.profile)?;
        self.native
            .call(&self.profile, arguments, positions, measurements)
    }
}
impl PcuPreparedHostKernel for PcuVulkanPreparedOrderedTransport {
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
