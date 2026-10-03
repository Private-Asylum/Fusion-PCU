//! Cold role projection into the existing one-input/private-output U32 native ABI.
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_checked_float_unary_roles_to_spirv,
    PcuSpirvCheckedUnaryRoleProfile,
    PcuSpirvLoweringOptions,
};
#[rustfmt::skip]
use crate::{
    ffi::VulkanDevice,
    ffi::VulkanPreparedUnary,
    PcuVulkanCallMeasurements,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
};
#[rustfmt::skip]
use crate::ffi::{
    VulkanMixedInput,
    VulkanMixedOutput,
    VulkanWriteState,
};

/// Actual input/output native storage and original typed declarations retained only as cold metadata.
pub struct PcuVulkanPreparedUnaryRoles {
    native: VulkanPreparedUnary,
    profile: PcuSpirvCheckedUnaryRoleProfile,
    schema: Vec<(PcuBindingRef, PcuBindingAccess, usize)>,
}
impl PcuVulkanPreparedUnaryRoles {
    pub(crate) const fn declaration_schema(&self) -> &[(PcuBindingRef, PcuBindingAccess, usize)] {
        self.schema.as_slice()
    }

    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        self.native.enable_mixed()
    }

    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        self.native.call_mixed(inputs, outputs, state)
    }

    pub(crate) fn prepare(
        device: Rc<VulkanDevice>,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self, PcuVulkanError> {
        let mut words = Vec::new();
        let (_, profile) = lower_checked_float_unary_roles_to_spirv(
            kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut words,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        device.validate_unary_geometry(profile.native)?;
        let mut schema = Vec::with_capacity(kernel.bindings.len());
        for binding in kernel.bindings {
            let target = binding.reference();
            let count = if target == profile.description.output_binding {
                profile.native.extent
            } else if target == profile.description.input_binding {
                profile.native.input_extent()
            } else {
                0
            };
            let bytes = usize::try_from(count)
                .ok()
                .and_then(|count| count.checked_mul(profile.native.element_bytes()))
                .ok_or(PcuVulkanError::BufferTooLarge)?;
            schema.push((target, binding.access, bytes));
        }
        let native = VulkanPreparedUnary::new(device, &words, profile.native)?;
        Ok(Self {
            native,
            profile,
            schema,
        })
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.schema.len()
    }
    #[must_use]
    pub const fn profile(&self) -> PcuSpirvCheckedUnaryRoleProfile {
        self.profile
    }
    #[must_use]
    pub fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.native.memory_realizations()
    }
    /// Measures one complete call, including transactional status arbitration and publication.
    /// # Errors
    /// Returns preflight, arithmetic or native protocol failure; recovered Clamp publishes useful output.
    pub fn call_profiled(
        &mut self,
        arguments: &mut [PcuHostArgument<'_>],
    ) -> Result<PcuVulkanCallMeasurements, PcuVulkanError> {
        let mut measurements = PcuVulkanCallMeasurements::default();
        self.call_measured(arguments, Some(&mut measurements))?;
        Ok(measurements)
    }
    fn positions(&self, arguments: &[PcuHostArgument<'_>]) -> Result<[usize; 2], PcuVulkanError> {
        if arguments.len() != self.schema.len() {
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
        let mut positions = [0; 2];
        for &(target, access, bytes) in &self.schema {
            let index = arguments
                .iter()
                .position(|arg| arg.target() == target)
                .ok_or(PcuVulkanError::InvalidArguments)?;
            let argument = &arguments[index];
            if argument.scalar() != self.profile.native.scalar
                || argument.access() != access
                || argument.bytes().len() < bytes
            {
                return Err(PcuVulkanError::InvalidArguments);
            }
            if target == self.profile.description.input_binding {
                positions[0] = index;
            }
            if target == self.profile.description.output_binding {
                positions[1] = index;
            }
        }
        Ok(positions)
    }
    fn call_measured(
        &mut self,
        arguments: &mut [PcuHostArgument<'_>],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        let positions = self.positions(arguments)?;
        let (before, rest) = arguments.split_at_mut(positions[1]);
        let (output, after) = rest.split_first_mut().expect("validated output position");
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
impl PcuPreparedHostKernel for PcuVulkanPreparedUnaryRoles {
    type Error = PcuVulkanError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_measured(arguments, None)
    }
}
