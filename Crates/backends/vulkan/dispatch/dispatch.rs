//! Admission and SPIR-V lowering for the limited Vulkan prototype.
#[rustfmt::skip]
use core::{
    cell::Cell,
    marker::PhantomData,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDispatchSubmission,
    PcuInvocationBinding,
    PcuInvocationBindings,
    PcuInvocationParameters,
    PcuInvocationTarget,
    PcuKernelIrContract,
    dispatch::{
        validate_dispatch_submission,
        validate_invocation_bindings,
        validate_parameters,
    },
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_dispatch_to_spirv,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
};
#[rustfmt::skip]
use crate::{
    error::PcuVulkanError,
    ffi::VulkanDevice,
    types::{
        PcuVulkanCaps,
        PcuVulkanDescriptorHeapBudget,
        PcuVulkanDispatchReport,
        PcuVulkanLoweredSpirvDispatch,
    },
};

const SPIRV_WORD_CAPACITY: usize = 1024;

/// Concrete synchronous Vulkan prototype for legacy dispatch IR.
///
/// Uses two input buffers and one output buffer at set zero, bindings zero through two.
/// Scalar parameters and binding profiles beyond these three slots are rejected.
///
/// Vulkan queue host access requires exclusive serialization. This synchronous prototype
/// prevents shared cross-thread backend references by keeping the backend non-`Sync`.
///
/// ```compile_fail
/// use fusion_pcu_vulkan::PcuVulkanBackend;
/// fn requires_sync<T: Sync>() {}
/// requires_sync::<PcuVulkanBackend>();
/// ```
pub struct PcuVulkanBackend {
    device: VulkanDevice,
    // Queue submission must not be called concurrently through shared backend references.
    _exclusive_queue: PhantomData<Cell<()>>,
}

impl PcuVulkanBackend {
    /// Opens the first compute-capable Vulkan device.
    ///
    /// # Errors
    /// Returns loader, discovery, or device creation failures.
    pub fn new() -> Result<Self, PcuVulkanError> {
        VulkanDevice::new().map(|device| Self {
            device,
            _exclusive_queue: PhantomData,
        })
    }

    /// Inspects the first compute-capable device without retaining a logical device.
    ///
    /// # Errors
    /// Returns loader or compute-device discovery failures.
    pub fn probe() -> Result<PcuVulkanCaps, PcuVulkanError> {
        VulkanDevice::probe()
    }

    /// Opens a device with a requested descriptor-heap capability budget.
    ///
    /// The execution path still uses three fixed descriptors.
    ///
    /// # Errors
    /// Returns loader, discovery, or device creation failures.
    pub fn with_descriptor_heap_budget(
        requested_heap_budget: PcuVulkanDescriptorHeapBudget,
    ) -> Result<Self, PcuVulkanError> {
        VulkanDevice::with_descriptor_heap_budget(requested_heap_budget).map(|device| Self {
            device,
            _exclusive_queue: PhantomData,
        })
    }

    /// Inspects capabilities using a requested descriptor-heap budget.
    ///
    /// # Errors
    /// Returns loader or compute-device discovery failures.
    pub fn probe_with_descriptor_heap_budget(
        requested_heap_budget: PcuVulkanDescriptorHeapBudget,
    ) -> Result<PcuVulkanCaps, PcuVulkanError> {
        VulkanDevice::probe_with_descriptor_heap_budget(requested_heap_budget)
    }

    /// Returns capabilities observed for the selected device.
    #[must_use]
    pub const fn caps(&self) -> PcuVulkanCaps {
        self.device.caps()
    }

    /// Returns the physical device name reported by Vulkan.
    #[must_use]
    pub fn name(&self) -> &str {
        self.device.name()
    }

    /// Validates, lowers, and executes one dispatch, copying output before returning.
    ///
    /// # Errors
    /// Returns common admission, SPIR-V lowering, fixed-buffer geometry, or Vulkan failures.
    pub fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuInvocationBinding<'_>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuVulkanDispatchReport, PcuVulkanError> {
        validate_dispatch_submission(submission)
            .map_err(|error| PcuVulkanError::DispatchAdmission { error })?;
        validate_parameters(submission.kernel.signature(), parameters)
            .map_err(|error| PcuVulkanError::DispatchAdmission { error })?;
        validate_invocation_bindings(
            submission.kernel.signature(),
            PcuInvocationBindings { bindings },
        )
        .map_err(|error| PcuVulkanError::DispatchAdmission { error })?;

        validate_prototype_profile(bindings, parameters)?;

        let mut sink = PcuSpirvFixedSink::<SPIRV_WORD_CAPACITY>::new();
        let info = lower_dispatch_to_spirv(
            submission.kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        let lowered = PcuVulkanLoweredSpirvDispatch {
            words: sink.as_slice(),
            // The minimal shader emits one invocation per workgroup.
            local_size: [1, 1, 1],
        };
        let execution = self
            .device
            .submit_dispatch_spirv(&lowered, submission, bindings)?;
        Ok(PcuVulkanDispatchReport {
            device_name: self.name().to_owned(),
            spirv_words: info.word_count,
            spirv_bound: info.bound,
            execution,
        })
    }
}

fn validate_prototype_profile(
    bindings: &[PcuInvocationBinding<'_>],
    parameters: PcuInvocationParameters<'_>,
) -> Result<(), PcuVulkanError> {
    if !parameters.is_empty() {
        return Err(PcuVulkanError::UnsupportedParameters);
    }
    if bindings.len() != 3 {
        return Err(PcuVulkanError::UnsupportedBindingProfile);
    }
    let mut slots = 0_u8;
    for binding in bindings {
        let PcuInvocationTarget::Binding(reference) = binding.target else {
            return Err(PcuVulkanError::UnsupportedBindingProfile);
        };
        if reference.set != 0 || reference.binding > 2 {
            return Err(PcuVulkanError::UnsupportedBindingProfile);
        }
        slots |= 1 << reference.binding;
    }
    if slots != 0b111 {
        return Err(PcuVulkanError::UnsupportedBindingProfile);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBindingRef,
        PcuInvocationBinding,
        PcuInvocationBuffer,
        PcuInvocationParameters,
        PcuInvocationTarget,
        PcuParameterBinding,
        PcuParameterSlot,
        PcuParameterValue,
    };
    #[rustfmt::skip]
    use super::{
        validate_prototype_profile,
        PcuVulkanError,
    };

    const fn binding(set: u32, slot: u32) -> PcuInvocationBinding<'static> {
        PcuInvocationBinding {
            target: PcuInvocationTarget::Binding(PcuBindingRef::new(set, slot)),
            buffer: PcuInvocationBuffer::WordsIn(&[1]),
        }
    }

    #[test]
    fn fixed_profile_accepts_reordered_exact_slots() {
        let bindings = [binding(0, 2), binding(0, 0), binding(0, 1)];
        assert!(validate_prototype_profile(&bindings, PcuInvocationParameters::empty()).is_ok());
    }

    #[test]
    fn fixed_profile_rejects_extra_duplicate_or_wrong_set_bindings() {
        for bindings in [
            vec![binding(0, 0), binding(0, 1), binding(0, 2), binding(0, 3)],
            vec![binding(0, 0), binding(0, 1), binding(0, 1)],
            vec![binding(0, 0), binding(0, 1), binding(1, 2)],
        ] {
            assert!(matches!(
                validate_prototype_profile(&bindings, PcuInvocationParameters::empty()),
                Err(PcuVulkanError::UnsupportedBindingProfile)
            ));
        }
    }

    #[test]
    fn fixed_profile_rejects_scalar_parameters() {
        let bindings = [binding(0, 0), binding(0, 1), binding(0, 2)];
        let parameters = [PcuParameterBinding {
            slot: PcuParameterSlot(0),
            value: PcuParameterValue::U32(1),
        }];
        assert!(matches!(
            validate_prototype_profile(
                &bindings,
                PcuInvocationParameters {
                    bindings: &parameters
                }
            ),
            Err(PcuVulkanError::UnsupportedParameters)
        ));
    }
}
