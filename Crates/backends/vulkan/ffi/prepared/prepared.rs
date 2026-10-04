//! Retained native bit-map resources, foreign calls and terminal/quarantine handling.

use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFault,
    PcuCheckedScalarFaultLaw,
    PcuRangePolicy,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    PcuSpirvBitMapProfile,
    PcuSpirvCheckedBinaryProfile,
};
#[rustfmt::skip]
use super::{
    allocate_command_buffer,
    allocate_descriptor_set,
    create_command_pool,
    create_compute_pipeline,
    create_map_descriptor_pool,
    create_map_descriptor_set_layout,
    create_fence,
    create_pipeline_layout,
    create_shader_module,
    record_compute_commands,
    submit_and_wait_measured,
    storage_buffer_write,
    validate_device_geometry,
    vk_try,
    mem,
    ptr,
    vk,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
    PcuVulkanCallMeasurements,
    VulkanBuffer,
    VulkanDevice,
    SHADER_ENTRY_POINT,
};

#[path = "composed/composed.rs"]
mod composed;
#[path = "ordered_transport/ordered_transport.rs"]
mod ordered_transport;
pub use ordered_transport::VulkanPreparedOrderedTransport;
#[path = "mixed/mixed.rs"]
mod mixed;
#[path = "status/status.rs"]
mod status;
pub use composed::{PcuVulkanComposedMemoryRealizations, VulkanPreparedComposed};
pub use status::StatusPolicy;
pub use mixed::{
    Input as VulkanMixedInput, Output as VulkanMixedOutput, WriteState as VulkanWriteState,
};

struct RetainedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    realization: PcuVulkanMemoryRealization,
}

struct NativeResources<const N: usize> {
    buffers: [RetainedBuffer; N],
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    descriptors: vk::DescriptorSet,
    descriptor_buffers: [usize; 5],
    descriptor_count: usize,
}

/// Persistent prepared execution owns the device and all native resource handles.
struct VulkanPreparedMap<const N: usize, const CHECKED: bool = true, const OUTPUTS: usize = 1> {
    device: Rc<VulkanDevice>,
    resources: Option<NativeResources<N>>,
    extent: usize,
    lengths: [usize; N],
    status_policy: StatusPolicy,
    groups: u32,
    commit: Option<mixed::commit::Commit>,
}

impl<const N: usize, const CHECKED: bool, const OUTPUTS: usize>
    VulkanPreparedMap<N, CHECKED, OUTPUTS>
{
    const OUTPUT: usize = N - OUTPUTS - CHECKED as usize;

    #[allow(clippy::too_many_lines)] // RAII locals retain complete cold-construction rollback.
    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        extent: u32,
        dispatch_extent: u32,
        local_size: [u32; 3],
        lengths: [usize; N],
        status_policy: StatusPolicy,
    ) -> Result<Self, PcuVulkanError> {
        Self::new_with_descriptors::<N>(
            device,
            words,
            extent,
            dispatch_extent,
            local_size,
            lengths,
            status_policy,
            core::array::from_fn(|index| index),
        )
    }

    #[allow(clippy::too_many_lines, clippy::too_many_arguments)] // Cold RAII construction freezes native owners and validated readonly descriptor aliases.
    fn new_with_descriptors<const D: usize>(
        device: Rc<VulkanDevice>,
        words: &[u32],
        extent: u32,
        dispatch_extent: u32,
        local_size: [u32; 3],
        lengths: [usize; N],
        status_policy: StatusPolicy,
        descriptor_buffers: [usize; D],
    ) -> Result<Self, PcuVulkanError> {
        #[cfg(feature = "insights")]
        let _api_scope = device.api_scope();
        if D > 5 || descriptor_buffers.iter().any(|index| *index >= N) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        if device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        // Logical transfers remain exact; private narrow descriptors span complete U32 words.
        let mut storage_lengths = [0; N];
        for (storage, logical) in storage_lengths.iter_mut().zip(lengths) {
            *storage = logical
                .checked_add(3)
                .map(|bytes| bytes & !3)
                .ok_or(PcuVulkanError::BufferTooLarge)?;
        }
        let groups = [dispatch_extent.div_ceil(local_size[0]), 1, 1];
        validate_device_geometry(
            &device.limits,
            groups,
            local_size,
            storage_lengths.into_iter().max().unwrap_or(0),
        )?;
        let extent = usize::try_from(extent).map_err(|_| PcuVulkanError::BufferTooLarge)?;
        let mut allocations: [Option<VulkanBuffer<'_>>; N] = core::array::from_fn(|_| None);
        for (allocation, length) in allocations.iter_mut().zip(storage_lengths) {
            *allocation = Some(VulkanBuffer::new_buffer(
                &device.instance,
                device.physical_device,
                &device.device,
                device.caps.api_version,
                length,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            )?);
        }
        let allocations =
            allocations.map(|allocation| allocation.expect("all buffers constructed"));
        let shader = create_shader_module(&device, words)?;
        let descriptor_layout = create_map_descriptor_set_layout::<D>(&device.device)?;
        let pipeline_layout = create_pipeline_layout(&device.device, descriptor_layout.handle)?;
        let pipeline = create_compute_pipeline(
            &device.device,
            shader.handle,
            pipeline_layout.handle,
            SHADER_ENTRY_POINT,
        )?;
        let descriptor_pool = create_map_descriptor_pool::<D>(&device.device)?;
        let descriptors = allocate_descriptor_set(
            &device.device,
            descriptor_pool.handle,
            descriptor_layout.handle,
        )?;
        let infos: [_; D] =
            core::array::from_fn(
                |index| [allocations[descriptor_buffers[index]].descriptor_info()],
            );
        let writes: [_; D] = core::array::from_fn(|index| {
            storage_buffer_write(
                descriptors,
                u32::try_from(index).expect("bounded map bindings"),
                &infos[index],
            )
        });
        unsafe {
            // SAFETY: All infos refer to live owners on this device. Callers prove aliases
            // readonly or statically unused; actual written resources and status never alias.
            vk_api_owner!(
                device,
                UpdateDescriptorSets,
                device.device.update_descriptor_sets(&writes, &[])
            );
        }
        let command_pool = create_command_pool(&device.device, device.queue_family_index)?;
        let command = allocate_command_buffer(&device.device, command_pool.handle)?;
        record_compute_commands(
            &device.device,
            command,
            pipeline.handle,
            pipeline_layout.handle,
            descriptors,
            groups[0],
        )?;
        let fence = create_fence(&device.device)?;
        let mut buffers: [_; N] = core::array::from_fn(|index| RetainedBuffer {
            buffer: allocations[index].buffer,
            memory: allocations[index].memory,
            mapped: ptr::null_mut(),
            realization: allocations[index].realization,
        });
        for (index, buffer) in buffers.iter_mut().enumerate() {
            buffer.mapped = vk_try("map retained Vulkan bit-map memory", unsafe {
                // SAFETY: Each memory owner is host-visible/coherent, live and not yet mapped.
                // Failed cold construction frees its own allocations, including mapped memory.
                vk_api_owner!(
                    device,
                    MapMemory,
                    device.device.map_memory(
                        buffer.memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::empty(),
                    )
                )
            })?
            .cast();
            if storage_lengths[index] != lengths[index] {
                unsafe {
                    // SAFETY: Cold uniquely owned mapping spans the complete padded descriptor.
                    // No command is submitted. Padding stays initialized through later logical
                    // prefix uploads and is never published into caller memory.
                    ptr::write_bytes(buffer.mapped, 0, storage_lengths[index]);
                }
            }
        }
        let mut retained_descriptors = [0; 5];
        retained_descriptors[..D].copy_from_slice(&descriptor_buffers);
        let resources = NativeResources {
            buffers,
            shader: shader.handle,
            descriptor_layout: descriptor_layout.handle,
            pipeline_layout: pipeline_layout.handle,
            pipeline: pipeline.handle,
            descriptor_pool: descriptor_pool.handle,
            command_pool: command_pool.handle,
            command,
            fence: fence.handle,
            descriptors,
            descriptor_buffers: retained_descriptors,
            descriptor_count: D,
        };
        // Ownership moves to the retained native handle set, whose Drop runs before its device.
        #[cfg(feature = "insights")]
        for allocation in allocations {
            vk_transfer_guards!(allocation);
        }
        #[cfg(not(feature = "insights"))]
        mem::forget(allocations);
        vk_transfer_guards!(
            shader,
            descriptor_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            command_pool,
            fence,
        );
        Ok(Self {
            device,
            resources: Some(resources),
            extent,
            lengths,
            status_policy,
            groups: groups[0],
            commit: None,
        })
    }

    pub(crate) fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; N]> {
        self.resources
            .as_ref()
            .map(|resources| core::array::from_fn(|index| resources.buffers[index].realization))
    }

    pub(crate) fn call(
        &mut self,
        inputs: &[&[u8]],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        self.call_outputs(inputs, &mut [output], measurements)
    }

    fn call_outputs(
        &mut self,
        inputs: &[&[u8]],
        outputs: &mut [&mut [u8]],
        mut measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        if self.device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        if inputs.len() != Self::OUTPUT
            || inputs
                .iter()
                .zip(self.lengths)
                .any(|(input, required)| input.len() < required)
            || outputs.len() != OUTPUTS
            || outputs
                .iter()
                .enumerate()
                .any(|(index, output)| output.len() < self.lengths[Self::OUTPUT + index])
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let resources = self.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
        let started = measurements.as_ref().map(|_| std::time::Instant::now());
        for (index, input) in inputs.iter().enumerate() {
            unsafe {
                // SAFETY: Previous calls are terminal; this mapping covers lengths[index],
                // including the independent four-byte allocation for a broadcast input.
                // Only a CPU copy observes caller RAM, and no host pointer is submitted to the GPU.
                ptr::copy_nonoverlapping(
                    input.as_ptr(),
                    resources.buffers[index].mapped,
                    self.lengths[index],
                );
            }
        }
        let uploaded = started.map(|_| std::time::Instant::now());
        vk_try("reset retained Vulkan bit-map fence", unsafe {
            // SAFETY: Every earlier successful or known-quiescent failed call has completed.
            vk_api_owner!(
                self.device,
                ResetFences,
                self.device.device.reset_fences(&[resources.fence])
            )
        })?;
        if let Err(error) = submit_and_wait_measured(
            &self.device,
            resources.command,
            resources.fence,
            measurements.as_deref_mut(),
        ) {
            if matches!(error, PcuVulkanError::CompletionUnknown) {
                self.device.poisoned.set(true);
                self.resources.take(); // Raw handles remain deliberately alive and mapped.
                // Retain the loader and whole session even if every public backend owner is dropped.
                mem::forget(Rc::clone(&self.device));
            }
            return Err(error);
        }
        let completed = started.map(|_| std::time::Instant::now());
        let recovered = if CHECKED {
            self.status_policy.scan(self.extent, |invocation| {
                // SAFETY: The complete writer initializes one U32 record per logical element;
                // the compute-to-host dependency and terminal fence precede this mapped read.
                Ok(unsafe {
                    resources.buffers[N - 1]
                        .mapped
                        .add(invocation * 4)
                        .cast::<u32>()
                        .read_unaligned()
                })
            })?
        } else {
            None
        };
        for (index, output) in outputs.iter_mut().enumerate() {
            unsafe {
                // SAFETY: Complete fatal scan and terminal fence precede publication of every
                // initialized output. Exclusive caller prefixes are disjoint from native buffers.
                ptr::copy_nonoverlapping(
                    resources.buffers[Self::OUTPUT + index].mapped,
                    output.as_mut_ptr(),
                    self.lengths[Self::OUTPUT + index],
                );
            }
        }
        if let (Some(measurements), Some(started), Some(uploaded), Some(completed)) =
            (measurements, started, uploaded, completed)
        {
            let end = std::time::Instant::now();
            measurements.upload = uploaded - started;
            measurements.submission =
                (completed - uploaded).saturating_sub(measurements.completion);
            measurements.diagnostic_publication = end - completed;
            measurements.wall = end - started;
        }
        recovered.map_or(Ok(()), |fault| Err(PcuVulkanError::Fault(fault)))
    }
}

impl<const N: usize, const CHECKED: bool, const OUTPUTS: usize> Drop
    for VulkanPreparedMap<N, CHECKED, OUTPUTS>
{
    fn drop(&mut self) {
        let Some(resources) = self.resources.take() else {
            return;
        };
        if self.device.poisoned.get() {
            mem::forget(Rc::clone(&self.device));
            return; // Another call's unknown completion keeps the entire session quarantined.
        }
        unsafe {
            // SAFETY: Calls wait synchronously, or remove/quarantine all owners on unknown
            // completion. These handles are uniquely owned and their Rc device is still live.
            let device = &self.device.device;
            vk_api_owner!(
                self.device,
                DestroyFence,
                device.destroy_fence(resources.fence, None)
            );
            vk_api_owner!(
                self.device,
                DestroyCommandPool,
                device.destroy_command_pool(resources.command_pool, None)
            );
            vk_api_owner!(
                self.device,
                DestroyDescriptorPool,
                device.destroy_descriptor_pool(resources.descriptor_pool, None)
            );
            vk_api_owner!(
                self.device,
                DestroyPipeline,
                device.destroy_pipeline(resources.pipeline, None)
            );
            vk_api_owner!(
                self.device,
                DestroyPipelineLayout,
                device.destroy_pipeline_layout(resources.pipeline_layout, None)
            );
            vk_api_owner!(
                self.device,
                DestroyDescriptorSetLayout,
                device.destroy_descriptor_set_layout(resources.descriptor_layout, None)
            );
            vk_api_owner!(
                self.device,
                DestroyShaderModule,
                device.destroy_shader_module(resources.shader, None)
            );
            for buffer in resources.buffers {
                vk_api_owner!(self.device, UnmapMemory, device.unmap_memory(buffer.memory));
                vk_api_owner!(
                    self.device,
                    DestroyBuffer,
                    device.destroy_buffer(buffer.buffer, None)
                );
                vk_api_owner!(
                    self.device,
                    FreeMemory,
                    device.free_memory(buffer.memory, None)
                );
            }
        }
    }
}

/// Exact unary transport/Neg owner; shares terminal and quarantine machinery with binary maps.
pub struct VulkanPreparedBitMap(VulkanPreparedMap<3>);
impl VulkanPreparedBitMap {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        self.0.enable_mixed()
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        self.0.call_mixed(inputs, outputs, state)
    }

    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: PcuSpirvBitMapProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_bit_map_geometry(profile)?;
        let extent = usize::try_from(profile.extent).map_err(|_| PcuVulkanError::BufferTooLarge)?;
        let bytes = extent
            .checked_mul(if profile.scalar == fusion_pcu::PcuScalarType::F64 {
                8
            } else {
                4
            })
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        Ok(Self(VulkanPreparedMap::new(
            device,
            words,
            profile.extent,
            profile.extent,
            profile.local_size,
            [
                bytes,
                bytes,
                extent
                    .checked_mul(4)
                    .ok_or(PcuVulkanError::BufferTooLarge)?,
            ],
            match profile.operation {
                fusion_pcu_spirv::PcuSpirvBitOperation::Copy => StatusPolicy::ZERO_ONLY,
                fusion_pcu_spirv::PcuSpirvBitOperation::CheckedNeg(underflow) => {
                    StatusPolicy::checked(
                        PcuCheckedScalarFaultLaw::float_unary(
                            profile.scalar,
                            fusion_pcu::model::PcuDispatchFloatUnaryOp::Neg,
                            PcuRangePolicy::Reject,
                            underflow,
                        ),
                        PcuRangePolicy::Reject,
                    )?
                }
            },
        )?))
    }
    pub(crate) fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.0.memory_realizations()
    }
    pub(crate) fn call(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        self.0.call(&[input], output, measurements)
    }
}
/// Integer-synthesized binary32/binary64 owner with two inputs and private output/status.
pub struct VulkanPreparedBinary(BinaryOwners);
enum BinaryOwners {
    Repeated(VulkanPreparedMap<3>),
    Distinct(VulkanPreparedMap<4>),
}
impl VulkanPreparedBinary {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            BinaryOwners::Repeated(plan) => plan.enable_mixed(),
            BinaryOwners::Distinct(plan) => plan.enable_mixed(),
        }
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            BinaryOwners::Repeated(plan) => plan.call_mixed(inputs, outputs, state),
            BinaryOwners::Distinct(plan) => plan.call_mixed(inputs, outputs, state),
        }
    }

    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: PcuSpirvCheckedBinaryProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_binary_geometry(profile)?;
        let bytes = |extent| {
            usize::try_from(extent)
                .ok()
                .and_then(|extent| extent.checked_mul(profile.element_bytes()))
                .ok_or(PcuVulkanError::BufferTooLarge)
        };
        let output = bytes(profile.extent)?;
        let status = usize::try_from(profile.extent)
            .ok()
            .and_then(|extent| extent.checked_mul(4))
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        let policy = StatusPolicy::checked(
            PcuCheckedScalarFaultLaw::float_binary(
                profile.scalar,
                profile.operation,
                profile.range,
                profile.underflow,
            ),
            profile.range,
        )?;
        if profile.input_count == 1 {
            Ok(Self(BinaryOwners::Repeated(
                VulkanPreparedMap::new_with_descriptors::<4>(
                    device,
                    words,
                    profile.extent,
                    profile.dispatch_extent(),
                    profile.local_size,
                    [bytes(profile.input_extent(0))?, output, status],
                    policy,
                    [0, 0, 1, 2],
                )?,
            )))
        } else {
            Ok(Self(BinaryOwners::Distinct(VulkanPreparedMap::new(
                device,
                words,
                profile.extent,
                profile.dispatch_extent(),
                profile.local_size,
                [
                    bytes(profile.input_extent(0))?,
                    bytes(profile.input_extent(1))?,
                    output,
                    status,
                ],
                policy,
            )?)))
        }
    }
    pub(crate) fn memory_realizations(&self) -> Option<crate::PcuVulkanBinaryMemoryRealizations> {
        match &self.0 {
            BinaryOwners::Repeated(plan) => plan
                .memory_realizations()
                .map(crate::PcuVulkanBinaryMemoryRealizations::RepeatedInput),
            BinaryOwners::Distinct(plan) => plan
                .memory_realizations()
                .map(crate::PcuVulkanBinaryMemoryRealizations::DistinctInputs),
        }
    }
    pub(crate) fn call(
        &mut self,
        inputs: &[&[u8]; 2],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            BinaryOwners::Repeated(plan) => plan.call(&inputs[..1], output, measurements),
            BinaryOwners::Distinct(plan) => plan.call(inputs, output, measurements),
        }
    }
}

/// U32-limb fourteen-carrier checked integer owner with unique inputs and private output/status.
pub struct VulkanPreparedInteger(IntegerOwners);
enum IntegerOwners {
    Repeated(VulkanPreparedMap<3>),
    Distinct(VulkanPreparedMap<4>),
}
impl VulkanPreparedInteger {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            IntegerOwners::Repeated(plan) => plan.enable_mixed(),
            IntegerOwners::Distinct(plan) => plan.enable_mixed(),
        }
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            IntegerOwners::Repeated(plan) => plan.call_mixed(inputs, outputs, state),
            IntegerOwners::Distinct(plan) => plan.call_mixed(inputs, outputs, state),
        }
    }

    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: fusion_pcu_spirv::PcuSpirvCheckedIntegerProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_integer_geometry(profile)?;
        let bytes = |extent| {
            usize::try_from(extent)
                .ok()
                .and_then(|extent| extent.checked_mul(profile.element_bytes()))
                .ok_or(PcuVulkanError::BufferTooLarge)
        };
        let output = bytes(profile.extent)?;
        let status = usize::try_from(profile.extent)
            .ok()
            .and_then(|extent| extent.checked_mul(4))
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        let policy = StatusPolicy::checked(
            PcuCheckedScalarFaultLaw::integer_binary(
                profile.scalar,
                profile.operation,
                profile.range,
            ),
            profile.range,
        )?;
        if profile.input_count == 1 {
            Ok(Self(IntegerOwners::Repeated(
                VulkanPreparedMap::new_with_descriptors::<4>(
                    device,
                    words,
                    profile.extent,
                    profile.dispatch_extent(),
                    profile.local_size,
                    [bytes(profile.input_extent(0))?, output, status],
                    policy,
                    [0, 0, 1, 2],
                )?,
            )))
        } else {
            Ok(Self(IntegerOwners::Distinct(VulkanPreparedMap::new(
                device,
                words,
                profile.extent,
                profile.dispatch_extent(),
                profile.local_size,
                [
                    bytes(profile.input_extent(0))?,
                    bytes(profile.input_extent(1))?,
                    output,
                    status,
                ],
                policy,
            )?)))
        }
    }
    pub(crate) fn memory_realizations(&self) -> Option<crate::PcuVulkanIntegerMemoryRealizations> {
        match &self.0 {
            IntegerOwners::Repeated(plan) => plan
                .memory_realizations()
                .map(crate::PcuVulkanIntegerMemoryRealizations::RepeatedInput),
            IntegerOwners::Distinct(plan) => plan
                .memory_realizations()
                .map(crate::PcuVulkanIntegerMemoryRealizations::DistinctInputs),
        }
    }
    pub(crate) fn call(
        &mut self,
        inputs: &[&[u8]; 2],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            IntegerOwners::Repeated(plan) => plan.call(&inputs[..1], output, measurements),
            IntegerOwners::Distinct(plan) => plan.call(inputs, output, measurements),
        }
    }
}

/// Pure-U32 six-format unary owner with one input and private output/status.
pub struct VulkanPreparedUnary(VulkanPreparedMap<3>);
impl VulkanPreparedUnary {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        self.0.enable_mixed()
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        self.0.call_mixed(inputs, outputs, state)
    }

    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: fusion_pcu_spirv::PcuSpirvCheckedUnaryProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_unary_geometry(profile)?;
        let bytes = |extent| {
            usize::try_from(extent)
                .ok()
                .and_then(|extent| extent.checked_mul(profile.element_bytes()))
                .ok_or(PcuVulkanError::BufferTooLarge)
        };
        Ok(Self(VulkanPreparedMap::new(
            device,
            words,
            profile.extent,
            profile.dispatch_extent(),
            profile.local_size,
            [
                bytes(profile.input_extent())?,
                bytes(profile.extent)?,
                usize::try_from(profile.extent)
                    .ok()
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(PcuVulkanError::BufferTooLarge)?,
            ],
            StatusPolicy::checked(
                PcuCheckedScalarFaultLaw::float_unary(
                    profile.scalar,
                    profile.operation,
                    profile.range,
                    profile.underflow,
                ),
                profile.range,
            )?,
        )?))
    }
    pub(crate) fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.0.memory_realizations()
    }
    pub(crate) fn call(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        self.0.call(&[input], output, measurements)
    }
}

/// Two-descriptor raw carrier transport, without a numerical status buffer or scan.
pub struct VulkanPreparedScalarTransport(VulkanPreparedMap<2, false>);
impl VulkanPreparedScalarTransport {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        self.0.enable_mixed()
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        self.0.call_mixed(inputs, outputs, state)
    }

    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: fusion_pcu_spirv::PcuSpirvScalarTransportProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_scalar_transport_geometry(profile)?;
        let output = usize::try_from(
            profile
                .logical_bytes()
                .ok_or(PcuVulkanError::BufferTooLarge)?,
        )
        .map_err(|_| PcuVulkanError::BufferTooLarge)?;
        let input = usize::try_from(profile.input_extent())
            .ok()
            .and_then(|n| n.checked_mul(profile.element_bytes()))
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        Ok(Self(VulkanPreparedMap::new(
            device,
            words,
            profile.extent,
            profile
                .dispatch_extent()
                .ok_or(PcuVulkanError::BufferTooLarge)?,
            profile.local_size,
            [input, output],
            StatusPolicy::ZERO_ONLY,
        )?))
    }
    pub(crate) fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 2]> {
        self.0.memory_realizations()
    }
    pub(crate) fn call(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        self.0.call(&[input], output, measurements)
    }
}

/// Two-output terminal integer division resources; one owner per distinct read buffer.
pub struct VulkanPreparedDivRem(DivRemOwners);
enum DivRemOwners {
    Repeated(VulkanPreparedMap<4, true, 2>),
    Distinct(VulkanPreparedMap<5, true, 2>),
}
impl VulkanPreparedDivRem {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            DivRemOwners::Repeated(plan) => plan.enable_mixed(),
            DivRemOwners::Distinct(plan) => plan.enable_mixed(),
        }
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            DivRemOwners::Repeated(plan) => plan.call_mixed(inputs, outputs, state),
            DivRemOwners::Distinct(plan) => plan.call_mixed(inputs, outputs, state),
        }
    }
    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: fusion_pcu_spirv::PcuSpirvCheckedDivRemProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_div_rem_geometry(profile)?;
        let bytes = |count| {
            usize::try_from(count)
                .ok()
                .and_then(|n| n.checked_mul(profile.element_bytes()))
                .ok_or(PcuVulkanError::BufferTooLarge)
        };
        let output = bytes(profile.extent)?;
        let status = usize::try_from(profile.extent)
            .ok()
            .and_then(|n| n.checked_mul(4))
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        if profile.input_count == 1 {
            Ok(Self(DivRemOwners::Repeated(
                VulkanPreparedMap::new_with_descriptors::<5>(
                    device,
                    words,
                    profile.extent,
                    profile.dispatch_extent(),
                    profile.local_size,
                    [bytes(profile.input_extents[0])?, output, output, status],
                    StatusPolicy::checked(
                        PcuCheckedScalarFaultLaw::integer_div_rem(profile.scalar),
                        PcuRangePolicy::Reject,
                    )?,
                    [0, 0, 1, 2, 3],
                )?,
            )))
        } else {
            Ok(Self(DivRemOwners::Distinct(VulkanPreparedMap::new(
                device,
                words,
                profile.extent,
                profile.dispatch_extent(),
                profile.local_size,
                [
                    bytes(profile.input_extents[0])?,
                    bytes(profile.input_extents[1])?,
                    output,
                    output,
                    status,
                ],
                StatusPolicy::checked(
                    PcuCheckedScalarFaultLaw::integer_div_rem(profile.scalar),
                    PcuRangePolicy::Reject,
                )?,
            )?)))
        }
    }
    pub(crate) fn memory_realizations(&self) -> Option<crate::PcuVulkanDivRemMemoryRealizations> {
        match &self.0 {
            DivRemOwners::Repeated(plan) => plan
                .memory_realizations()
                .map(crate::PcuVulkanDivRemMemoryRealizations::RepeatedInput),
            DivRemOwners::Distinct(plan) => plan
                .memory_realizations()
                .map(crate::PcuVulkanDivRemMemoryRealizations::DistinctInputs),
        }
    }
    pub(crate) fn call(
        &mut self,
        inputs: &[&[u8]; 2],
        outputs: &mut [&mut [u8]; 2],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        match &mut self.0 {
            DivRemOwners::Repeated(plan) => plan.call_outputs(&inputs[..1], outputs, measurements),
            DivRemOwners::Distinct(plan) => plan.call_outputs(inputs, outputs, measurements),
        }
    }
}

pub struct VulkanPreparedConversion(VulkanPreparedMap<3>);
impl VulkanPreparedConversion {
    pub(crate) fn enable_mixed(&mut self) -> Result<(), PcuVulkanError> {
        self.0.enable_mixed()
    }
    pub(crate) fn call_mixed(
        &mut self,
        inputs: &[VulkanMixedInput<'_>],
        outputs: &mut [VulkanMixedOutput<'_>],
        state: &mut VulkanWriteState,
    ) -> Result<(), PcuVulkanError> {
        self.0.call_mixed(inputs, outputs, state)
    }

    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: fusion_pcu_spirv::PcuSpirvCheckedConversionProfile,
    ) -> Result<Self, PcuVulkanError> {
        device.validate_conversion_geometry(profile)?;
        let bytes = |extent, scalar: fusion_pcu::PcuScalarType| {
            usize::try_from(extent)
                .ok()
                .and_then(|extent| extent.checked_mul(usize::from(scalar.bit_width() / 8)))
                .ok_or(PcuVulkanError::BufferTooLarge)
        };
        Ok(Self(VulkanPreparedMap::new(
            device,
            words,
            profile.extent,
            profile.extent,
            profile.local_size,
            [
                bytes(profile.input_extent(), profile.source_scalar())?,
                bytes(profile.extent, profile.output_scalar())?,
                usize::try_from(profile.extent)
                    .ok()
                    .and_then(|n| n.checked_mul(4))
                    .ok_or(PcuVulkanError::BufferTooLarge)?,
            ],
            StatusPolicy::checked(
                Some(PcuCheckedScalarFaultLaw::float_conversion(
                    profile.conversion,
                    profile.range,
                    profile.underflow,
                )),
                profile.range,
            )?,
        )?))
    }
    pub(crate) fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.0.memory_realizations()
    }
    pub(crate) fn call(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        self.0.call(&[input], output, measurements)
    }
}
