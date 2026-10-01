//! Retained native bit-map resources, foreign calls and terminal/quarantine handling.

use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
};
use fusion_pcu_spirv::PcuSpirvBitMapProfile;
#[rustfmt::skip]
use super::{
    allocate_command_buffer,
    allocate_descriptor_set,
    create_command_pool,
    create_compute_pipeline,
    create_descriptor_pool,
    create_descriptor_set_layout,
    create_fence,
    create_pipeline_layout,
    create_shader_module,
    record_compute_commands,
    submit_and_wait_measured,
    update_storage_descriptors,
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

struct RetainedBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    realization: PcuVulkanMemoryRealization,
}

struct NativeResources {
    buffers: [RetainedBuffer; 3],
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

/// Persistent prepared execution owns the device and all native resource handles.
pub struct VulkanPreparedBitMap {
    device: Rc<VulkanDevice>,
    resources: Option<NativeResources>,
    byte_len: usize,
    extent: usize,
}

impl VulkanPreparedBitMap {
    #[allow(clippy::too_many_lines)] // RAII locals retain complete cold-construction rollback.
    pub(crate) fn new(
        device: Rc<VulkanDevice>,
        words: &[u32],
        profile: PcuSpirvBitMapProfile,
    ) -> Result<Self, PcuVulkanError> {
        if device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        device.validate_bit_map_geometry(profile)?;
        let extent = usize::try_from(profile.extent).map_err(|_| PcuVulkanError::BufferTooLarge)?;
        let byte_len = usize::try_from(profile.extent)
            .ok()
            .and_then(|extent| {
                extent.checked_mul(if profile.scalar == fusion_pcu::PcuScalarType::F64 {
                    8
                } else {
                    4
                })
            })
            .ok_or(PcuVulkanError::BufferTooLarge)?;
        let groups = [profile.extent.div_ceil(profile.local_size[0]), 1, 1];
        validate_device_geometry(&device.limits, groups, profile.local_size, byte_len)?;
        let input = VulkanBuffer::new_storage_buffer(
            &device.instance,
            device.physical_device,
            &device.device,
            byte_len,
        )?;
        let output = VulkanBuffer::new_storage_buffer(
            &device.instance,
            device.physical_device,
            &device.device,
            byte_len,
        )?;
        let status = VulkanBuffer::new_storage_buffer(
            &device.instance,
            device.physical_device,
            &device.device,
            extent
                .checked_mul(4)
                .ok_or(PcuVulkanError::BufferTooLarge)?,
        )?;
        let shader = create_shader_module(&device.device, words)?;
        let descriptor_layout = create_descriptor_set_layout(&device.device)?;
        let pipeline_layout = create_pipeline_layout(&device.device, descriptor_layout.handle)?;
        let pipeline = create_compute_pipeline(
            &device.device,
            shader.handle,
            pipeline_layout.handle,
            SHADER_ENTRY_POINT,
        )?;
        let descriptor_pool = create_descriptor_pool(&device.device)?;
        let descriptors = allocate_descriptor_set(
            &device.device,
            descriptor_pool.handle,
            descriptor_layout.handle,
        )?;
        update_storage_descriptors(&device.device, descriptors, &input, &output, &status);
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
        let mut buffers = [
            RetainedBuffer {
                buffer: input.buffer,
                memory: input.memory,
                mapped: ptr::null_mut(),
                realization: input.realization,
            },
            RetainedBuffer {
                buffer: output.buffer,
                memory: output.memory,
                mapped: ptr::null_mut(),
                realization: output.realization,
            },
            RetainedBuffer {
                buffer: status.buffer,
                memory: status.memory,
                mapped: ptr::null_mut(),
                realization: status.realization,
            },
        ];
        for buffer in &mut buffers {
            buffer.mapped = vk_try("map retained Vulkan bit-map memory", unsafe {
                // SAFETY: Each memory owner is host-visible/coherent, live and not yet mapped.
                // Failed cold construction frees its own allocations, including mapped memory.
                device.device.map_memory(
                    buffer.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            })?
            .cast();
        }
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
        };
        // Ownership moves to the retained native handle set, whose Drop runs before its device.
        mem::forget((
            input,
            output,
            status,
            shader,
            descriptor_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            command_pool,
            fence,
        ));
        Ok(Self {
            device,
            resources: Some(resources),
            byte_len,
            extent,
        })
    }

    pub(crate) fn memory_realizations(&self) -> Option<[PcuVulkanMemoryRealization; 3]> {
        self.resources
            .as_ref()
            .map(|resources| core::array::from_fn(|index| resources.buffers[index].realization))
    }

    pub(crate) fn call(
        &mut self,
        input: &[u8],
        output: &mut [u8],
        mut measurements: Option<&mut PcuVulkanCallMeasurements>,
    ) -> Result<(), PcuVulkanError> {
        if self.device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        if input.len() < self.byte_len || output.len() < self.byte_len {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let resources = self.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
        let started = measurements.as_ref().map(|_| std::time::Instant::now());
        unsafe {
            // SAFETY: Previous calls are terminal; the persistent input mapping covers byte_len.
            // Only a CPU copy observes caller RAM, and no host pointer is submitted to the GPU.
            ptr::copy_nonoverlapping(input.as_ptr(), resources.buffers[0].mapped, self.byte_len);
        }
        let uploaded = started.map(|_| std::time::Instant::now());
        vk_try("reset retained Vulkan bit-map fence", unsafe {
            // SAFETY: Every earlier successful or known-quiescent failed call has completed.
            self.device.device.reset_fences(&[resources.fence])
        })?;
        if let Err(error) = submit_and_wait_measured(
            &self.device.device,
            self.device.queue,
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
        for invocation in 0..self.extent {
            let status = unsafe {
                // SAFETY: The complete writer initializes one u32 status per logical element;
                // compute-to-host dependency and fence wait precede this coherent mapped read.
                u32::from_ne_bytes(
                    resources.buffers[2]
                        .mapped
                        .add(invocation * 4)
                        .cast::<[u8; 4]>()
                        .read(),
                )
            };
            let kind = match status {
                0 => continue,
                1 => PcuExecutionFaultKind::InvalidFloatingOperand,
                2 => PcuExecutionFaultKind::ArithmeticUnderflow,
                _ => return Err(PcuVulkanError::InvalidArguments),
            };
            return Err(PcuVulkanError::Fault(PcuExecutionFault {
                recovered: false,
                kind,
                invocation_id: invocation as u64,
            }));
        }
        unsafe {
            // SAFETY: All checked status succeeded and output prefix is fully initialized and
            // terminal. Exclusive caller bytes are disjoint from the retained native allocation.
            ptr::copy_nonoverlapping(
                resources.buffers[1].mapped,
                output.as_mut_ptr(),
                self.byte_len,
            );
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
        Ok(())
    }
}

impl Drop for VulkanPreparedBitMap {
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
            device.destroy_fence(resources.fence, None);
            device.destroy_command_pool(resources.command_pool, None);
            device.destroy_descriptor_pool(resources.descriptor_pool, None);
            device.destroy_pipeline(resources.pipeline, None);
            device.destroy_pipeline_layout(resources.pipeline_layout, None);
            device.destroy_descriptor_set_layout(resources.descriptor_layout, None);
            device.destroy_shader_module(resources.shader, None);
            for buffer in resources.buffers {
                device.unmap_memory(buffer.memory);
                device.destroy_buffer(buffer.buffer, None);
                device.free_memory(buffer.memory, None);
            }
        }
    }
}
