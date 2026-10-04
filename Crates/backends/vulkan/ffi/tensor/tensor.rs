//! Retained compute resources binding actual same-session tensor owners, without host readback.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFaultKind,
};
use fusion_pcu::dialect::tensor::TensorArithmeticStep;
use super::prepared::StatusPolicy;
#[path = "status/status.rs"]
mod status;
pub use status::TensorStatusPolicy;
#[rustfmt::skip]
use super::{
    allocate_command_buffer,
    allocate_descriptor_set,
    create_command_pool,
    create_compute_pipeline,
    create_fence,
    create_map_descriptor_pool,
    create_map_descriptor_set_layout,
    create_pipeline_layout,
    create_shader_module,
    mem,
    storage_buffer_write,
    submit_and_wait,
    validate_device_geometry,
    validate_map_descriptor_count,
    vk,
    vk_try,
    PcuVulkanError,
    VulkanDevice,
    VulkanOwnedBuffer,
    SHADER_ENTRY_POINT,
};

pub struct VulkanPreparedTensorMap<const N: usize> {
    device: Rc<VulkanDevice>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    descriptors: vk::DescriptorSet,
    command_pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    status: VulkanOwnedBuffer,
    lengths: [usize; N],
    extent: usize,
    groups: u32,
    status_policy: TensorStatusPolicy,
}

pub struct VulkanCompoundFault {
    pub element_index: usize,
    pub reduction_index: usize,
    pub step: TensorArithmeticStep,
    pub kind: PcuExecutionFaultKind,
}

impl<const N: usize> VulkanPreparedTensorMap<N> {
    #[allow(clippy::too_many_lines)] // Cold native RAII locals preserve rollback for every partial allocation.
    pub(crate) fn new(
        device: &Rc<VulkanDevice>,
        words: &[u32],
        extent: u32,
        dispatch_extent: u32,
        local_size: [u32; 3],
        lengths: [usize; N],
        status_policy: TensorStatusPolicy,
    ) -> Result<Self, PcuVulkanError> {
        #[cfg(feature = "insights")]
        let _api_scope = device.api_scope();
        if !matches!(N, 3 | 4) || extent == 0 || device.poisoned.get() {
            return Err(if device.poisoned.get() {
                PcuVulkanError::Quarantined
            } else {
                PcuVulkanError::InvalidArguments
            });
        }
        if usize::try_from(extent)
            .ok()
            .and_then(|n| n.checked_mul(status_policy.record_bytes()))
            != Some(lengths[N - 1])
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
        if local_size[0] == 0 {
            return Err(PcuVulkanError::InvalidArguments);
        }
        validate_map_descriptor_count(
            &device.limits,
            u32::try_from(N).map_err(|_| PcuVulkanError::InvalidArguments)?,
        )?;
        let groups = dispatch_extent.div_ceil(local_size[0]);
        let maximum = lengths
            .into_iter()
            .max()
            .unwrap_or(0)
            .checked_add(3)
            .ok_or(PcuVulkanError::BufferTooLarge)?
            & !3;
        // The logical U32 status span is 4*extent and fits maxStorageBufferRange (a U32
        // Vulkan limit). Thus extent+3, packed lane products and F64 doubled indices cannot
        // wrap. Wide integer lowering independently proves the full limb byte product.
        validate_device_geometry(&device.limits, [groups, 1, 1], local_size, maximum)?;
        let status = VulkanOwnedBuffer::new(device, lengths[N - 1])?;
        let shader = create_shader_module(device, words)?;
        let descriptor_layout = create_map_descriptor_set_layout::<N>(&device.device)?;
        let pipeline_layout = create_pipeline_layout(&device.device, descriptor_layout.handle)?;
        let pipeline = create_compute_pipeline(
            &device.device,
            shader.handle,
            pipeline_layout.handle,
            SHADER_ENTRY_POINT,
        )?;
        let descriptor_pool = create_map_descriptor_pool::<N>(&device.device)?;
        let descriptors = allocate_descriptor_set(
            &device.device,
            descriptor_pool.handle,
            descriptor_layout.handle,
        )?;
        let command_pool = create_command_pool(&device.device, device.queue_family_index)?;
        let command = allocate_command_buffer(&device.device, command_pool.handle)?;
        let fence = create_fence(&device.device)?;
        let result = Self {
            device: Rc::clone(device),
            shader: shader.handle,
            descriptor_layout: descriptor_layout.handle,
            pipeline_layout: pipeline_layout.handle,
            pipeline: pipeline.handle,
            descriptor_pool: descriptor_pool.handle,
            descriptors,
            command_pool: command_pool.handle,
            command,
            fence: fence.handle,
            status,
            lengths,
            extent: extent as usize,
            groups,
            status_policy,
        };
        vk_transfer_guards!(
            shader,
            descriptor_layout,
            pipeline_layout,
            pipeline,
            descriptor_pool,
            command_pool,
            fence,
        );
        Ok(result)
    }

    /// All binders are borrowed through terminal completion; outputs never alias inputs.
    pub(crate) fn call(&mut self, values: &[&VulkanOwnedBuffer]) -> Result<(), PcuVulkanError> {
        if self.extent.checked_mul(4) != Some(self.status.byte_len()) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let TensorStatusPolicy::Scalar(policy) = self.status_policy else {
            return Err(PcuVulkanError::InvalidArguments);
        };
        self.bind_and_submit(values)?;
        policy.scan(self.extent, |invocation| self.status.status(invocation))?;
        Ok(())
    }

    /// Separate three-word status ABI retains the ordered reduction/step provenance.
    pub(crate) fn call_compound(
        &mut self,
        values: &[&VulkanOwnedBuffer],
    ) -> Result<Option<VulkanCompoundFault>, PcuVulkanError> {
        if self.extent.checked_mul(12) != Some(self.status.byte_len()) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let TensorStatusPolicy::Compound(policy) = self.status_policy else {
            return Err(PcuVulkanError::InvalidArguments);
        };
        self.bind_and_submit(values)?;
        policy.scan(self.extent, |element| {
            Ok([
                self.status.status(element * 3)?,
                self.status.status(element * 3 + 1)?,
                self.status.status(element * 3 + 2)?,
            ])
        })
    }

    fn bind_and_submit(&mut self, values: &[&VulkanOwnedBuffer]) -> Result<(), PcuVulkanError> {
        if self.device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        if values.len() != N - 1 {
            return Err(PcuVulkanError::InvalidArguments);
        }
        for (value, bytes) in values.iter().zip(self.lengths) {
            value.validate_access_available()?;
            if !value.same_session(&self.device) || value.byte_len() != bytes {
                return Err(PcuVulkanError::InvalidArguments);
            }
        }
        let output = values[N - 2].descriptor_info().buffer;
        if values[..N - 2]
            .iter()
            .any(|value| value.descriptor_info().buffer == output)
        {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let infos: [_; N] = core::array::from_fn(|index| {
            [if index == N - 1 {
                self.status.descriptor_info()
            } else {
                values[index].descriptor_info()
            }]
        });
        let writes: [_; N] = core::array::from_fn(|index| {
            storage_buffer_write(
                self.descriptors,
                u32::try_from(index).expect("bounded three/four descriptor profile"),
                &infos[index],
            )
        });
        self.submit(&writes)
    }

    fn submit(&mut self, writes: &[vk::WriteDescriptorSet<'_>]) -> Result<(), PcuVulkanError> {
        let device = &self.device.device;
        vk_try("reset tensor compute command", unsafe {
            // SAFETY: Every previous submission is terminal, and this pool supports reset.
            vk_api_owner!(
                self.device,
                ResetCommandBuffer,
                device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
            )
        })?;
        unsafe {
            // SAFETY: All descriptors refer to same-session retained owners through completion.
            // Reset precedes updates: non-update-after-bind descriptor changes invalidate old commands.
            vk_api_owner!(
                self.device,
                UpdateDescriptorSets,
                device.update_descriptor_sets(writes, &[])
            );
        }
        vk_try("reset tensor compute fence", unsafe {
            // SAFETY: Synchronous previous submission is terminal; fence is not pending.
            vk_api_owner!(self.device, ResetFences, device.reset_fences(&[self.fence]))
        })?;
        record(
            &self.device,
            self.command,
            self.pipeline,
            self.pipeline_layout,
            self.descriptors,
            self.groups,
        )?;
        if let Err(error) = submit_and_wait(&self.device, self.command, self.fence) {
            if matches!(error, PcuVulkanError::CompletionUnknown) {
                // All tensor buffers/submission objects deliberately retain native handles on Drop.
                self.device.poisoned.set(true);
                mem::forget(Rc::clone(&self.device));
            }
            return Err(error);
        }
        Ok(())
    }
}

fn record(
    session: &VulkanDevice,
    command: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    descriptors: vk::DescriptorSet,
    groups: u32,
) -> Result<(), PcuVulkanError> {
    let device = &session.device;
    vk_try("begin tensor compute", unsafe {
        // SAFETY: Command was reset, belongs to this device and is not pending.
        vk_api_owner!(
            session,
            BeginCommandBuffer,
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
        )
    })?;
    unsafe {
        // SAFETY: Compatible pipeline/layout/descriptors and actual initialized owner extents.
        // Host uploads, earlier copies and previous graph steps explicitly become shader-readable.
        let readable = [vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        vk_api_owner!(
            session,
            PipelineBarrier,
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &readable,
                &[],
                &[],
            )
        );
        vk_api_owner!(
            session,
            BindPipeline,
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, pipeline)
        );
        vk_api_owner!(
            session,
            BindDescriptorSets,
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &[descriptors],
                &[],
            )
        );
        vk_api_owner!(
            session,
            Dispatch,
            device.cmd_dispatch(command, groups, 1, 1)
        );
        let terminal = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        vk_api_owner!(
            session,
            PipelineBarrier,
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &terminal,
                &[],
                &[],
            )
        );
    }
    vk_try("end tensor compute", unsafe {
        // SAFETY: Command remains recording and every referenced owner is retained.
        vk_api_owner!(
            session,
            EndCommandBuffer,
            device.end_command_buffer(command)
        )
    })
}

impl<const N: usize> Drop for VulkanPreparedTensorMap<N> {
    fn drop(&mut self) {
        if self.device.poisoned.get() {
            return;
        }
        unsafe {
            // SAFETY: Every submission completed; all handles are unique and outlive references.
            let device = &self.device.device;
            vk_api_owner!(
                self.device,
                DestroyFence,
                device.destroy_fence(self.fence, None)
            );
            vk_api_owner!(
                self.device,
                DestroyCommandPool,
                device.destroy_command_pool(self.command_pool, None)
            );
            vk_api_owner!(
                self.device,
                DestroyDescriptorPool,
                device.destroy_descriptor_pool(self.descriptor_pool, None)
            );
            vk_api_owner!(
                self.device,
                DestroyPipeline,
                device.destroy_pipeline(self.pipeline, None)
            );
            vk_api_owner!(
                self.device,
                DestroyPipelineLayout,
                device.destroy_pipeline_layout(self.pipeline_layout, None)
            );
            vk_api_owner!(
                self.device,
                DestroyDescriptorSetLayout,
                device.destroy_descriptor_set_layout(self.descriptor_layout, None)
            );
            vk_api_owner!(
                self.device,
                DestroyShaderModule,
                device.destroy_shader_module(self.shader, None)
            );
        }
    }
}
