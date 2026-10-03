//! Retained exact-byte public commits after complete private arithmetic validation.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use crate::ffi::{
    allocate_command_buffer,
    create_command_pool,
    create_fence,
    mem,
    submit_and_wait,
    vk,
    vk_try,
    VulkanDevice,
    VulkanOwnedBuffer,
    PcuVulkanError,
};

#[derive(Default)]
pub struct WriteState {
    pub(crate) may_have_written: bool,
    pub(crate) completion_uncertain: bool,
}
pub(super) struct Region<'a> {
    pub(super) source: vk::Buffer,
    pub(super) target: &'a VulkanOwnedBuffer,
    pub(super) bytes: usize,
}
pub(in crate::ffi::prepared) struct Commit {
    device: Rc<VulkanDevice>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}
impl Commit {
    pub(super) fn new(device: &Rc<VulkanDevice>) -> Result<Self, PcuVulkanError> {
        if device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        let pool = create_command_pool(&device.device, device.queue_family_index)?;
        let command = allocate_command_buffer(&device.device, pool.handle)?;
        let fence = create_fence(&device.device)?;
        let result = Self {
            device: Rc::clone(device),
            pool: pool.handle,
            command,
            fence: fence.handle,
        };
        mem::forget((pool, fence));
        Ok(result)
    }
    pub(super) fn execute(
        &mut self,
        regions: &[Region<'_>],
        state: &mut WriteState,
    ) -> Result<(), PcuVulkanError> {
        if self.device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        for (index, region) in regions.iter().enumerate() {
            region.target.validate_access_available()?;
            if !region.target.same_session(&self.device)
                || region.bytes == 0
                || region.bytes > region.target.byte_len()
                || region.source == region.target.descriptor_info().buffer
                || regions[..index].iter().any(|prior| {
                    prior.target.descriptor_info().buffer == region.target.descriptor_info().buffer
                })
            {
                return Err(PcuVulkanError::InvalidArguments);
            }
        }
        if regions.is_empty() {
            return Ok(());
        }
        let device = &self.device.device;
        vk_try("reset retained public commit command", unsafe {
            // SAFETY: Prior submissions are terminal; this uniquely owned pool permits reset.
            device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
        })?;
        vk_try("reset retained public commit fence", unsafe {
            // SAFETY: Every preceding call is terminal and this fence is not in use.
            device.reset_fences(&[self.fence])
        })?;
        record(device, self.command, regions)?;
        // Mutation is possible only when the validated public transfer is submitted. Earlier
        // argument failures, private-kernel faults and command-recording failures preserve Ready.
        state.may_have_written = true;
        if let Err(error) = submit_and_wait(device, self.device.queue, self.command, self.fence) {
            // This error is the native submission protocol's explicit lack of quiescence proof,
            // rather than a facade guess from an arbitrary public error variant.
            if matches!(error, PcuVulkanError::CompletionUnknown) {
                state.completion_uncertain = true;
                self.device.poisoned.set(true);
                mem::forget(Rc::clone(&self.device));
            }
            return Err(error);
        }
        Ok(())
    }
}
fn record(
    device: &ash::Device,
    command: vk::CommandBuffer,
    regions: &[Region<'_>],
) -> Result<(), PcuVulkanError> {
    vk_try("begin exact public Vulkan commit", unsafe {
        // SAFETY: Retained command belongs to a reset, nonpending pool.
        device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
    })?;
    unsafe {
        // SAFETY: Private compute outputs and exclusive public destination owners remain live
        // on this logical device through terminal completion; all spans/aliases were validated.
        let readable = [vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &readable,
            &[],
            &[],
        );
        for region in regions {
            // VkBufferCopy permits any positive exact byte size. Deliberately do not round
            // packed tails up: caller-owned U8/FP8/F16/BF16 suffix carriers must stay unchanged.
            // https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCopy.html
            // https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdCopyBuffer.html
            device.cmd_copy_buffer(
                command,
                region.source,
                region.target.descriptor_info().buffer,
                &[vk::BufferCopy::default().size(region.bytes as u64)],
            );
        }
        let visible = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(
                vk::AccessFlags::HOST_READ
                    | vk::AccessFlags::SHADER_READ
                    | vk::AccessFlags::TRANSFER_READ,
            )];
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &visible,
            &[],
            &[],
        );
    }
    vk_try("end exact public Vulkan commit", unsafe {
        // SAFETY: Command is recording and every referenced live owner outlives submission.
        device.end_command_buffer(command)
    })
}
impl Drop for Commit {
    fn drop(&mut self) {
        if self.device.poisoned.get() {
            return;
        }
        unsafe {
            // SAFETY: All submitted work is terminal; this uniquely owned pool/fence is retained
            // until completion and destroyed before its logical-device owner.
            self.device.device.destroy_fence(self.fence, None);
            self.device.device.destroy_command_pool(self.pool, None);
        }
    }
}
