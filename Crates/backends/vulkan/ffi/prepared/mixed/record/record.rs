//! Fixed stack descriptor rebinding and explicit producer-to-compute dependencies.
#[rustfmt::skip]
use super::{
    Input,
    VulkanPreparedMap,
    PcuVulkanError,
    vk,
    vk_try,
    storage_buffer_write,
};
pub(super) fn record<const N: usize, const CHECKED: bool, const OUTPUTS: usize>(
    plan: &mut VulkanPreparedMap<N, CHECKED, OUTPUTS>,
    inputs: &[Input<'_>],
) -> Result<(), PcuVulkanError> {
    let resources = plan.resources.as_ref().ok_or(PcuVulkanError::Quarantined)?;
    let device = &plan.device.device;
    let infos: [_; 5] = core::array::from_fn(|descriptor| {
        let slot = resources.descriptor_buffers[descriptor];
        let buffer = if slot < inputs.len() {
            match inputs[slot] {
                Input::Host(_) => resources.buffers[slot].buffer,
                Input::Owned(owner) => owner.descriptor_info().buffer,
            }
        } else {
            resources.buffers[slot].buffer
        };
        let length = plan.lengths[slot]
            .checked_add(3)
            .expect("cold padded descriptor proof")
            & !3;
        [vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(0)
            .range(length.max(4) as u64)]
    });
    let writes: [_; 5] = core::array::from_fn(|descriptor| {
        storage_buffer_write(
            resources.descriptors,
            u32::try_from(descriptor).expect("cold descriptor count at most five"),
            &infos[descriptor],
        )
    });
    vk_try("reset mixed compute command", unsafe {
        // SAFETY: Previous commands are terminal and this pool permits individual reset.
        device.reset_command_buffer(resources.command, vk::CommandBufferResetFlags::empty())
    })?;
    unsafe {
        // SAFETY: Actual same-session owners were preflighted, descriptor ranges are bounded
        // by logical required spans, and no command remains pending or executable after reset.
        device.update_descriptor_sets(&writes[..resources.descriptor_count], &[]);
    }
    vk_try("begin mixed Vulkan compute", unsafe {
        // SAFETY: Retained command is reset and all bound resources remain live until completion.
        device.begin_command_buffer(resources.command, &vk::CommandBufferBeginInfo::default())
    })?;
    unsafe {
        // SAFETY: Validated same-session buffers, retained compatible pipeline/layout and bounded
        // cold dispatch geometry remain live. All earlier producer kinds are synchronized.
        let producers = [vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        device.cmd_pipeline_barrier(
            resources.command,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::TRANSFER
                | vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &producers,
            &[],
            &[],
        );
        device.cmd_bind_pipeline(
            resources.command,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline,
        );
        device.cmd_bind_descriptor_sets(
            resources.command,
            vk::PipelineBindPoint::COMPUTE,
            resources.pipeline_layout,
            0,
            &[resources.descriptors],
            &[],
        );
        device.cmd_dispatch(resources.command, plan.groups, 1, 1);
        let diagnostics = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        device.cmd_pipeline_barrier(
            resources.command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &diagnostics,
            &[],
            &[],
        );
    }
    vk_try("end mixed Vulkan compute", unsafe {
        // SAFETY: Command is recording and all referenced owners are retained through submission.
        device.end_command_buffer(resources.command)
    })
}
