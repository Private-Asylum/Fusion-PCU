//! Independent shader/ash resident publisher: private compute then exact-byte public transfer.
#[rustfmt::skip]
use super::{
    create_buffer,
    NativeResult,
    NativeTransport,
    ptr,
    vk,
};
use pcu_facade::PcuStableDeviceIdentity;

pub struct NativeResident {
    owner: NativeTransport,
    commit: vk::CommandBuffer,
    commit_fence: vk::Fence,
    output_bytes: usize,
}
impl NativeResident {
    pub fn new(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        element_bytes: usize,
        inputs: [&[u8]; 2],
        output: &[u8],
    ) -> NativeResult<Self> {
        let mut owner = NativeTransport::new(identity, extent, element_bytes, false)?;
        if inputs.iter().any(|input| input.len() != owner.input_len)
            || output.len() < owner.byte_len
        {
            return Err("native resident cold extent".into());
        }
        for bytes in [inputs[0], inputs[1], output] {
            let storage = create_buffer(
                &owner.instance,
                owner.physical,
                &owner.device,
                bytes.len().checked_add(3).ok_or("native padded extent")? & !3,
            )?;
            // SAFETY: Fresh coherent private owner spans padded bytes; no work submitted.
            unsafe {
                ptr::write_bytes(storage.mapped, 0, bytes.len().checked_add(3).unwrap() & !3);
                ptr::copy_nonoverlapping(bytes.as_ptr(), storage.mapped, bytes.len());
            }
            owner.buffers.push(storage);
        }
        // SAFETY: Retained resettable pool/device are live, and this command is uniquely owned.
        let commit = unsafe {
            owner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(owner.commands)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        // SAFETY: Valid unsignaled fence; used only by this synchronous public publisher.
        let commit_fence = unsafe {
            owner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        owner.extra_fences.push(commit_fence);
        Ok(Self {
            owner,
            commit,
            commit_fence,
            output_bytes: output.len(),
        })
    }
    pub fn call_owned(&mut self, bank: usize) -> NativeResult<()> {
        if bank >= 2 {
            return Err("native bank index".into());
        }
        self.call(bank + 2)
    }
    pub fn call_host(&mut self, input: &[u8]) -> NativeResult<()> {
        if input.len() < self.owner.input_len {
            return Err("native short host input".into());
        }
        // SAFETY: Prior work terminal, mapped upload is coherent and disjoint from caller RAM.
        unsafe {
            ptr::copy_nonoverlapping(
                input.as_ptr(),
                self.owner.buffers[0].mapped,
                self.owner.input_len,
            );
        }
        self.call(0)
    }
    fn call(&mut self, input: usize) -> NativeResult<()> {
        if self.owner.pending {
            return Err("native unresolved work".into());
        }
        self.record_compute(input)?;
        self.submit(self.owner.command, self.owner.fence)?;
        self.record_commit()?;
        self.submit(self.commit, self.commit_fence)
    }
    pub fn read(&self, output: &mut [u8]) {
        assert!(!self.owner.pending);
        assert!(output.len() >= self.output_bytes);
        // SAFETY: Complete transfer-to-host dependency and terminal wait precede initialized bytes.
        unsafe {
            ptr::copy_nonoverlapping(
                self.owner.buffers[4].mapped,
                output.as_mut_ptr(),
                self.output_bytes,
            );
        }
    }
    fn submit(&mut self, command: vk::CommandBuffer, fence: vk::Fence) -> NativeResult<()> {
        // SAFETY: All earlier work terminal; each private fence is unsignaled before submission.
        unsafe {
            self.owner.device.reset_fences(&[fence])?;
        }
        self.owner.pending = true;
        // SAFETY: All buffers and code are owned by this serial native root through terminal wait.
        unsafe {
            self.owner.device.queue_submit(
                self.owner.queue,
                &[vk::SubmitInfo::default().command_buffers(&[command])],
                fence,
            )?;
            self.owner
                .device
                .wait_for_fences(&[fence], true, u64::MAX)?;
        }
        self.owner.pending = false;
        Ok(())
    }
    fn record_compute(&self, input: usize) -> NativeResult<()> {
        let device = &self.owner.device;
        let infos = [
            [vk::DescriptorBufferInfo::default()
                .buffer(self.owner.buffers[input].buffer)
                .range(self.owner.input_storage_len as u64)],
            [vk::DescriptorBufferInfo::default()
                .buffer(self.owner.buffers[1].buffer)
                .range(self.owner.storage_len as u64)],
        ];
        let writes: [_; 2] = core::array::from_fn(|i| {
            vk::WriteDescriptorSet::default()
                .dst_set(self.owner.descriptor_set)
                .dst_binding(u32::try_from(i).unwrap())
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&infos[i])
        });
        // SAFETY: All previous commands terminal; reset invalidates old descriptor references.
        // Exact private bank owners remain live, padded shader spans cannot touch public tails.
        unsafe {
            device
                .reset_command_buffer(self.owner.command, vk::CommandBufferResetFlags::empty())?;
            device.update_descriptor_sets(&writes, &[]);
            device
                .begin_command_buffer(self.owner.command, &vk::CommandBufferBeginInfo::default())?;
            let before = [vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
            device.cmd_pipeline_barrier(
                self.owner.command,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &before,
                &[],
                &[],
            );
            device.cmd_bind_pipeline(
                self.owner.command,
                vk::PipelineBindPoint::COMPUTE,
                self.owner.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                self.owner.command,
                vk::PipelineBindPoint::COMPUTE,
                self.owner.pipeline_layout,
                0,
                &[self.owner.descriptor_set],
                &[],
            );
            device.cmd_dispatch(
                self.owner.command,
                u32::try_from(self.owner.byte_len)?.div_ceil(4).div_ceil(64),
                1,
                1,
            );
            let after = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(
                self.owner.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &after,
                &[],
                &[],
            );
            device.end_command_buffer(self.owner.command)?;
        }
        Ok(())
    }
    fn record_commit(&self) -> NativeResult<()> {
        let device = &self.owner.device;
        // SAFETY: Native compute and earlier public writes terminal; resettable retained command
        // exclusively owns exact positive nonoverlapping byte ranges on one logical device.
        unsafe {
            device.reset_command_buffer(self.commit, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(self.commit, &vk::CommandBufferBeginInfo::default())?;
            let before = [vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
            device.cmd_pipeline_barrier(
                self.commit,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &before,
                &[],
                &[],
            );
            device.cmd_copy_buffer(
                self.commit,
                self.owner.buffers[1].buffer,
                self.owner.buffers[4].buffer,
                &[vk::BufferCopy::default().size(u64::try_from(self.owner.byte_len)?)],
            );
            let after = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(
                    vk::AccessFlags::HOST_READ
                        | vk::AccessFlags::SHADER_READ
                        | vk::AccessFlags::TRANSFER_READ,
                )];
            device.cmd_pipeline_barrier(
                self.commit,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::COMPUTE_SHADER
                    | vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &after,
                &[],
                &[],
            );
            device.end_command_buffer(self.commit)?;
        }
        Ok(())
    }
}
