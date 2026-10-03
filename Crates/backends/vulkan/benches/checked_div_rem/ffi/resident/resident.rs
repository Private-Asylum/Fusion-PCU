//! Independent resident joint control: retained inputs, private status, then exact public transfers.
#[rustfmt::skip]
use super::{
    create_buffer,
    mem,
    NativeDivRem,
    NativeResult,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuStableDeviceIdentity,
    ptr,
    vk,
};

pub struct NativeResidentDivRem {
    owner: Option<NativeDivRem>,
    inputs: [[usize; 2]; 2],
    outputs: [usize; 2],
    pool: vk::CommandPool,
    compute: vk::CommandBuffer,
    commit: vk::CommandBuffer,
    fence: vk::Fence,
    public_bytes: usize,
    element_bytes: usize,
}
impl NativeResidentDivRem {
    /// Creates an independent compiler/device root and cold retained banks.
    /// # Errors
    /// Rejects invalid extents, roles and native compiler/device/allocation failures.
    pub fn new(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        signed: bool,
        element_bytes: usize,
        kind: usize,
        inputs: [[&[u8]; 2]; 2],
        outputs: [&[u8]; 2],
    ) -> NativeResult<Self> {
        let owner = NativeDivRem::new_roles(identity, extent, signed, element_bytes, kind)?;
        if outputs[0].len() != outputs[1].len() || outputs[0].len() < owner.byte_len {
            return Err("native public output extent".into());
        }
        let mut result = Self {
            owner: Some(owner),
            inputs: [[0; 2]; 2],
            outputs: [0; 2],
            pool: vk::CommandPool::null(),
            compute: vk::CommandBuffer::null(),
            commit: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            public_bytes: outputs[0].len(),
            element_bytes,
        };
        result.allocate(inputs, outputs)?;
        let owner = result.owner.as_ref().ok_or("native missing owner")?;
        println!(
            "native resident cold buffers={} allocations {:?}; retained input banks and both public outputs",
            owner.buffers.len(),
            owner
                .buffers
                .iter()
                .map(|buffer| (buffer.allocation, buffer.memory_type, buffer.flags))
                .collect::<Vec<_>>()
        );
        Ok(result)
    }
    fn allocate(&mut self, inputs: [[&[u8]; 2]; 2], outputs: [&[u8]; 2]) -> NativeResult<()> {
        let owner = self.owner.as_mut().ok_or("native missing owner")?;
        for (input, banks) in inputs.iter().enumerate().take(owner.input_count) {
            for (bank, bytes) in banks.iter().enumerate() {
                if bytes.len() < owner.input_bytes[input] {
                    return Err("native resident input extent".into());
                }
                self.inputs[input][bank] = retain(owner, &bytes[..owner.input_bytes[input]])?;
            }
        }
        for (output, bytes) in outputs.into_iter().enumerate() {
            self.outputs[output] = retain(owner, bytes)?;
        }
        // SAFETY: The retained logical device and queue family are from one cold selection;
        // this resettable pool/fence belongs exclusively to the serial resident control.
        unsafe {
            self.pool = owner.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(owner.queue_family),
                None,
            )?;
            let commands = owner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2),
            )?;
            self.compute = commands[0];
            self.commit = commands[1];
            self.fence = owner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
        }
        Ok(())
    }
    /// Computes from one retained bank and commits both public owners after a full status scan.
    /// # Errors
    /// Rejects invalid banks, unresolved work, unknown status and native API failures.
    pub fn call(&mut self, bank: usize) -> NativeResult<Option<PcuExecutionFault>> {
        if bank >= 2 {
            return Err("native resident bank".into());
        }
        let owner = self.owner.as_mut().ok_or("native missing owner")?;
        if owner.pending {
            return Err("native unresolved work".into());
        }
        record_compute(
            owner,
            self.pool,
            self.compute,
            self.inputs,
            bank,
            self.element_bytes,
        )?;
        submit(owner, self.compute, self.fence)?;
        if let Some(fault) = fault(owner)? {
            return Ok(Some(fault));
        }
        record_commit(owner, self.commit, self.outputs)?;
        submit(owner, self.commit, self.fence)?;
        Ok(None)
    }
    /// Reads both completed public owners, including preserved storage tails.
    /// # Errors
    /// Rejects unresolved work or insufficient caller output extents.
    pub fn read(&self, outputs: [&mut [u8]; 2]) -> NativeResult<()> {
        let owner = self.owner.as_ref().ok_or("native missing owner")?;
        if owner.pending || outputs.iter().any(|bytes| bytes.len() < self.public_bytes) {
            return Err("native resident read extent/completion".into());
        }
        for (index, output) in outputs.into_iter().enumerate() {
            // SAFETY: Each private coherent owner has this initialized extent; terminal
            // transfer-to-host dependency precedes read, and Rust output slices are exclusive.
            unsafe {
                ptr::copy_nonoverlapping(
                    owner.buffers[self.outputs[index]].mapped,
                    output.as_mut_ptr(),
                    self.public_bytes,
                );
            }
        }
        Ok(())
    }
}
fn retain(owner: &mut NativeDivRem, bytes: &[u8]) -> NativeResult<usize> {
    let padded = bytes.len().checked_add(3).ok_or("native padded extent")? & !3;
    let storage = create_buffer(&owner.instance, owner.physical, &owner.device, padded)?;
    // SAFETY: Fresh coherent cold allocation is initialized before any submission.
    unsafe {
        ptr::write_bytes(storage.mapped, 0, padded);
        ptr::copy_nonoverlapping(bytes.as_ptr(), storage.mapped, bytes.len());
    }
    let index = owner.buffers.len();
    owner.buffers.push(storage);
    Ok(index)
}
fn submit(
    owner: &mut NativeDivRem,
    command: vk::CommandBuffer,
    fence: vk::Fence,
) -> NativeResult<()> {
    // SAFETY: Earlier submissions completed; this fence serializes one retained queue.
    unsafe {
        owner.device.reset_fences(&[fence])?;
    }
    owner.pending = true;
    // SAFETY: All shader, descriptor, command, buffer and device roots are retained through wait.
    unsafe {
        owner.device.queue_submit(
            owner.queue,
            &[vk::SubmitInfo::default().command_buffers(&[command])],
            fence,
        )?;
        owner.device.wait_for_fences(&[fence], true, u64::MAX)?;
    }
    owner.pending = false;
    Ok(())
}
fn fault(owner: &NativeDivRem) -> NativeResult<Option<PcuExecutionFault>> {
    for lane in 0..owner.extent {
        // SAFETY: Completed compute-to-host dependency precedes initialized coherent status reads.
        let status = u32::from_ne_bytes(unsafe {
            owner.buffers[owner.input_count + 2]
                .mapped
                .add(lane * 4)
                .cast::<[u8; 4]>()
                .read()
        });
        if status == 0 {
            continue;
        }
        let kind = match status {
            4 => PcuExecutionFaultKind::DivideByZero,
            5 => PcuExecutionFaultKind::SignedDivisionOverflow,
            _ => return Err("native unknown joint status".into()),
        };
        return Ok(Some(PcuExecutionFault {
            kind,
            invocation_id: u64::try_from(lane)?,
            recovered: false,
        }));
    }
    Ok(None)
}
fn record_compute(
    owner: &NativeDivRem,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    inputs: [[usize; 2]; 2],
    bank: usize,
    element_bytes: usize,
) -> NativeResult<()> {
    let left = inputs[0][bank];
    let right = if owner.input_count == 1 {
        left
    } else {
        inputs[1][bank]
    };
    let mapping = [
        left,
        right,
        owner.input_count,
        owner.input_count + 1,
        owner.input_count + 2,
    ];
    let infos: [_; 5] = core::array::from_fn(|binding| {
        let logical = match binding {
            1 if owner.input_count == 2 => owner.input_bytes[1],
            0 | 1 => owner.input_bytes[0],
            4 => owner.extent * 4,
            _ => owner.byte_len,
        };
        [vk::DescriptorBufferInfo::default()
            .buffer(owner.buffers[mapping[binding]].buffer)
            .range(((logical + 3) & !3) as u64)]
    });
    let writes: [_; 5] = core::array::from_fn(|binding| {
        vk::WriteDescriptorSet::default()
            .dst_set(owner.descriptor_set)
            .dst_binding(u32::try_from(binding).expect("five bindings"))
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&infos[binding])
    });
    // SAFETY: Previous work is terminal. Reset invalidates previous descriptor references;
    // repeated readonly operands alias one actual retained buffer and private writes are disjoint.
    unsafe {
        owner
            .device
            .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())?;
        owner.device.update_descriptor_sets(&writes, &[]);
        owner
            .device
            .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
        let before = [vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)];
        owner.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &before,
            &[],
            &[],
        );
        owner
            .device
            .cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, owner.pipeline);
        owner.device.cmd_bind_descriptor_sets(
            command,
            vk::PipelineBindPoint::COMPUTE,
            owner.pipeline_layout,
            0,
            &[owner.descriptor_set],
            &[],
        );
        owner.device.cmd_dispatch(
            command,
            u32::try_from(owner.extent)?
                .div_ceil(u32::try_from((4 / element_bytes).max(1))?)
                .div_ceil(64),
            1,
            1,
        );
        let after = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        owner.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &after,
            &[],
            &[],
        );
        owner.device.end_command_buffer(command)?;
    }
    Ok(())
}
fn record_commit(
    owner: &NativeDivRem,
    command: vk::CommandBuffer,
    outputs: [usize; 2],
) -> NativeResult<()> {
    // SAFETY: All status is terminal/nonfatal before either public write. Both ranges
    // are positive, disjoint and byte-exact; odd prefixes are deliberately not rounded.
    // VkBufferCopy/vkCmdCopyBuffer valid usage has no four-byte size restriction:
    // https://registry.khronos.org/vulkan/specs/latest/man/html/VkBufferCopy.html
    unsafe {
        owner
            .device
            .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())?;
        let before = [vk::MemoryBarrier::default()
            .src_access_mask(
                vk::AccessFlags::SHADER_WRITE
                    | vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ | vk::AccessFlags::TRANSFER_WRITE)];
        owner.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::COMPUTE_SHADER
                | vk::PipelineStageFlags::HOST
                | vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &before,
            &[],
            &[],
        );
        for (index, output) in outputs.into_iter().enumerate() {
            owner.device.cmd_copy_buffer(
                command,
                owner.buffers[owner.input_count + index].buffer,
                owner.buffers[output].buffer,
                &[vk::BufferCopy::default().size(owner.byte_len as u64)],
            );
        }
        let after = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        owner.device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &after,
            &[],
            &[],
        );
        owner.device.end_command_buffer(command)?;
    }
    Ok(())
}
impl Drop for NativeResidentDivRem {
    fn drop(&mut self) {
        let Some(owner) = self.owner.take() else {
            return;
        };
        // SAFETY: An unresolved native submission requires quiescence before freeing ANY
        // command/descriptor/code/buffer/queue root. If no proof exists, leak the entire lease.
        unsafe {
            if owner.pending && owner.device.device_wait_idle().is_err() {
                mem::forget(owner);
                return;
            }
            if self.fence != vk::Fence::null() {
                owner.device.destroy_fence(self.fence, None);
            }
            if self.pool != vk::CommandPool::null() {
                owner.device.destroy_command_pool(self.pool, None);
            }
        }
        drop(owner);
    }
}
