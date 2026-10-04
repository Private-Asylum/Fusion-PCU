//! Native terminal buffer ownership and transfer submission.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use super::{
    allocate_command_buffer,
    create_command_pool,
    create_fence,
    ptr,
    submit_and_wait,
    vk,
    vk_try,
    PcuVulkanError,
    PcuVulkanMemoryRealization,
    VulkanBuffer,
    VulkanDevice,
};

pub struct VulkanOwnedBuffer {
    device: Rc<VulkanDevice>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    bytes: usize,
    realization: PcuVulkanMemoryRealization,
}

impl VulkanOwnedBuffer {
    pub(crate) const fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo {
            buffer: self.buffer,
            offset: 0,
            range: vk::WHOLE_SIZE,
        }
    }

    pub(crate) const fn byte_len(&self) -> usize {
        self.bytes
    }

    #[cfg(feature = "tensor")]
    pub(crate) fn status(&self, index: usize) -> Result<u32, PcuVulkanError> {
        self.check(self.bytes)?;
        let offset = index.checked_mul(4).ok_or(PcuVulkanError::BufferTooLarge)?;
        if offset.checked_add(4).is_none_or(|end| end > self.bytes) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        Ok(unsafe {
            // SAFETY: Caller waits for terminal compute with a compute-to-host dependency.
            // Checked offset spans an initialized complete U32 diagnostic in this mapping.
            self.mapped.add(offset).cast::<u32>().read_unaligned()
        })
    }
    pub(crate) fn new(device: &Rc<VulkanDevice>, bytes: usize) -> Result<Self, PcuVulkanError> {
        #[cfg(feature = "insights")]
        let _api_scope = device.api_scope();
        if device.poisoned.get() {
            return Err(PcuVulkanError::Quarantined);
        }
        // The prepared U32 storage ABI uses complete words. Logical byte extents remain exact
        // and padding is initialized privately, including the physical owner for an empty value.
        let padded = bytes.checked_add(3).ok_or(PcuVulkanError::BufferTooLarge)? & !3;
        let allocation = VulkanBuffer::new_buffer(
            &device.instance,
            device.physical_device,
            &device.device,
            device.caps.api_version,
            padded.max(4),
            vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_SRC
                | vk::BufferUsageFlags::TRANSFER_DST,
        )?;
        let mapped = vk_try("map owned Vulkan memory", unsafe {
            // SAFETY: The uniquely owned host-visible/coherent allocation is not already mapped.
            vk_api_owner!(
                device,
                MapMemory,
                device.device.map_memory(
                    allocation.memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
            )
        })?
        .cast::<u8>();
        unsafe {
            // SAFETY: This cold exclusive mapping spans the complete allocation descriptor.
            ptr::write_bytes(mapped, 0, padded.max(4));
        }
        let result = Self {
            device: Rc::clone(device),
            buffer: allocation.buffer,
            memory: allocation.memory,
            mapped,
            bytes,
            realization: allocation.realization,
        };
        vk_transfer_guards!(allocation); // Raw handles transfer into this owner, before its retained device.
        Ok(result)
    }

    pub(crate) fn write(&mut self, source: &[u8]) -> Result<(), PcuVulkanError> {
        self.check(source.len())?;
        unsafe {
            // SAFETY: No asynchronous API exposes this buffer; all previous work is terminal.
            // The initialized coherent allocation spans bytes and never aliases caller storage.
            ptr::copy_nonoverlapping(source.as_ptr(), self.mapped, self.bytes);
        }
        Ok(())
    }

    pub(crate) fn read(&self, target: &mut [u8]) -> Result<(), PcuVulkanError> {
        self.check(target.len())?;
        unsafe {
            // SAFETY: Completion and transfer-to-host dependency precede publication. Only the
            // exact logical prefix is copied; the caller's tail and private padding are untouched.
            ptr::copy_nonoverlapping(self.mapped, target.as_mut_ptr(), self.bytes);
        }
        Ok(())
    }

    fn check(&self, bytes: usize) -> Result<(), PcuVulkanError> {
        if self.device.poisoned.get() {
            Err(PcuVulkanError::Quarantined)
        } else if bytes < self.bytes {
            Err(PcuVulkanError::BufferTooSmall)
        } else {
            Ok(())
        }
    }

    pub(crate) fn validate_access_available(&self) -> Result<(), PcuVulkanError> {
        self.check(self.bytes)
    }

    pub(crate) fn same_session(&self, device: &Rc<VulkanDevice>) -> bool {
        Rc::ptr_eq(&self.device, device)
    }

    pub(crate) fn same_buffer_session(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.device, &other.device)
    }

    pub(crate) const fn realization(&self) -> PcuVulkanMemoryRealization {
        self.realization
    }

    pub(crate) fn copy_owned(&self) -> Result<Self, PcuVulkanError> {
        VulkanPreparedOwnedCopy::new(&self.device)?.copy(self)
    }
}

/// Cold retained copy resources; warm execution allocates only the distinct escaping buffer.
pub struct VulkanPreparedOwnedCopy {
    device: Rc<VulkanDevice>,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}

impl VulkanPreparedOwnedCopy {
    pub(crate) fn new(device: &Rc<VulkanDevice>) -> Result<Self, PcuVulkanError> {
        #[cfg(feature = "insights")]
        let _api_scope = device.api_scope();
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
        vk_transfer_guards!(pool, fence); // Retained handles are released before the Rc device owner.
        Ok(result)
    }

    pub(crate) fn copy(
        &mut self,
        source: &VulkanOwnedBuffer,
    ) -> Result<VulkanOwnedBuffer, PcuVulkanError> {
        source.check(source.bytes)?;
        if !source.same_session(&self.device) {
            return Err(PcuVulkanError::InvalidArguments);
        }
        let output = VulkanOwnedBuffer::new(&self.device, source.bytes)?;
        if source.bytes == 0 {
            return Ok(output);
        }
        let device = &self.device.device;
        vk_try("reset retained owned copy command", unsafe {
            // SAFETY: Every previous submission is terminal and the pool permits individual reset.
            vk_api_owner!(
                self.device,
                ResetCommandBuffer,
                device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
            )
        })?;
        vk_try("reset retained owned copy fence", unsafe {
            // SAFETY: Every previous submission is terminal; the fence is not currently in use.
            vk_api_owner!(self.device, ResetFences, device.reset_fences(&[self.fence]))
        })?;
        record_copy(
            &self.device,
            self.command,
            source.buffer,
            output.buffer,
            source.bytes,
        )?;
        if let Err(error) = submit_and_wait(&self.device, self.command, self.fence) {
            if matches!(error, PcuVulkanError::CompletionUnknown) {
                // No proof of quiescence: every owner and this retained pool/fence deliberately
                // keep their native handles alive on drop once the common session is poisoned.
                self.device.poisoned.set(true);
            }
            return Err(error);
        }
        Ok(output)
    }
}

impl Drop for VulkanPreparedOwnedCopy {
    fn drop(&mut self) {
        if self.device.poisoned.get() {
            return;
        }
        unsafe {
            // SAFETY: Synchronous transfer is terminal; pool also owns and frees its command.
            vk_api_owner!(
                self.device,
                DestroyFence,
                self.device.device.destroy_fence(self.fence, None)
            );
            vk_api_owner!(
                self.device,
                DestroyCommandPool,
                self.device.device.destroy_command_pool(self.pool, None)
            );
        }
    }
}

fn record_copy(
    session: &VulkanDevice,
    command: vk::CommandBuffer,
    source: vk::Buffer,
    target: vk::Buffer,
    bytes: usize,
) -> Result<(), PcuVulkanError> {
    let device = &session.device;
    let bytes = bytes.checked_add(3).ok_or(PcuVulkanError::BufferTooLarge)? & !3;
    let size = u64::try_from(bytes).map_err(|_| PcuVulkanError::BufferTooLarge)?;
    vk_try("begin owned Vulkan copy", unsafe {
        // SAFETY: The command belongs to a new pool and has not been submitted.
        vk_api_owner!(
            session,
            BeginCommandBuffer,
            device.begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
        )
    })?;
    unsafe {
        // SAFETY: Buffers belong to this device, have compatible transfer usage, disjoint live
        // allocations and full padded extents. Both owners remain retained until terminal wait.
        let upload = [vk::MemoryBarrier::default()
            // A borrowed source can be either a host upload or an earlier terminal native copy.
            // A fence wait alone is not used as a transfer-to-transfer memory dependency.
            .src_access_mask(
                vk::AccessFlags::HOST_WRITE
                    | vk::AccessFlags::TRANSFER_WRITE
                    | vk::AccessFlags::SHADER_WRITE,
            )
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
        vk_api_owner!(
            session,
            PipelineBarrier,
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &upload,
                &[],
                &[],
            )
        );
        vk_api_owner!(
            session,
            CopyBuffer,
            device.cmd_copy_buffer(
                command,
                source,
                target,
                &[vk::BufferCopy::default().size(size)],
            )
        );
        let readback = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        vk_api_owner!(
            session,
            PipelineBarrier,
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &readback,
                &[],
                &[],
            )
        );
    }
    vk_try("end owned Vulkan copy", unsafe {
        // SAFETY: Command remains in recording state and every referenced owner is live.
        vk_api_owner!(
            session,
            EndCommandBuffer,
            device.end_command_buffer(command)
        )
    })
}

impl Drop for VulkanOwnedBuffer {
    fn drop(&mut self) {
        if self.device.poisoned.get() {
            return; // Unknown completion retains mappings and native handles, never frees in use.
        }
        unsafe {
            // SAFETY: Synchronous submission is terminal; this owner uniquely holds these handles.
            vk_api_owner!(
                self.device,
                UnmapMemory,
                self.device.device.unmap_memory(self.memory)
            );
            vk_api_owner!(
                self.device,
                DestroyBuffer,
                self.device.device.destroy_buffer(self.buffer, None)
            );
            vk_api_owner!(
                self.device,
                FreeMemory,
                self.device.device.free_memory(self.memory, None)
            );
        }
    }
}
