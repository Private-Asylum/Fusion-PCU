//! Independent native ash ownership/copy control; no PCU native or preparation calls.
#[rustfmt::skip]
use std::{ptr,rc::Rc};
use ash::vk;
use pcu_facade::PcuStableDeviceIdentity;
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
#[path = "../../checked_tensor/ffi/ffi.rs"]
#[allow(dead_code)]
// Only the owned pointwise benchmark consumes this shared native owner extension.
pub mod pointwise;

struct Context {
    poisoned: std::cell::Cell<bool>,
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    family: u32,
    memory: vk::PhysicalDeviceMemoryProperties,
}
impl Drop for Context {
    fn drop(&mut self) {
        if self.poisoned.get() {
            return;
        }
        // SAFETY: Copy/read APIs are synchronous and all buffers retain this logical device.
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
pub struct Owner {
    context: Rc<Context>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    bytes: usize,
}
impl Owner {
    fn new(context: &Rc<Context>, bytes: usize) -> Result<Self> {
        let size = bytes.checked_add(3).ok_or("extent overflow")? & !3;
        // SAFETY: Valid create info; buffers are private and all transfers stay on this device.
        let buffer = unsafe {
            context.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(u64::try_from(size.max(4))?)
                    .usage(
                        vk::BufferUsageFlags::TRANSFER_SRC
                            | vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                    )
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )?
        };
        // SAFETY: buffer was created on this live device.
        let requirements = unsafe { context.device.get_buffer_memory_requirements(buffer) };
        let memory_type = (0..context.memory.memory_type_count)
            .find(|&i| {
                requirements.memory_type_bits & (1 << i) != 0
                    && context.memory.memory_types[i as usize]
                        .property_flags
                        .contains(
                            vk::MemoryPropertyFlags::HOST_VISIBLE
                                | vk::MemoryPropertyFlags::HOST_COHERENT,
                        )
            })
            .ok_or("no coherent host memory")?;
        // SAFETY: Selected memory type satisfies the buffer requirements, allocation is fresh.
        let memory = unsafe {
            context.device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )?
        };
        // SAFETY: Allocation meets size/alignment/type requirements; unique allocation maps once.
        let mapped = unsafe {
            context.device.bind_buffer_memory(buffer, memory, 0)?;
            context
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
        }
        .cast::<u8>();
        // SAFETY: Coherent mapping spans the initialized private payload and word padding.
        unsafe {
            ptr::write_bytes(mapped, 0, size.max(4));
        }
        Ok(Self {
            context: Rc::clone(context),
            buffer,
            memory,
            mapped,
            bytes,
        })
    }
    pub fn read(&self, output: &mut [u8]) {
        assert_eq!(output.len(), self.bytes);
        // SAFETY: Terminal fence and transfer-to-host barrier precede this exact prefix copy.
        unsafe {
            ptr::copy_nonoverlapping(self.mapped, output.as_mut_ptr(), self.bytes);
        }
    }
    fn write(&mut self, input: &[u8]) {
        assert_eq!(input.len(), self.bytes);
        // SAFETY: Exclusive owner; prior operations completed, initialized mapping is coherent.
        unsafe {
            ptr::copy_nonoverlapping(input.as_ptr(), self.mapped, self.bytes);
        }
    }
}
impl Drop for Owner {
    fn drop(&mut self) {
        if self.context.poisoned.get() {
            return;
        }
        // SAFETY: All copy calls complete before exposing owners and the device is retained.
        unsafe {
            self.context.device.unmap_memory(self.memory);
            self.context.device.destroy_buffer(self.buffer, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}
pub struct NativeCopy {
    context: Rc<Context>,
    upload: Owner,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
}
impl NativeCopy {
    #[allow(clippy::too_many_lines)] // Independent native context/queue creation remains one cold owner.
    pub fn new(identity: PcuStableDeviceIdentity, bytes: usize) -> Result<Self> {
        // SAFETY: ash loads Vulkan; Context retains the loader through instance/device release.
        let entry = unsafe { ash::Entry::load()? };
        // SAFETY: Valid immutable create info, no borrowed extension pointers.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(
                    &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1),
                ),
                None,
            )?
        };
        // SAFETY: Live instance enumeration only.
        let devices = unsafe { instance.enumerate_physical_devices()? };
        let selected = devices
            .into_iter()
            .find_map(|physical| {
                let mut ids = vk::PhysicalDeviceIDProperties::default();
                let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut ids);
                // SAFETY: Properly linked stack output chains and live physical device.
                unsafe {
                    instance.get_physical_device_properties2(physical, &mut properties);
                }
                if identity.namespace() != "vulkan.deviceUUID"
                    || ids.device_uuid.as_slice() != identity.value()
                {
                    return None;
                }
                // SAFETY: Queries queue descriptors for this live physical device.
                let queues =
                    unsafe { instance.get_physical_device_queue_family_properties(physical) };
                queues
                    .iter()
                    .position(|queue| {
                        queue.queue_count != 0
                            && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                    .map(|family| (physical, u32::try_from(family).unwrap()))
            })
            .ok_or("native UUID/compute queue absent")?;
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(selected.1)
            .queue_priorities(&[1.0])];
        // SAFETY: Valid selected compute queue; no optional feature is required for transfer.
        let device = unsafe {
            instance.create_device(
                selected.0,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queues),
                None,
            )?
        };
        // SAFETY: Queue0 was created and physical memory query returns native fixed descriptors.
        let queue = unsafe { device.get_device_queue(selected.1, 0) };
        let memory = unsafe { instance.get_physical_device_memory_properties(selected.0) };
        let context = Rc::new(Context {
            poisoned: std::cell::Cell::new(false),
            _entry: entry,
            instance,
            device,
            queue,
            family: selected.1,
            memory,
        });
        let upload = Owner::new(&context, bytes)?;
        // SAFETY: Private resettable pool on selected queue, one retained primary command buffer.
        let pool = unsafe {
            context.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(selected.1)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?
        };
        let command = unsafe {
            context.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?
        }[0];
        // SAFETY: Valid initial unsignaled fence, used only by synchronous submissions.
        let fence = unsafe {
            context
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };
        Ok(Self {
            context,
            upload,
            pool,
            command,
            fence,
        })
    }
    pub fn copy_host(&mut self, input: &[u8]) -> Result<Owner> {
        self.upload.write(input);
        self.copy(&self.upload)
    }
    pub fn copy_owned(&self, input: &Owner) -> Result<Owner> {
        assert!(Rc::ptr_eq(&self.context, &input.context));
        self.copy(input)
    }
    fn copy(&self, input: &Owner) -> Result<Owner> {
        let output = Owner::new(&self.context, input.bytes)?;
        let device = &self.context.device;
        // SAFETY: All earlier uses are terminal; command and fence belong exclusively to this control.
        unsafe {
            device.reset_fences(&[self.fence])?;
            device.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                self.command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            let before = [vk::MemoryBarrier::default()
                .src_access_mask(
                    vk::AccessFlags::HOST_WRITE
                        | vk::AccessFlags::TRANSFER_WRITE
                        | vk::AccessFlags::SHADER_WRITE,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)];
            device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::HOST
                    | vk::PipelineStageFlags::TRANSFER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &before,
                &[],
                &[],
            );
            if input.bytes != 0 {
                device.cmd_copy_buffer(
                    self.command,
                    input.buffer,
                    output.buffer,
                    &[vk::BufferCopy::default().size(u64::try_from(
                        input.bytes.checked_add(3).ok_or("extent overflow")? & !3,
                    )?)],
                );
            }
            let after = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            device.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &after,
                &[],
                &[],
            );
            device.end_command_buffer(self.command)?;
            let completion = device
                .queue_submit(
                    self.context.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[self.command])],
                    self.fence,
                )
                .and_then(|()| device.wait_for_fences(&[self.fence], true, u64::MAX));
            if let Err(error) = completion {
                // Unknown completion must retain all native resources; this control stops here.
                std::mem::forget(output);
                self.context.poisoned.set(true);
                std::mem::forget(Rc::clone(&self.context));
                panic!("native owner control uncertain completion: {error:?}");
            }
        }
        Ok(output)
    }
}
impl Drop for NativeCopy {
    fn drop(&mut self) {
        if self.context.poisoned.get() {
            return;
        }
        // SAFETY: All calls terminal; retained owner context outlives these submission objects.
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context.device.destroy_command_pool(self.pool, None);
        }
    }
}
