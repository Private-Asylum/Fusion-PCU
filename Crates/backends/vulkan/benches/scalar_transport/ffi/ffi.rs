//! Independent ash/compiler transport control. Diagnostic `from_words` explicitly accepts provider bytecode; new compiles the audited arithmetic source independently.

#[rustfmt::skip]
use std::{
    error::Error,
    ffi::CStr,
    mem,
    ptr,
};
use ash::vk;
#[rustfmt::skip]
use pcu_facade::{PcuStableDeviceIdentity,PcuScalar};

type NativeResult<T> = Result<T, Box<dyn Error>>;

#[path = "resident/resident.rs"]
pub mod resident;

struct Storage {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    allocation: u64,
    memory_type: u32,
    flags: u32,
}

/// An independent logical device, two retained buffers and one terminal fence.
pub struct NativeTransport {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    physical: vk::PhysicalDevice,
    descriptor_set: vk::DescriptorSet,
    buffers: Vec<Storage>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptors: vk::DescriptorPool,
    commands: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    extra_fences: Vec<vk::Fence>,
    input_len: usize,
    input_storage_len: usize,
    byte_len: usize,
    storage_len: usize,
    pending: bool,
}

impl NativeTransport {
    /// Independently compiles the audited U32 transport shader and creates native ownership.
    /// # Errors
    /// Returns shader-tool, UUID/device, geometry, allocation or native API failures.
    pub fn new(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        element_bytes: usize,
        broadcast: bool,
    ) -> NativeResult<Self> {
        let words = compile_shader(extent, element_bytes, broadcast)?;
        Self::from_words(identity, extent, element_bytes, broadcast, &words)
    }
    /// Diagnostic preparation for explicitly supplied provider words.
    /// # Errors
    /// Returns schema, UUID/device, geometry, allocation or native API failures.
    #[allow(clippy::too_many_lines)] // Complete cold native RAII rollback remains adjacent.
    pub fn from_words(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        element_bytes: usize,
        broadcast: bool,
        words: &[u32],
    ) -> NativeResult<Self> {
        if cfg!(target_endian = "big")
            || extent == 0
            || !matches!(element_bytes, 1 | 2 | 4 | 8 | 16 | 32 | 64)
        {
            return Err("native transport host/schema unsupported".into());
        }
        let byte_len = usize::try_from(extent)?
            .checked_mul(element_bytes)
            .filter(|bytes| u32::try_from(*bytes).is_ok())
            .ok_or("native logical extent overflow")?;
        let input_len = if broadcast { element_bytes } else { byte_len };
        let storage_len = byte_len
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .ok_or("native padded extent overflow")?;
        let input_storage_len = input_len
            .checked_add(3)
            .map(|bytes| bytes & !3)
            .ok_or("native padded input overflow")?;
        // SAFETY: ash loads the system Vulkan library and the owner retains it through device Drop.
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default()
            .application_name(c"pcu-native-neg-control")
            .api_version(vk::API_VERSION_1_1);
        // SAFETY: The stack create info is valid and no extension pointers are supplied.
        let instance = unsafe {
            entry.create_instance(
                &vk::InstanceCreateInfo::default().application_info(&app),
                None,
            )?
        };
        let selection = select_device(&instance, identity, false);
        let (physical, family) = match selection {
            Ok(selected) => selected,
            Err(error) => {
                // SAFETY: No logical device or native resource has been created.
                unsafe {
                    instance.destroy_instance(None);
                }
                return Err(error);
            }
        };
        let priorities = [1.0];
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&priorities)];
        let features = vk::PhysicalDeviceFeatures::default().shader_float64(false);
        // SAFETY: Selected physical device and compute queue belong to this instance; only Shader/U32 is required.
        let device = match unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(&queues)
                    .enabled_features(&features),
                None,
            )
        } {
            Ok(device) => device,
            Err(error) => {
                // SAFETY: Failed logical-device creation leaves this instance with no child owners.
                unsafe {
                    instance.destroy_instance(None);
                }
                return Err(error.into());
            }
        };
        // SAFETY: Queue zero was requested for this selected compute family.
        let queue = unsafe { device.get_device_queue(family, 0) };
        let mut owner = Self {
            entry,
            instance,
            device,
            queue,
            physical,
            descriptor_set: vk::DescriptorSet::null(),
            buffers: Vec::new(),
            shader: vk::ShaderModule::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptors: vk::DescriptorPool::null(),
            commands: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            extra_fences: Vec::new(),
            input_len,
            input_storage_len,
            byte_len,
            storage_len,
            pending: false,
        };
        for bytes in [owner.input_storage_len, owner.storage_len] {
            owner.buffers.push(create_buffer(
                &owner.instance,
                physical,
                &owner.device,
                bytes,
            )?);
        }
        for (buffer, length) in owner.buffers.iter().zip([input_storage_len, storage_len]) {
            unsafe {
                // SAFETY: Cold uniquely owned mapping covers this complete private buffer; no pending commands.
                ptr::write_bytes(buffer.mapped, 0, length);
            }
        }
        let layouts: Vec<_> = (0..2)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        // SAFETY: All create infos reference live owner handles or stack/owned arrays through each call.
        unsafe {
            owner.shader = owner
                .device
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)?;
            owner.descriptor_layout = owner.device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layouts),
                None,
            )?;
            let sets = [owner.descriptor_layout];
            owner.pipeline_layout = owner.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                None,
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(owner.shader)
                .name(c"main");
            owner.pipeline = match owner.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                &[vk::ComputePipelineCreateInfo::default()
                    .stage(stage)
                    .layout(owner.pipeline_layout)],
                None,
            ) {
                Ok(pipelines) => pipelines[0],
                Err((partial, error)) => {
                    for pipeline in partial {
                        owner.device.destroy_pipeline(pipeline, None);
                    }
                    return Err(error.into());
                }
            };
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(2)];
            owner.descriptors = owner.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let set = owner.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(owner.descriptors)
                    .set_layouts(&sets),
            )?[0];
            owner.descriptor_set = set;
            let infos: Vec<_> = owner
                .buffers
                .iter()
                .enumerate()
                .map(|(id, buffer)| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(buffer.buffer)
                        .range(if id == 0 {
                            owner.input_storage_len as u64
                        } else {
                            owner.storage_len as u64
                        })]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(u32::try_from(binding).expect("two bindings"))
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            owner.device.update_descriptor_sets(&writes, &[]);
            owner.commands = owner.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            owner.command = owner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(owner.commands)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            owner
                .device
                .begin_command_buffer(owner.command, &vk::CommandBufferBeginInfo::default())?;
            owner.device.cmd_bind_pipeline(
                owner.command,
                vk::PipelineBindPoint::COMPUTE,
                owner.pipeline,
            );
            owner.device.cmd_bind_descriptor_sets(
                owner.command,
                vk::PipelineBindPoint::COMPUTE,
                owner.pipeline_layout,
                0,
                &[set],
                &[],
            );
            owner.device.cmd_dispatch(
                owner.command,
                u32::try_from(byte_len)?.div_ceil(4).div_ceil(64),
                1,
                1,
            );
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            owner.device.cmd_pipeline_barrier(
                owner.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &barrier,
                &[],
                &[],
            );
            owner.device.end_command_buffer(owner.command)?;
            owner.fence = owner
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?;
        }
        println!(
            "native cold allocations {:?}",
            owner
                .buffers
                .iter()
                .map(|buffer| (buffer.allocation, buffer.memory_type, buffer.flags))
                .collect::<Vec<_>>()
        );
        Ok(owner)
    }

    /// Transfers one complete initialized output after terminal native completion.
    /// # Errors
    /// Returns preflight or native submission/completion failure.
    pub fn call(&mut self, input: &[u8], output: &mut [u8]) -> NativeResult<()> {
        if output.len() < self.byte_len {
            return Err("native output extent".into());
        }
        self.submit(input)?;
        unsafe {
            // SAFETY: Terminal initialized private output and exclusive caller RAM are disjoint.
            ptr::copy_nonoverlapping(self.buffers[1].mapped, output.as_mut_ptr(), self.byte_len);
        }
        Ok(())
    }
    fn submit(&mut self, input: &[u8]) -> NativeResult<()> {
        if self.pending || input.len() < self.input_len {
            return Err("native input extent or unresolved work".into());
        }
        unsafe {
            // SAFETY: Coherent persistent mapping is disjoint from caller RAM; all prior work completed.
            ptr::copy_nonoverlapping(input.as_ptr(), self.buffers[0].mapped, self.input_len);
            self.device.reset_fences(&[self.fence])?;
        }
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        self.pending = true;
        unsafe {
            // SAFETY: Recorded command and retained resources belong to this serial queue; fence proves terminal completion.
            self.device.queue_submit(self.queue, &submits, self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        self.pending = false;
        Ok(())
    }
}

impl Drop for NativeTransport {
    fn drop(&mut self) {
        // SAFETY: Successfully completed calls are terminal. A failed pending call requires idle proof;
        // absent proof, abandon this independent session rather than freeing possibly used resources.
        unsafe {
            if self.pending && self.device.device_wait_idle().is_err() {
                mem::forget(self.entry.clone());
                return;
            }
            for fence in &self.extra_fences {
                self.device.destroy_fence(*fence, None);
            }
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.commands, None);
            self.device.destroy_descriptor_pool(self.descriptors, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.descriptor_layout, None);
            self.device.destroy_shader_module(self.shader, None);
            for buffer in &self.buffers {
                self.device.unmap_memory(buffer.memory);
                self.device.destroy_buffer(buffer.buffer, None);
                self.device.free_memory(buffer.memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn compile_shader(extent: u32, element_bytes: usize, broadcast: bool) -> NativeResult<Vec<u32>> {
    let stem = std::env::temp_dir().join(format!(
        "pcu-native-transport-{}-{extent}-{element_bytes}-{broadcast}",
        std::process::id()
    ));
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../spirv/scalar_transport/shader/scalar_transport.comp"
    ))
    .replace(
        "const uint ELEMENT_BYTES=4u",
        &format!("const uint ELEMENT_BYTES={element_bytes}u"),
    )
    .replace(
        "const uint EXTENT=1u",
        &format!("const uint EXTENT={extent}u"),
    )
    .replace(
        "const uint BROADCAST=0u",
        &format!("const uint BROADCAST={}u", u32::from(broadcast)),
    );
    let shader = stem.with_extension("comp");
    let output = stem.with_extension("spv");
    std::fs::write(&shader, source)?;
    let result = std::process::Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.0", "-Os"])
        .arg(&shader)
        .arg("-o")
        .arg(&output)
        .output()?;
    std::fs::remove_file(shader)?;
    if !result.status.success() {
        return Err(format!(
            "native compiler {} {}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        )
        .into());
    }
    let bytes = std::fs::read(&output)?;
    std::fs::remove_file(output)?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

fn select_device(
    instance: &ash::Instance,
    identity: PcuStableDeviceIdentity,
    f64: bool,
) -> NativeResult<(vk::PhysicalDevice, u32)> {
    // SAFETY: All inventory queries use a live instance and returned physical handles.
    unsafe {
        for physical in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical);
            if properties.device_type == vk::PhysicalDeviceType::CPU {
                continue;
            }
            let mut id = vk::PhysicalDeviceIDProperties::default();
            instance.get_physical_device_properties2(
                physical,
                &mut vk::PhysicalDeviceProperties2::default().push_next(&mut id),
            );
            if identity.namespace() != "vulkan.deviceUUID" || identity.value() != id.device_uuid {
                continue;
            }
            if f64
                && instance
                    .get_physical_device_features(physical)
                    .shader_float64
                    == 0
            {
                return Err("native shaderFloat64 unavailable".into());
            }
            for (family, queue) in instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .enumerate()
            {
                if queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    println!(
                        "native physical UUID{:?}: {}",
                        id.device_uuid,
                        CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy()
                    );
                    return Ok((physical, u32::try_from(family)?));
                }
            }
        }
    }
    Err("native matched physical device unavailable".into())
}

fn create_buffer(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
    device: &ash::Device,
    bytes: usize,
) -> NativeResult<Storage> {
    // SAFETY: Device/physical handles share this instance and every failed partial allocation is cleaned.
    unsafe {
        let buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(bytes as u64)
                .usage(
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_SRC
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )?;
        let requirements = device.get_buffer_memory_requirements(buffer);
        let properties = instance.get_physical_device_memory_properties(physical);
        let needed = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let compatible = |index: u32, cached: bool| {
            requirements.memory_type_bits & (1 << index) != 0
                && properties.memory_types[index as usize]
                    .property_flags
                    .contains(
                        needed
                            | if cached {
                                vk::MemoryPropertyFlags::HOST_CACHED
                            } else {
                                vk::MemoryPropertyFlags::empty()
                            },
                    )
        };
        let selected = (0..properties.memory_type_count)
            .find(|index| compatible(*index, true))
            .or_else(|| (0..properties.memory_type_count).find(|index| compatible(*index, false)));
        let Some(memory_type) = selected else {
            device.destroy_buffer(buffer, None);
            return Err("native coherent memory unavailable".into());
        };
        let memory = match device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        ) {
            Ok(memory) => memory,
            Err(error) => {
                device.destroy_buffer(buffer, None);
                return Err(error.into());
            }
        };
        let mapped = match device.bind_buffer_memory(buffer, memory, 0).and_then(|()| {
            device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }) {
            Ok(mapped) => mapped.cast(),
            Err(error) => {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return Err(error.into());
            }
        };
        Ok(Storage {
            buffer,
            memory,
            mapped,
            allocation: requirements.size,
            memory_type,
            flags: properties.memory_types[memory_type as usize]
                .property_flags
                .as_raw(),
        })
    }
}

#[derive(Default, Debug)]
pub struct HeapCounts {
    pub allocations: usize,
    pub reallocations: usize,
    pub frees: usize,
}

pub struct CountingAllocator;
static COUNTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static ALLOCATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static REALLOCATIONS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
static FREES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

// SAFETY: Every request is forwarded unchanged to System, and returned allocation pointers are unchanged.
unsafe impl std::alloc::GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if COUNTING.load(std::sync::atomic::Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: Caller supplies a valid GlobalAlloc request; System receives the same layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        if COUNTING.load(std::sync::atomic::Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: Caller supplies a valid GlobalAlloc request; System receives the same layout.
        unsafe { std::alloc::System.alloc_zeroed(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: std::alloc::Layout) {
        if COUNTING.load(std::sync::atomic::Ordering::Relaxed) {
            FREES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: Caller supplies the live allocation and original layout unchanged to System.
        unsafe {
            std::alloc::System.dealloc(pointer, layout);
        }
    }
    unsafe fn realloc(
        &self,
        pointer: *mut u8,
        layout: std::alloc::Layout,
        bytes: usize,
    ) -> *mut u8 {
        if COUNTING.load(std::sync::atomic::Ordering::Relaxed) {
            REALLOCATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        // SAFETY: Caller supplies a valid allocation/layout and nonzero new size unchanged to System.
        unsafe { std::alloc::System.realloc(pointer, layout, bytes) }
    }
}

pub fn count_heap(run: impl FnOnce()) -> HeapCounts {
    use std::sync::atomic::Ordering;
    ALLOCATIONS.store(0, Ordering::Relaxed);
    REALLOCATIONS.store(0, Ordering::Relaxed);
    FREES.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    run();
    COUNTING.store(false, Ordering::Relaxed);
    HeapCounts {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        reallocations: REALLOCATIONS.load(Ordering::Relaxed),
        frees: FREES.load(Ordering::Relaxed),
    }
}

/// Exact sealed, initialized, padding-free scalar bytes.
pub const fn bytes<T: PcuScalar>(values: &[T]) -> &[u8] {
    // SAFETY: PcuScalar is sealed to padding-free initialized carriers; shared lifetime is preserved.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}
/// Exact sealed scalar storage; every scalar carrier permits arbitrary bit patterns.
pub const fn bytes_mut<T: PcuScalar>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: Sealed PcuScalar carriers have no padding and all bit patterns are valid; unique lifetime is preserved.
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), std::mem::size_of_val(values))
    }
}
