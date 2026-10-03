//! Independent ash/compiler control: no PCU lowering, preparation, executor or submission calls.
//! Arithmetic source is shared with the audited integer kernel; native lifecycle/compiler path is independent.

#[rustfmt::skip]
use std::{
    error::Error,
    ffi::CStr,
    mem,
    ptr,
    time::{
        Duration,
        Instant,
    },
};
use ash::vk;
#[rustfmt::skip]
use pcu_facade::{PcuStableDeviceIdentity,PcuExecutionFault,PcuExecutionFaultKind};

type NativeResult<T> = Result<T, Box<dyn Error>>;

#[derive(Debug, Default, Clone, Copy)]
#[allow(dead_code)] // Stage fields are read by Reject benchmarks; Clamp uses the same owner without stage timing.
pub struct Timings {
    pub upload: Duration,
    pub submission: Duration,
    pub completion: Duration,
    pub diagnostic_publication: Duration,
    pub wall: Duration,
}

struct Storage {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u8,
    allocation: u64,
    memory_type: u32,
    flags: u32,
}

/// An independent logical device, three retained buffers with aliased readonly descriptors and one terminal fence.
pub struct NativeOperand {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    buffers: Vec<Storage>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptors: vk::DescriptorPool,
    commands: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    extent: usize,
    byte_len: usize,
    pending: bool,
}

impl NativeOperand {
    /// Compiles the independent GLSL oracle and builds native Vulkan objects cold.
    ///
    /// # Errors
    /// Returns shader-tool, UUID/device, feature, allocation or native API failures.
    #[allow(clippy::too_many_lines)] // Complete RAII cold construction stays in one owner.
    pub fn new(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        operation: u32,
        policy: u32,
        scalar: pcu_facade::PcuScalarType,
        broadcast: bool,
    ) -> NativeResult<Self> {
        let words = compile_shader(operation, policy, scalar, extent, broadcast)?;
        // SAFETY: ash loads the system Vulkan library and the owner retains it through device Drop.
        let entry = unsafe { ash::Entry::load()? };
        let app = vk::ApplicationInfo::default()
            .application_name(c"pcu-native-binary-control")
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
        // SAFETY: Selected physical device and compute queue belong to this instance; Float64 was queried.
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
            buffers: Vec::new(),
            shader: vk::ShaderModule::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptors: vk::DescriptorPool::null(),
            commands: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            extent: extent as usize,
            byte_len: extent as usize * usize::from(scalar.bit_width() / 8),
            pending: false,
        };
        for bytes in [
            owner.byte_len.div_ceil(4) * 4,
            owner.byte_len.div_ceil(4) * 4,
            owner.extent * 4,
        ] {
            owner.buffers.push(create_buffer(
                &owner.instance,
                physical,
                &owner.device,
                bytes,
            )?);
        }
        let layouts: Vec<_> = (0..4)
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
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&words), None)?;
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
                .descriptor_count(4)];
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
            let infos: Vec<_> = [0, 0, 1, 2]
                .into_iter()
                .map(|index| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(owner.buffers[index].buffer)
                        .range(if index == 2 {
                            owner.extent as u64 * 4
                        } else {
                            owner.byte_len.div_ceil(4) as u64 * 4
                        })]
                })
                .collect();
            let writes: Vec<_> = infos
                .iter()
                .enumerate()
                .map(|(binding, info)| {
                    vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(u32::try_from(binding).expect("four bindings"))
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(info)
                })
                .collect();
            owner.device.update_descriptor_sets(&writes, &[]);
            owner.commands = owner.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(family),
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
                extent
                    .div_ceil(if scalar.bit_width() < 32 {
                        32 / u32::from(scalar.bit_width())
                    } else {
                        1
                    })
                    .div_ceil(64),
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

    /// Runs one retained input upload, complete terminal status scan and transactional output publication.
    /// # Errors
    /// Returns native/preflight failures; arithmetic notices retain precise lane/kind without heap formatting.
    pub fn call(
        &mut self,
        input: &[u8],
        output: &mut [u8],
    ) -> NativeResult<Option<PcuExecutionFault>> {
        self.call_policy(input, output, None, false)
    }

    fn call_policy(
        &mut self,
        left: &[u8],
        output: &mut [u8],
        timings: Option<&mut Timings>,
        recover_range: bool,
    ) -> NativeResult<Option<PcuExecutionFault>> {
        if self.pending {
            return Err("native session has unresolved work".into());
        }
        if left.len() < self.byte_len || output.len() < self.byte_len {
            return Err("native argument extent".into());
        }
        let started = timings.as_ref().map(|_| Instant::now());
        // SAFETY: Persistent coherent mapping is disjoint from caller RAM and prior work is terminal.
        unsafe {
            ptr::copy_nonoverlapping(left.as_ptr(), self.buffers[0].mapped, self.byte_len);
        }
        let uploaded = started.map(|_| Instant::now());
        // SAFETY: Previous synchronous call completed and owner retains all handles.
        unsafe {
            self.device.reset_fences(&[self.fence])?;
        }
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        // SAFETY: Recorded command uses owner resources; queue submissions are serial here.
        self.pending = true;
        unsafe {
            self.device.queue_submit(self.queue, &submits, self.fence)?;
        }
        let submitted = started.map(|_| Instant::now());
        // SAFETY: Fence belongs to this one outstanding submission; infinite wait proves terminal completion.
        unsafe {
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        self.pending = false;
        let completed = started.map(|_| Instant::now());
        let mut recovered = None;
        for invocation in 0..self.extent {
            // SAFETY: Compute-to-host barrier/fence precede coherent reads of initialized u32 statuses.
            let status = u32::from_ne_bytes(unsafe {
                self.buffers[2]
                    .mapped
                    .add(invocation * 4)
                    .cast::<[u8; 4]>()
                    .read()
            });
            if recover_range && matches!(status, 2 | 3) {
                recovered.get_or_insert(PcuExecutionFault {
                    recovered: true,
                    invocation_id: invocation as u64,
                    kind: if status == 2 {
                        PcuExecutionFaultKind::ArithmeticUnderflow
                    } else {
                        PcuExecutionFaultKind::ArithmeticOverflow
                    },
                });
            } else if status != 0 {
                return Ok(Some(PcuExecutionFault {
                    recovered: false,
                    invocation_id: invocation as u64,
                    kind: match status {
                        1 => PcuExecutionFaultKind::InvalidFloatingOperand,
                        2 => PcuExecutionFaultKind::ArithmeticUnderflow,
                        3 => PcuExecutionFaultKind::ArithmeticOverflow,
                        4 => PcuExecutionFaultKind::DivideByZero,
                        _ => return Err("native invalid status".into()),
                    },
                }));
            }
        }
        // SAFETY: All diagnostics succeeded, mapped output is terminal and caller output is exclusive/disjoint.
        unsafe {
            ptr::copy_nonoverlapping(self.buffers[1].mapped, output.as_mut_ptr(), self.byte_len);
        }
        if let (Some(timings), Some(started), Some(uploaded), Some(submitted), Some(completed)) =
            (timings, started, uploaded, submitted, completed)
        {
            let end = Instant::now();
            *timings = Timings {
                upload: uploaded - started,
                submission: submitted - uploaded,
                completion: completed - submitted,
                diagnostic_publication: end - completed,
                wall: end - started,
            };
        }
        Ok(recovered)
    }
}

impl Drop for NativeOperand {
    fn drop(&mut self) {
        // SAFETY: Successfully completed calls are terminal. A failed pending call requires idle proof;
        // absent proof, abandon this independent session rather than freeing possibly used resources.
        unsafe {
            if self.pending && self.device.device_wait_idle().is_err() {
                mem::forget(self.entry.clone());
                return;
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

fn compile_shader(
    operation: u32,
    policy: u32,
    scalar: pcu_facade::PcuScalarType,
    extent: u32,
    broadcast: bool,
) -> NativeResult<Vec<u32>> {
    if operation > 3 || policy > 2 {
        return Err("native unsupported operation/policy".into());
    }
    let (source, format) = match scalar {
        pcu_facade::PcuScalarType::F32 => (
            include_str!("../../../../spirv/checked_binary/shader/checked_binary.comp"),
            0,
        ),
        pcu_facade::PcuScalarType::F64 => (
            include_str!("../../../../spirv/checked_binary/shader/checked_binary_f64.comp"),
            0,
        ),
        pcu_facade::PcuScalarType::F16 => (
            include_str!("../../../../spirv/checked_binary/shader/checked_binary_low.comp"),
            0,
        ),
        pcu_facade::PcuScalarType::BF16 => (
            include_str!("../../../../spirv/checked_binary/shader/checked_binary_low.comp"),
            1,
        ),
        pcu_facade::PcuScalarType::F8E4M3FN => (
            include_str!("../../../../spirv/checked_binary/shader/checked_binary_low.comp"),
            2,
        ),
        pcu_facade::PcuScalarType::F8E5M2 => (
            include_str!("../../../../spirv/checked_binary/shader/checked_binary_low.comp"),
            3,
        ),
        _ => return Err("native unsupported scalar".into()),
    };
    let source = source
        .replace(
            "const uint OP = 0;",
            &format!("const uint OP = {operation};"),
        )
        .replace(
            "const uint POLICY = 0;",
            &format!("const uint POLICY = {policy};"),
        )
        .replace(
            "const uint RIGHT_SOURCE = 1;",
            "const uint RIGHT_SOURCE = 0;",
        )
        .replace(
            "const uint SOURCE1_BROADCAST = 0;",
            &format!("const uint SOURCE1_BROADCAST = {};", u32::from(broadcast)),
        )
        .replace(
            "const uint FORMAT = 0;",
            &format!("const uint FORMAT = {format};"),
        )
        .replace(
            "const uint EXTENT = 1;",
            &format!("const uint EXTENT = {extent};"),
        );
    let base = std::env::temp_dir().join(format!(
        "pcu-native-operand-{}-{operation}-{policy}",
        std::process::id()
    ));
    let input = base.with_extension("comp");
    let output = base.with_extension("spv");
    std::fs::write(&input, source)?;
    let result = std::process::Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.0", "-Os", "-S", "comp"])
        .arg(&input)
        .arg("-o")
        .arg(&output)
        .output()?;
    std::fs::remove_file(input)?;
    if !result.status.success() {
        return Err(format!(
            "native GLSL compiler: {} {}",
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
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
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
        ptr::write_bytes(mapped, 0, bytes);
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

#[derive(Default)]
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

pub const fn bytes<T: pcu_facade::PcuCheckedFloat>(values: &[T]) -> &[u8] {
    // SAFETY: The six sealed checked float carriers have no padding; a read-only byte view has identical lifetime and valid extent.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}
pub const fn bytes_mut<T: pcu_facade::PcuCheckedFloat>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: Every representation bit pattern of the sealed checked float carriers is valid; exclusive bytes retain the original slice lifetime.
    unsafe {
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), core::mem::size_of_val(values))
    }
}
