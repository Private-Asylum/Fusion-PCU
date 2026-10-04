//! Independent ash/GLSL lifecycle for a fixed two-step checked composed expression.
//! Floating-bit arithmetic helpers are shared and disclosed; integer arithmetic
//! is independently handwritten. No PCU lowering,
//! preparation, executor or submission routine is called by this control.

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
use pcu_facade::{
    PcuCheckedScalarFaultLaw,
    PcuDispatchFloatBinaryOp,
    PcuDispatchIntegerBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
    PcuScalarType,
    PcuStableDeviceIdentity,
};

#[cfg(feature = "insights")]
#[path = "api_census/api_census.rs"]
mod api_census;
#[cfg(feature = "insights")]
#[allow(unused_imports)] // Explicit API-census harness attaches the handwritten owner.
pub use api_census::with_scope as with_api_census_scope;
#[cfg(feature = "insights")]
macro_rules! sdk_call {
    ($point:ident, $call:expr) => {{
        api_census::count_current(fusion_pcu_vulkan::PcuVulkanApiPoint::$point);
        $call
    }};
}
#[cfg(not(feature = "insights"))]
macro_rules! sdk_call {
    ($point:ident, $call:expr) => {
        $call
    };
}
#[cfg(feature = "insights")]
macro_rules! owned_sdk_call {
    ($owner:expr, $point:ident, $call:expr) => {{
        api_census::count(
            $owner.api_census.as_deref(),
            fusion_pcu_vulkan::PcuVulkanApiPoint::$point,
        );
        $call
    }};
}
#[cfg(not(feature = "insights"))]
macro_rules! owned_sdk_call {
    ($owner:expr, $point:ident, $call:expr) => {
        $call
    };
}

type NativeResult<T> = Result<T, Box<dyn Error>>;
#[path = "compile/compile.rs"]
mod compile;

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
pub struct NativeComposed {
    #[cfg(feature = "insights")]
    api_census: Option<std::rc::Rc<fusion_pcu_vulkan::PcuVulkanApiInsights>>,
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
    laws: [Option<PcuCheckedScalarFaultLaw>; 2],
    // Exact cold parameters used to emit GLSL; permission headers have no native analogue.
    policy: (PcuScalarType, PcuFloatUnderflowPolicy, PcuRangePolicy),
}

impl NativeComposed {
    /// Verifies the actual GLSL scalar and lexical policies before accepting a comparison.
    pub fn assert_policy(
        &self,
        scalar: PcuScalarType,
        underflow: PcuFloatUnderflowPolicy,
        range: PcuRangePolicy,
    ) {
        assert_eq!(self.policy, (scalar, underflow, range));
    }
    /// Compiles the independent GLSL oracle and builds native Vulkan objects cold.
    ///
    /// # Errors
    /// Returns shader-tool, UUID/device, feature, allocation or native API failures.
    #[allow(clippy::too_many_lines)] // Complete RAII cold construction stays in one owner.
    pub fn new(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        policy: u32,
        range: u32,
        scalar: PcuScalarType,
    ) -> NativeResult<Self> {
        Self::build(identity, extent, policy, range, scalar, false)
    }

    /// Independent checked Add whose result is discarded before an input reload.
    ///
    /// # Errors
    /// Returns cold compiler, resource and native device failures.
    #[allow(dead_code)] // The integer companion reuses this owner but has no one-effect case.
    pub fn one_effect(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        policy: u32,
        range: u32,
        scalar: PcuScalarType,
    ) -> NativeResult<Self> {
        Self::build(identity, extent, policy, range, scalar, true)
    }

    #[allow(clippy::too_many_lines)] // One retained cold RAII owner, identical native objects for both fixed shaders.
    fn build(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        policy: u32,
        range: u32,
        scalar: PcuScalarType,
        one_effect: bool,
    ) -> NativeResult<Self> {
        let words = compile::shader(policy, range, scalar, extent, one_effect)?;
        let underflow = match policy {
            0 => PcuFloatUnderflowPolicy::IeeeAfterRounding,
            1 => PcuFloatUnderflowPolicy::RejectSubnormalResult,
            2 => PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            _ => return Err("native underflow policy".into()),
        };
        let range = match range {
            0 => PcuRangePolicy::Reject,
            1 => PcuRangePolicy::Clamp,
            _ => return Err("native range policy".into()),
        };
        let laws = if matches!(
            scalar,
            PcuScalarType::U8
                | PcuScalarType::I8
                | PcuScalarType::U16
                | PcuScalarType::I16
                | PcuScalarType::U32
                | PcuScalarType::I32
                | PcuScalarType::U64
                | PcuScalarType::I64
                | PcuScalarType::U128
                | PcuScalarType::I128
                | PcuScalarType::U256
                | PcuScalarType::I256
                | PcuScalarType::U512
                | PcuScalarType::I512
        ) {
            [
                PcuDispatchIntegerBinaryOp::Add,
                PcuDispatchIntegerBinaryOp::Mul,
            ]
            .map(|op| PcuCheckedScalarFaultLaw::integer_binary(scalar, op, range))
        } else {
            [PcuDispatchFloatBinaryOp::Add, PcuDispatchFloatBinaryOp::Mul]
                .map(|op| PcuCheckedScalarFaultLaw::float_binary(scalar, op, range, underflow))
        };
        let [Some(add), Some(mul)] = laws else {
            return Err("native scalar law".into());
        };
        // SAFETY: ash loads the system Vulkan library and the owner retains it through device Drop.
        let entry = unsafe { sdk_call!(LoaderLoad, ash::Entry::load())? };
        let app = vk::ApplicationInfo::default()
            .application_name(c"pcu-native-composed-control")
            .api_version(vk::API_VERSION_1_1);
        // SAFETY: The stack create info is valid and no extension pointers are supplied.
        let instance = unsafe {
            sdk_call!(
                CreateInstance,
                entry.create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app),
                    None,
                )
            )?
        };
        let selection = select_device(&instance, identity, false);
        let (physical, family) = match selection {
            Ok(selected) => selected,
            Err(error) => {
                // SAFETY: No logical device or native resource has been created.
                unsafe {
                    sdk_call!(DestroyInstance, instance.destroy_instance(None));
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
            sdk_call!(
                CreateDevice,
                instance.create_device(
                    physical,
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&queues)
                        .enabled_features(&features),
                    None,
                )
            )
        } {
            Ok(device) => device,
            Err(error) => {
                // SAFETY: Failed logical-device creation leaves this instance with no child owners.
                unsafe {
                    sdk_call!(DestroyInstance, instance.destroy_instance(None));
                }
                return Err(error.into());
            }
        };
        // SAFETY: Queue zero was requested for this selected compute family.
        let queue = unsafe { sdk_call!(GetDeviceQueue, device.get_device_queue(family, 0)) };
        let mut owner = Self {
            #[cfg(feature = "insights")]
            api_census: api_census::current(),
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
            laws: [Some(add), if one_effect { None } else { Some(mul) }],
            policy: (scalar, underflow, range),
        };
        for bytes in [
            owner.byte_len.div_ceil(4) * 4,
            owner.byte_len.div_ceil(4) * 4,
            owner.extent * 8,
        ] {
            owner.buffers.push(create_buffer(
                &owner.instance,
                physical,
                &owner.device,
                bytes,
            )?);
        }
        let layouts: Vec<_> = (0..5)
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
            owner.shader = owned_sdk_call!(
                owner,
                CreateShaderModule,
                owner.device.create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&words),
                    None
                )
            )?;
            owner.descriptor_layout = owned_sdk_call!(
                owner,
                CreateDescriptorSetLayout,
                owner.device.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&layouts),
                    None,
                )
            )?;
            let sets = [owner.descriptor_layout];
            owner.pipeline_layout = owned_sdk_call!(
                owner,
                CreatePipelineLayout,
                owner.device.create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default().set_layouts(&sets),
                    None,
                )
            )?;
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(owner.shader)
                .name(c"main");
            owner.pipeline = match owned_sdk_call!(
                owner,
                CreateComputePipelines,
                owner.device.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(owner.pipeline_layout)],
                    None,
                )
            ) {
                Ok(pipelines) => pipelines[0],
                Err((partial, error)) => {
                    for pipeline in partial {
                        owned_sdk_call!(
                            owner,
                            DestroyPipeline,
                            owner.device.destroy_pipeline(pipeline, None)
                        );
                    }
                    return Err(error.into());
                }
            };
            let sizes = [vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(5)];
            owner.descriptors = owned_sdk_call!(
                owner,
                CreateDescriptorPool,
                owner.device.create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(&sizes),
                    None,
                )
            )?;
            let set = owned_sdk_call!(
                owner,
                AllocateDescriptorSets,
                owner.device.allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(owner.descriptors)
                        .set_layouts(&sets),
                )
            )?[0];
            let infos: Vec<_> = [1, 0, 0, 0, 2]
                .into_iter()
                .map(|index| {
                    [vk::DescriptorBufferInfo::default()
                        .buffer(owner.buffers[index].buffer)
                        .range(if index == 2 {
                            owner.extent as u64 * 8
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
            owned_sdk_call!(
                owner,
                UpdateDescriptorSets,
                owner.device.update_descriptor_sets(&writes, &[])
            );
            owner.commands = owned_sdk_call!(
                owner,
                CreateCommandPool,
                owner.device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default().queue_family_index(family),
                    None,
                )
            )?;
            owner.command = owned_sdk_call!(
                owner,
                AllocateCommandBuffers,
                owner.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(owner.commands)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            )?[0];
            owned_sdk_call!(
                owner,
                BeginCommandBuffer,
                owner
                    .device
                    .begin_command_buffer(owner.command, &vk::CommandBufferBeginInfo::default())
            )?;
            owned_sdk_call!(
                owner,
                BindPipeline,
                owner.device.cmd_bind_pipeline(
                    owner.command,
                    vk::PipelineBindPoint::COMPUTE,
                    owner.pipeline,
                )
            );
            owned_sdk_call!(
                owner,
                BindDescriptorSets,
                owner.device.cmd_bind_descriptor_sets(
                    owner.command,
                    vk::PipelineBindPoint::COMPUTE,
                    owner.pipeline_layout,
                    0,
                    &[set],
                    &[],
                )
            );
            owned_sdk_call!(
                owner,
                Dispatch,
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
                )
            );
            let barrier = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            owned_sdk_call!(
                owner,
                PipelineBarrier,
                owner.device.cmd_pipeline_barrier(
                    owner.command,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::HOST,
                    vk::DependencyFlags::empty(),
                    &barrier,
                    &[],
                    &[],
                )
            );
            owned_sdk_call!(
                owner,
                EndCommandBuffer,
                owner.device.end_command_buffer(owner.command)
            )?;
            owner.fence = owned_sdk_call!(
                owner,
                CreateFence,
                owner
                    .device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
            )?;
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
        self.call_policy(input, output, None)
    }

    fn call_policy(
        &mut self,
        left: &[u8],
        output: &mut [u8],
        timings: Option<&mut Timings>,
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
            owned_sdk_call!(self, ResetFences, self.device.reset_fences(&[self.fence]))?;
        }
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        // SAFETY: Recorded command uses owner resources; queue submissions are serial here.
        self.pending = true;
        unsafe {
            owned_sdk_call!(
                self,
                QueueSubmit,
                self.device.queue_submit(self.queue, &submits, self.fence)
            )?;
        }
        let submitted = started.map(|_| Instant::now());
        // SAFETY: Fence belongs to this one outstanding submission; infinite wait proves terminal completion.
        unsafe {
            owned_sdk_call!(
                self,
                WaitForFences,
                self.device.wait_for_fences(&[self.fence], true, u64::MAX)
            )?;
        }
        self.pending = false;
        let completed = started.map(|_| Instant::now());
        let mut recovered = None;
        let mut fatal = None;
        for invocation in 0..self.extent {
            // SAFETY: The complete compute-to-host dependency and terminal fence precede
            // the coherent read. Every logical lane initializes its eight-byte record.
            let [raw, step] = unsafe {
                self.buffers[2]
                    .mapped
                    .add(invocation * 8)
                    .cast::<[u32; 2]>()
                    .read_unaligned()
            };
            if raw == 0 {
                if step != 0 {
                    return Err("native zero status with nonzero step".into());
                }
                continue;
            }
            let kind = match raw & !0x100 {
                1 => PcuExecutionFaultKind::InvalidFloatingOperand,
                2 => PcuExecutionFaultKind::ArithmeticUnderflow,
                3 => PcuExecutionFaultKind::ArithmeticOverflow,
                4 => PcuExecutionFaultKind::DivideByZero,
                _ => return Err("native invalid status code".into()),
            };
            let law = match step {
                2 => self.laws[0],
                4 => self.laws[1],
                _ => return Err("native invalid composed step".into()),
            };
            let fault = PcuExecutionFault {
                recovered: raw & 0x100 != 0,
                invocation_id: u64::try_from(invocation)?,
                kind,
            };
            let law = law.ok_or("native nonexistent arithmetic step")?;
            if !law.accepts(fault, u64::try_from(self.extent)?) {
                return Err("native status violates actual operation/extent law".into());
            }
            if fault.recovered {
                recovered.get_or_insert(fault);
            } else {
                fatal.get_or_insert(fault);
            }
        }
        if fatal.is_some() {
            return Ok(fatal);
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

impl Drop for NativeComposed {
    fn drop(&mut self) {
        // SAFETY: Successfully completed calls are terminal. A failed pending call requires idle proof;
        // absent proof, abandon this independent session rather than freeing possibly used resources.
        unsafe {
            if self.pending
                && owned_sdk_call!(self, DeviceWaitIdle, self.device.device_wait_idle()).is_err()
            {
                mem::forget(self.entry.clone());
                return;
            }
            owned_sdk_call!(
                self,
                DestroyFence,
                self.device.destroy_fence(self.fence, None)
            );
            owned_sdk_call!(
                self,
                DestroyCommandPool,
                self.device.destroy_command_pool(self.commands, None)
            );
            owned_sdk_call!(
                self,
                DestroyDescriptorPool,
                self.device.destroy_descriptor_pool(self.descriptors, None)
            );
            owned_sdk_call!(
                self,
                DestroyPipeline,
                self.device.destroy_pipeline(self.pipeline, None)
            );
            owned_sdk_call!(
                self,
                DestroyPipelineLayout,
                self.device
                    .destroy_pipeline_layout(self.pipeline_layout, None)
            );
            owned_sdk_call!(
                self,
                DestroyDescriptorSetLayout,
                self.device
                    .destroy_descriptor_set_layout(self.descriptor_layout, None)
            );
            owned_sdk_call!(
                self,
                DestroyShaderModule,
                self.device.destroy_shader_module(self.shader, None)
            );
            for buffer in &self.buffers {
                owned_sdk_call!(self, UnmapMemory, self.device.unmap_memory(buffer.memory));
                owned_sdk_call!(
                    self,
                    DestroyBuffer,
                    self.device.destroy_buffer(buffer.buffer, None)
                );
                owned_sdk_call!(
                    self,
                    FreeMemory,
                    self.device.free_memory(buffer.memory, None)
                );
            }
            owned_sdk_call!(self, DestroyDevice, self.device.destroy_device(None));
            owned_sdk_call!(self, DestroyInstance, self.instance.destroy_instance(None));
        }
    }
}

fn select_device(
    instance: &ash::Instance,
    identity: PcuStableDeviceIdentity,
    f64: bool,
) -> NativeResult<(vk::PhysicalDevice, u32)> {
    // SAFETY: All inventory queries use a live instance and returned physical handles.
    unsafe {
        for physical in sdk_call!(
            EnumeratePhysicalDevices,
            instance.enumerate_physical_devices()
        )? {
            let properties = sdk_call!(
                PhysicalDeviceProperties,
                instance.get_physical_device_properties(physical)
            );
            if properties.device_type == vk::PhysicalDeviceType::CPU {
                continue;
            }
            let mut id = vk::PhysicalDeviceIDProperties::default();
            sdk_call!(
                PhysicalDeviceProperties,
                instance.get_physical_device_properties2(
                    physical,
                    &mut vk::PhysicalDeviceProperties2::default().push_next(&mut id),
                )
            );
            if identity.namespace() != "vulkan.deviceUUID" || identity.value() != id.device_uuid {
                continue;
            }
            if f64
                && sdk_call!(
                    PhysicalDeviceFeatures,
                    instance.get_physical_device_features(physical)
                )
                .shader_float64
                    == 0
            {
                return Err("native shaderFloat64 unavailable".into());
            }
            for (family, queue) in sdk_call!(
                QueueFamilyProperties,
                instance.get_physical_device_queue_family_properties(physical)
            )
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
        let buffer = sdk_call!(
            CreateBuffer,
            device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(bytes as u64)
                    .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
        )?;
        let requirements = sdk_call!(
            BufferMemoryRequirements,
            device.get_buffer_memory_requirements(buffer)
        );
        let properties = sdk_call!(
            MemoryProperties,
            instance.get_physical_device_memory_properties(physical)
        );
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
            sdk_call!(DestroyBuffer, device.destroy_buffer(buffer, None));
            return Err("native coherent memory unavailable".into());
        };
        let memory = match sdk_call!(
            AllocateMemory,
            device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        ) {
            Ok(memory) => memory,
            Err(error) => {
                sdk_call!(DestroyBuffer, device.destroy_buffer(buffer, None));
                return Err(error.into());
            }
        };
        let mapped = match sdk_call!(
            BindBufferMemory,
            device.bind_buffer_memory(buffer, memory, 0)
        )
        .and_then(|()| {
            sdk_call!(
                MapMemory,
                device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
            )
        }) {
            Ok(mapped) => mapped.cast(),
            Err(error) => {
                sdk_call!(DestroyBuffer, device.destroy_buffer(buffer, None));
                sdk_call!(FreeMemory, device.free_memory(memory, None));
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

pub const fn bytes<T: pcu_facade::PcuScalar>(values: &[T]) -> &[u8] {
    // SAFETY: The sealed scalar carriers have no padding; a read-only byte view has identical lifetime and valid extent.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}
pub const fn bytes_mut<T: pcu_facade::PcuScalar>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: Every representation bit pattern of the sealed checked float carriers is valid; exclusive bytes retain the original slice lifetime.
    unsafe {
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), core::mem::size_of_val(values))
    }
}
