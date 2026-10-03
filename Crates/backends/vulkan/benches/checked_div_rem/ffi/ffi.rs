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

#[path = "resident/resident.rs"]
#[allow(dead_code)]
// Host-only benchmark targets share this owner; the resident cohort executes this child.
pub mod resident;

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

/// An independent logical device, four/five retained physical buffers and one terminal fence.
pub struct NativeDivRem {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    queue: vk::Queue,
    #[allow(dead_code)] // Used by the separately selected cold resident bank constructor.
    physical: vk::PhysicalDevice,
    #[allow(dead_code)] // Used by the separately selected cold resident command pool.
    queue_family: u32,
    buffers: Vec<Storage>,
    shader: vk::ShaderModule,
    descriptor_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptors: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    commands: vk::CommandPool,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    extent: usize,
    byte_len: usize,
    storage_len: usize,
    input_count: usize,
    input_bytes: [usize; 2],
    pending: bool,
}

impl NativeDivRem {
    /// Compiles the independent GLSL oracle and builds native Vulkan objects cold.
    ///
    /// # Errors
    /// Returns shader-tool, UUID/device, feature, allocation or native API failures.
    #[allow(clippy::too_many_lines)] // Complete RAII cold construction stays in one owner.
    pub fn new(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        signed: bool,
        element_bytes: usize,
    ) -> NativeResult<Self> {
        let words = compile_shader(signed, element_bytes, extent)?;
        Self::from_words(identity, extent, element_bytes, &words)
    }
    /// Compiles a separate source-role shader; repeated reads alias one physical input.
    /// # Errors
    /// Rejects unknown roles, invalid dimensions and native compiler/device failures.
    #[allow(dead_code)] // Only the separately qualified operand-role cohort uses this constructor.
    pub fn new_roles(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        signed: bool,
        element_bytes: usize,
        kind: usize,
    ) -> NativeResult<Self> {
        if kind > 5 {
            return Err("native role schema unsupported".into());
        }
        let count = if kind == 2 || kind == 5 { 2 } else { 1 };
        let spans = if kind == 5 { [1, 1] } else { [extent, extent] };
        let words = compile_role_shader(signed, element_bytes, extent, kind)?;
        Self::from_layout(identity, extent, element_bytes, &words, count, spans)
    }
    /// Creates a diagnostic-only owner for already validated provider words.
    /// Unlike `new()`, this seam is not an independent compiler or arithmetic control.
    /// # Errors
    /// Returns host/schema, UUID/device, allocation or native API failures.
    #[allow(clippy::too_many_lines)] // Complete cold RAII native ownership remains adjacent.
    pub fn from_words(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        element_bytes: usize,
        words: &[u32],
    ) -> NativeResult<Self> {
        Self::from_layout(identity, extent, element_bytes, words, 2, [extent; 2])
    }
    #[allow(clippy::too_many_lines)] // Complete cold RAII ownership; warm paths never reconstruct it.
    fn from_layout(
        identity: PcuStableDeviceIdentity,
        extent: u32,
        element_bytes: usize,
        words: &[u32],
        input_count: usize,
        input_extents: [u32; 2],
    ) -> NativeResult<Self> {
        if cfg!(target_endian = "big")
            || extent == 0
            || ![1, 2, 4, 8, 16, 32, 64].contains(&element_bytes)
        {
            return Err("native narrow host/schema unsupported".into());
        }
        let byte_len = usize::try_from(extent)?
            .checked_mul(element_bytes)
            .ok_or("native logical extent overflow")?;
        let storage_len = byte_len
            .checked_add(3)
            .map(|n| n & !3)
            .ok_or("native padded extent overflow")?;
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
            queue_family: family,
            buffers: Vec::new(),
            shader: vk::ShaderModule::null(),
            descriptor_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            descriptors: vk::DescriptorPool::null(),
            descriptor_set: vk::DescriptorSet::null(),
            commands: vk::CommandPool::null(),
            command: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            extent: extent as usize,
            byte_len,
            storage_len,
            input_count,
            input_bytes: [
                usize::try_from(input_extents[0])?
                    .checked_mul(element_bytes)
                    .ok_or("native input size")?,
                usize::try_from(input_extents[1])?
                    .checked_mul(element_bytes)
                    .ok_or("native input size")?,
            ],
            pending: false,
        };
        for id in 0..input_count + 3 {
            let logical = if id < input_count {
                owner.input_bytes[id]
            } else if id == input_count + 2 {
                owner.extent * 4
            } else {
                owner.byte_len
            };
            let bytes = logical
                .checked_add(3)
                .map(|n| n & !3)
                .ok_or("native padded input")?;
            owner.buffers.push(create_buffer(
                &owner.instance,
                physical,
                &owner.device,
                bytes,
            )?);
            // SAFETY: Complete unique coherent cold mapping; zero packed padding before submission.
            unsafe {
                ptr::write_bytes(owner.buffers[id].mapped, 0, bytes);
            }
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
                .descriptor_count(5)];
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
            let mapping = if input_count == 1 {
                [0, 0, 1, 2, 3]
            } else {
                [0, 1, 2, 3, 4]
            };
            let infos: Vec<_> = mapping
                .into_iter()
                .map(|id| {
                    let size = if id < input_count {
                        (owner.input_bytes[id] + 3) & !3
                    } else if id == input_count + 2 {
                        owner.extent * 4
                    } else {
                        owner.storage_len
                    };
                    [vk::DescriptorBufferInfo::default()
                        .buffer(owner.buffers[id].buffer)
                        .range(size as u64)]
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
                    .div_ceil(u32::try_from((4 / element_bytes).max(1))?)
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

    /// Runs the full matched host boundary, including dense status and transactional publication.
    ///
    /// # Errors
    /// Returns invalid extent, numerical status, submission or terminal wait failure.
    #[allow(dead_code)] // Reject and Clamp benchmark binaries share the independent native owner.
    pub fn call(
        &mut self,
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        remainder: &mut [u8],
        timings: Option<&mut Timings>,
    ) -> NativeResult<Option<PcuExecutionFault>> {
        self.call_policy(left, right, output, remainder, timings)
    }

    /// Test-only private arithmetic/status observation; no public PCU publication rule is bypassed.
    ///
    /// # Errors
    /// Returns extent, native submission or terminal completion failures.
    #[allow(dead_code)] // Complete encoding fixtures use this; semantic peers use transactional calls.
    pub fn diagnostics(
        &mut self,
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        remainder: &mut [u8],
        statuses: &mut [u32],
    ) -> NativeResult<()> {
        if self.pending
            || left.len() < self.input_bytes[0]
            || (self.input_count == 2 && right.len() < self.input_bytes[1])
            || output.len() < self.byte_len
            || remainder.len() < self.byte_len
            || statuses.len() < self.extent
        {
            return Err("native diagnostic extent/session".into());
        }
        // SAFETY: Complete coherent mappings are owned, terminal and disjoint from caller slices.
        unsafe {
            ptr::copy_nonoverlapping(left.as_ptr(), self.buffers[0].mapped, self.input_bytes[0]);
            if self.input_count == 2 {
                ptr::copy_nonoverlapping(
                    right.as_ptr(),
                    self.buffers[1].mapped,
                    self.input_bytes[1],
                );
            }
            self.device.reset_fences(&[self.fence])?;
        }
        let commands = [self.command];
        let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
        self.pending = true;
        // SAFETY: One retained command owns valid descriptors; this owner serializes its queue.
        unsafe {
            self.device.queue_submit(self.queue, &submits, self.fence)?;
            self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        }
        self.pending = false;
        // SAFETY: Terminal fence+compute-to-host barrier precede all initialized private output
        // and status reads. Diagnostic outputs are separate test-owned buffers, not PCU caller outputs.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffers[self.input_count].mapped,
                output.as_mut_ptr(),
                self.byte_len,
            );
            ptr::copy_nonoverlapping(
                self.buffers[self.input_count + 1].mapped,
                remainder.as_mut_ptr(),
                self.byte_len,
            );
            ptr::copy_nonoverlapping(
                self.buffers[self.input_count + 2].mapped,
                statuses.as_mut_ptr().cast::<u8>(),
                self.extent * 4,
            );
        }
        Ok(())
    }

    fn call_policy(
        &mut self,
        left: &[u8],
        right: &[u8],
        output: &mut [u8],
        remainder: &mut [u8],
        timings: Option<&mut Timings>,
    ) -> NativeResult<Option<PcuExecutionFault>> {
        if self.pending {
            return Err("native session has unresolved work".into());
        }
        if left.len() < self.input_bytes[0]
            || (self.input_count == 2 && right.len() < self.input_bytes[1])
            || output.len() < self.byte_len
            || remainder.len() < self.byte_len
        {
            return Err("native argument extent".into());
        }
        let started = timings.as_ref().map(|_| Instant::now());
        // SAFETY: Persistent coherent mapping is disjoint from caller RAM and prior work is terminal.
        unsafe {
            ptr::copy_nonoverlapping(left.as_ptr(), self.buffers[0].mapped, self.input_bytes[0]);
            if self.input_count == 2 {
                ptr::copy_nonoverlapping(
                    right.as_ptr(),
                    self.buffers[1].mapped,
                    self.input_bytes[1],
                );
            }
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
        for invocation in 0..self.extent {
            // SAFETY: Compute-to-host barrier/fence precede coherent reads of initialized u32 statuses.
            let status = u32::from_ne_bytes(unsafe {
                self.buffers[self.input_count + 2]
                    .mapped
                    .add(invocation * 4)
                    .cast::<[u8; 4]>()
                    .read()
            });
            if status != 0 {
                let kind = match status {
                    4 => PcuExecutionFaultKind::DivideByZero,
                    5 => PcuExecutionFaultKind::SignedDivisionOverflow,
                    _ => return Err("unknown native DivRem status".into()),
                };
                return Ok(Some(PcuExecutionFault {
                    kind,
                    invocation_id: invocation as u64,
                    recovered: false,
                }));
            }
        }
        // SAFETY: All diagnostics succeeded, mapped output is terminal and caller output is exclusive/disjoint.
        unsafe {
            ptr::copy_nonoverlapping(
                self.buffers[self.input_count].mapped,
                output.as_mut_ptr(),
                self.byte_len,
            );
            ptr::copy_nonoverlapping(
                self.buffers[self.input_count + 1].mapped,
                remainder.as_mut_ptr(),
                self.byte_len,
            );
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
        Ok(None)
    }
}

impl Drop for NativeDivRem {
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

fn compile_shader(signed: bool, element_bytes: usize, extent: u32) -> NativeResult<Vec<u32>> {
    if ![1, 2, 4, 8, 16, 32, 64].contains(&element_bytes) {
        return Err("native unsupported integer schema".into());
    }
    let source = if element_bytes > 8 {
        include_str!("../../../../spirv/checked_div_rem/shader/wide_div_rem.comp")
    } else {
        include_str!("../../../../spirv/checked_div_rem/shader/checked_div_rem.comp")
    }
    .replace(
        "const uint SIGNED=0u;",
        &format!("const uint SIGNED={}u;", u32::from(signed)),
    )
    .replace(
        if element_bytes > 8 {
            "const uint ELEMENT_BYTES=16u;"
        } else {
            "const uint ELEMENT_BYTES=4u;"
        },
        &format!("const uint ELEMENT_BYTES={element_bytes}u;"),
    )
    .replace(
        "const uint EXTENT=1u;",
        &format!("const uint EXTENT={extent}u;"),
    );
    compile_glsl(&source, element_bytes, signed)
}
fn compile_role_shader(
    signed: bool,
    element_bytes: usize,
    extent: u32,
    kind: usize,
) -> NativeResult<Vec<u32>> {
    let source = if element_bytes > 8 {
        include_str!("../../../../spirv/checked_div_rem/shader/wide_operand_div_rem.comp")
    } else {
        include_str!("../../../../spirv/checked_div_rem/shader/operand_div_rem.comp")
    };
    let source = source
        .replace(
            "const uint SIGNED=0u;",
            &format!("const uint SIGNED={}u;", u32::from(signed)),
        )
        .replace(
            if element_bytes > 8 {
                "const uint ELEMENT_BYTES=16u;"
            } else {
                "const uint ELEMENT_BYTES=4u;"
            },
            &format!("const uint ELEMENT_BYTES={element_bytes}u;"),
        )
        .replace(
            "const uint EXTENT=1u;",
            &format!("const uint EXTENT={extent}u;"),
        )
        .replace(
            "const uint OPERAND_B=1u;",
            &format!(
                "const uint OPERAND_B={}u;",
                u32::from(kind == 2 || kind == 5)
            ),
        )
        .replace(
            "const uint OPERAND_A_ZERO=0u;",
            &format!(
                "const uint OPERAND_A_ZERO={}u;",
                u32::from(kind == 4 || kind == 5)
            ),
        )
        .replace(
            "const uint OPERAND_B_ZERO=0u;",
            &format!(
                "const uint OPERAND_B_ZERO={}u;",
                u32::from(kind == 3 || kind == 5)
            ),
        );
    compile_glsl(&source, element_bytes, signed)
}
fn compile_glsl(source: &str, element_bytes: usize, signed: bool) -> NativeResult<Vec<u32>> {
    let base = std::env::temp_dir().join(format!(
        "pcu-native-div-rem-{}-{element_bytes}-{signed}",
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
    // SAFETY: Sealed PcuScalar has no padding; a read-only byte view has identical lifetime and valid extent.
    unsafe { core::slice::from_raw_parts(values.as_ptr().cast(), core::mem::size_of_val(values)) }
}
pub const fn bytes_mut<T: pcu_facade::PcuScalar>(values: &mut [T]) -> &mut [u8] {
    // SAFETY: Every sealed PcuScalar bit pattern is valid; exclusive bytes retain the original slice lifetime.
    unsafe {
        core::slice::from_raw_parts_mut(values.as_mut_ptr().cast(), core::mem::size_of_val(values))
    }
}
