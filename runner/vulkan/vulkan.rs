//! Vulkan runner for PCU SPIR-V dispatch modules.
//!
//! This module is execution glue, not SPIR-V lowering. It owns Vulkan device selection,
//! runner capability detection, fixed-descriptor execution, and the runtime mess so callers do
//! not have to duplicate Vulkan reference graphs by hand.

use core::fmt;
use core::mem;
use core::ptr;
use std::error::Error;
use std::ffi::CStr;
use std::string::String;

use ash::vk;

const RUNNER_APP_NAME: &CStr = c"fusion-pcu-vulkan-runner";
const RUNNER_ENGINE_NAME: &CStr = c"fusion-pcu";
const SHADER_ENTRY_POINT: &CStr = c"main";

/// Abstract resource-addressing models that a runner may select for a resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuResourceAddressingModel {
    FixedDescriptors,
    DescriptorIndex,
    BufferDeviceAddress,
}

/// Descriptor class used for heap budgeting and registration failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuVulkanDescriptorClass {
    SampledImage,
    StorageBuffer,
    StorageImage,
    Sampler,
}

/// Conservative descriptor-heap budget requested by the runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuVulkanDescriptorHeapBudget {
    pub sampled_images: u32,
    pub storage_buffers: u32,
    pub storage_images: u32,
    pub samplers: u32,
}

impl PcuVulkanDescriptorHeapBudget {
    pub const PORTABLE_DEFAULT: Self = Self {
        sampled_images: 16 * 1024,
        storage_buffers: 8 * 1024,
        storage_images: 8 * 1024,
        samplers: 1024,
    };

    #[must_use]
    pub const fn portable_default() -> Self {
        Self::PORTABLE_DEFAULT
    }

    #[must_use]
    pub const fn empty() -> Self {
        Self {
            sampled_images: 0,
            storage_buffers: 0,
            storage_images: 0,
            samplers: 0,
        }
    }

    #[must_use]
    pub const fn clamp_to(self, limits: Self) -> Self {
        Self {
            sampled_images: min_u32(self.sampled_images, limits.sampled_images),
            storage_buffers: min_u32(self.storage_buffers, limits.storage_buffers),
            storage_images: min_u32(self.storage_images, limits.storage_images),
            samplers: min_u32(self.samplers, limits.samplers),
        }
    }
}

impl Default for PcuVulkanDescriptorHeapBudget {
    fn default() -> Self {
        Self::empty()
    }
}

/// Vulkan descriptor-indexing support relevant to PCU resource routing.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuVulkanDescriptorIndexingCaps {
    pub runtime_descriptor_array: bool,
    pub sampled_image_non_uniform_indexing: bool,
    pub storage_buffer_non_uniform_indexing: bool,
    pub storage_image_non_uniform_indexing: bool,
    pub partially_bound: bool,
    pub update_unused_while_pending: bool,
    pub variable_descriptor_count: bool,
    pub mutable_descriptor_type: bool,
    pub requested_heap_budget: PcuVulkanDescriptorHeapBudget,
    pub actual_heap_budget: PcuVulkanDescriptorHeapBudget,
}

impl PcuVulkanDescriptorIndexingCaps {
    #[must_use]
    pub const fn supports_storage_buffer_heap(self) -> bool {
        self.runtime_descriptor_array && self.storage_buffer_non_uniform_indexing
    }

    #[must_use]
    pub const fn supports_sampled_image_heap(self) -> bool {
        self.runtime_descriptor_array && self.sampled_image_non_uniform_indexing
    }

    #[must_use]
    pub const fn supports_storage_image_heap(self) -> bool {
        self.runtime_descriptor_array && self.storage_image_non_uniform_indexing
    }
}

/// Vulkan buffer-device-address support relevant to PCU buffer routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuVulkanBufferDeviceAddressCaps {
    pub supported: bool,
    pub capture_replay: bool,
    pub multi_device: bool,
}

/// Vulkan push-constant support surfaced to PCU invocation metadata routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct PcuVulkanPushConstantCaps {
    pub max_size_bytes: u32,
}

/// Runtime-selected Vulkan runner capability truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuVulkanRunnerCaps {
    pub api_version: u32,
    pub descriptor_indexing: PcuVulkanDescriptorIndexingCaps,
    pub buffer_device_address: PcuVulkanBufferDeviceAddressCaps,
    pub push_constants: PcuVulkanPushConstantCaps,
    pub selected_storage_buffer_model: PcuResourceAddressingModel,
}

/// Result metadata returned by the narrow current fixed-descriptor float dispatch path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PcuVulkanParallelF32Report {
    pub element_count: usize,
    pub dispatch_groups_x: u32,
    pub sample_output: f32,
    pub resource_model: PcuResourceAddressingModel,
}

/// Vulkan PCU runner.
pub struct PcuVulkanRunner {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    queue: vk::Queue,
    caps: PcuVulkanRunnerCaps,
    physical_device_name: String,
}

impl PcuVulkanRunner {
    /// Creates a Vulkan runner using the first compute-capable physical device.
    ///
    /// # Errors
    ///
    /// Returns loader, device-enumeration, queue-family, memory, or Vulkan platform failures.
    pub fn new() -> Result<Self, PcuVulkanError> {
        Self::with_descriptor_heap_budget(PcuVulkanDescriptorHeapBudget::portable_default())
    }

    /// Creates a Vulkan runner with a caller-specified descriptor-heap budget request.
    ///
    /// # Errors
    ///
    /// Returns loader, device-enumeration, queue-family, memory, or Vulkan platform failures.
    pub fn with_descriptor_heap_budget(
        requested_heap_budget: PcuVulkanDescriptorHeapBudget,
    ) -> Result<Self, PcuVulkanError> {
        let entry = unsafe {
            // SAFETY: Loading the Vulkan loader is process-local and ash validates entry points.
            ash::Entry::load()
        }
        .map_err(PcuVulkanError::Loader)?;
        let instance_api_version = choose_instance_api_version(&entry)?;
        let instance = create_instance(&entry, instance_api_version)?;
        let selected = select_compute_device(&instance)?;
        let caps = query_runner_caps(
            &instance,
            selected.physical_device,
            &selected.properties,
            instance_api_version,
            requested_heap_budget,
        )?;
        let device = create_device(
            &instance,
            selected.physical_device,
            selected.queue_family_index,
        )?;
        let queue = unsafe {
            // SAFETY: The logical device was created with queue index 0 for this queue family.
            device.get_device_queue(selected.queue_family_index, 0)
        };

        Ok(Self {
            _entry: entry,
            instance,
            device,
            physical_device: selected.physical_device,
            queue_family_index: selected.queue_family_index,
            queue,
            caps,
            physical_device_name: physical_device_name(&selected.properties),
        })
    }

    #[must_use]
    pub const fn caps(&self) -> PcuVulkanRunnerCaps {
        self.caps
    }

    #[must_use]
    pub fn physical_device_name(&self) -> &str {
        &self.physical_device_name
    }

    #[must_use]
    pub const fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    #[must_use]
    pub const fn physical_device_raw(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Executes the current narrow PCU-generated SPIR-V float map kernel.
    ///
    /// The kernel contract is:
    ///
    /// `output[i] = source[i] * 2.0 + bias[i] + 1.0`
    ///
    /// This is intentionally a runner-owned Tier 0 fixed-descriptor path. Descriptor-indexed and
    /// BDA execution require different SPIR-V lowering and are reported as caps, not faked here.
    ///
    /// # Errors
    ///
    /// Returns shape, memory, Vulkan, or output-transfer failures.
    pub fn run_parallel_f32_spirv(
        &self,
        spirv: &[u32],
        local_size_x: u32,
        source: &[f32],
        bias: &[f32],
        output: &mut [f32],
    ) -> Result<PcuVulkanParallelF32Report, PcuVulkanError> {
        if local_size_x == 0 || source.len() != bias.len() || source.len() != output.len() {
            return Err(PcuVulkanError::InvalidDispatchShape);
        }
        let element_count = source.len();
        let element_count_u32 =
            u32::try_from(element_count).map_err(|_| PcuVulkanError::BufferTooLarge)?;
        if !element_count_u32.is_multiple_of(local_size_x) {
            return Err(PcuVulkanError::InvalidDispatchShape);
        }
        let dispatch_groups_x = element_count_u32 / local_size_x;

        let source_buffer = VulkanBuffer::new_storage_f32_buffer(
            &self.instance,
            self.physical_device,
            &self.device,
            element_count,
        )?;
        let bias_buffer = VulkanBuffer::new_storage_f32_buffer(
            &self.instance,
            self.physical_device,
            &self.device,
            element_count,
        )?;
        let output_buffer = VulkanBuffer::new_storage_f32_buffer(
            &self.instance,
            self.physical_device,
            &self.device,
            element_count,
        )?;

        source_buffer.write_f32s(source)?;
        bias_buffer.write_f32s(bias)?;

        let shader_module = create_shader_module(&self.device, spirv)?;
        let descriptor_set_layout = create_descriptor_set_layout(&self.device)?;
        let pipeline_layout = create_pipeline_layout(&self.device, descriptor_set_layout.handle)?;
        let pipeline = create_compute_pipeline(
            &self.device,
            shader_module.handle,
            pipeline_layout.handle,
            SHADER_ENTRY_POINT,
        )?;
        let descriptor_pool = create_descriptor_pool(&self.device)?;
        let descriptor_set = allocate_descriptor_set(
            &self.device,
            descriptor_pool.handle,
            descriptor_set_layout.handle,
        )?;
        update_float_storage_descriptors(
            &self.device,
            descriptor_set,
            &source_buffer,
            &bias_buffer,
            &output_buffer,
        );

        let command_pool = create_command_pool(&self.device, self.queue_family_index)?;
        let command_buffer = allocate_command_buffer(&self.device, command_pool.handle)?;
        record_compute_commands(
            &self.device,
            command_buffer,
            pipeline.handle,
            pipeline_layout.handle,
            descriptor_set,
            dispatch_groups_x,
        )?;

        let fence = create_fence(&self.device)?;
        submit_and_wait(&self.device, self.queue, command_buffer, fence.handle)?;
        output_buffer.read_f32s(output)?;

        let sample_output = output
            .last()
            .copied()
            .ok_or(PcuVulkanError::InvalidDispatchShape)?;
        Ok(PcuVulkanParallelF32Report {
            element_count,
            dispatch_groups_x,
            sample_output,
            resource_model: self.caps.selected_storage_buffer_model,
        })
    }
}

impl Drop for PcuVulkanRunner {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: The runner owns the logical device and instance and destroys them exactly once.
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[derive(Debug)]
pub enum PcuVulkanError {
    Loader(ash::LoadingError),
    Vulkan {
        context: &'static str,
        result: vk::Result,
    },
    NoPhysicalDevice,
    NoComputeQueueFamily,
    NoHostVisibleCoherentMemory,
    NoDescriptorSet,
    NoCommandBuffer,
    NoComputePipeline,
    InvalidDispatchShape,
    BufferTooLarge,
    BufferTooSmall,
    DescriptorHeapFull {
        class: PcuVulkanDescriptorClass,
        capacity: u32,
    },
    UnsupportedResourceAddressing {
        requested: PcuResourceAddressingModel,
        available: PcuResourceAddressingModel,
    },
}

impl fmt::Display for PcuVulkanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Loader(error) => write!(formatter, "Vulkan loader error: {error}"),
            Self::Vulkan { context, result } => write!(formatter, "{context}: {result:?}"),
            Self::NoPhysicalDevice => formatter.write_str("no Vulkan physical device found"),
            Self::NoComputeQueueFamily => {
                formatter.write_str("no Vulkan compute-capable queue family found")
            }
            Self::NoHostVisibleCoherentMemory => {
                formatter.write_str("no host-visible coherent Vulkan memory type found")
            }
            Self::NoDescriptorSet => {
                formatter.write_str("Vulkan descriptor set allocation returned no sets")
            }
            Self::NoCommandBuffer => {
                formatter.write_str("Vulkan command buffer allocation returned no buffers")
            }
            Self::NoComputePipeline => {
                formatter.write_str("Vulkan compute pipeline creation returned no pipelines")
            }
            Self::InvalidDispatchShape => formatter.write_str("invalid Vulkan dispatch shape"),
            Self::BufferTooLarge => {
                formatter.write_str("buffer size does not fit Vulkan device size")
            }
            Self::BufferTooSmall => {
                formatter.write_str("buffer is too small for the requested transfer")
            }
            Self::DescriptorHeapFull { class, capacity } => {
                write!(
                    formatter,
                    "Vulkan descriptor heap {class:?} is full at capacity {capacity}"
                )
            }
            Self::UnsupportedResourceAddressing {
                requested,
                available,
            } => write!(
                formatter,
                "Vulkan resource addressing {requested:?} unsupported; available path is {available:?}"
            ),
        }
    }
}

impl Error for PcuVulkanError {}

struct SelectedPhysicalDevice {
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
    properties: vk::PhysicalDeviceProperties,
}

fn choose_instance_api_version(entry: &ash::Entry) -> Result<u32, PcuVulkanError> {
    let supported = unsafe {
        // SAFETY: This only queries loader-level Vulkan version support.
        entry.try_enumerate_instance_version()
    }
    .map_err(|result| PcuVulkanError::Vulkan {
        context: "enumerate Vulkan instance version",
        result,
    })?;
    let version = supported.unwrap_or(vk::API_VERSION_1_0);
    if version >= vk::API_VERSION_1_1 {
        Ok(vk::API_VERSION_1_1)
    } else {
        Ok(vk::API_VERSION_1_0)
    }
}

fn create_instance(entry: &ash::Entry, api_version: u32) -> Result<ash::Instance, PcuVulkanError> {
    let application_info = vk::ApplicationInfo::default()
        .application_name(RUNNER_APP_NAME)
        .application_version(1)
        .engine_name(RUNNER_ENGINE_NAME)
        .engine_version(1)
        .api_version(api_version);

    let create_info = vk::InstanceCreateInfo::default().application_info(&application_info);
    vk_try("create Vulkan instance", unsafe {
        // SAFETY: The create info only references static C strings and no extension pointers.
        entry.create_instance(&create_info, None)
    })
}

fn select_compute_device(
    instance: &ash::Instance,
) -> Result<SelectedPhysicalDevice, PcuVulkanError> {
    let physical_devices = vk_try("enumerate physical Vulkan devices", unsafe {
        // SAFETY: The instance is live for the duration of the call.
        instance.enumerate_physical_devices()
    })?;
    if physical_devices.is_empty() {
        return Err(PcuVulkanError::NoPhysicalDevice);
    }

    for physical_device in physical_devices {
        let Some(queue_family_index) = find_compute_queue_family(instance, physical_device) else {
            continue;
        };
        let properties = unsafe {
            // SAFETY: The physical device handle was returned by this live instance.
            instance.get_physical_device_properties(physical_device)
        };
        return Ok(SelectedPhysicalDevice {
            physical_device,
            queue_family_index,
            properties,
        });
    }

    Err(PcuVulkanError::NoComputeQueueFamily)
}

fn create_device(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    queue_family_index: u32,
) -> Result<ash::Device, PcuVulkanError> {
    let queue_priorities = [1.0_f32];
    let queue_create_infos = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities)];
    let create_info = vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos);
    vk_try("create Vulkan logical device", unsafe {
        // SAFETY: The physical device and queue family index were enumerated from this instance.
        instance.create_device(physical_device, &create_info, None)
    })
}

fn query_runner_caps(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    properties: &vk::PhysicalDeviceProperties,
    instance_api_version: u32,
    requested_heap_budget: PcuVulkanDescriptorHeapBudget,
) -> Result<PcuVulkanRunnerCaps, PcuVulkanError> {
    let mut descriptor_indexing = PcuVulkanDescriptorIndexingCaps {
        requested_heap_budget,
        ..PcuVulkanDescriptorIndexingCaps::default()
    };
    let mut buffer_device_address = PcuVulkanBufferDeviceAddressCaps::default();
    let push_constants = PcuVulkanPushConstantCaps {
        max_size_bytes: properties.limits.max_push_constants_size,
    };

    if instance_api_version >= vk::API_VERSION_1_1 {
        let extensions = query_device_extensions(instance, physical_device)?;
        let mut descriptor_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
        let mut buffer_address_features = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
        let mut mutable_descriptor_features =
            vk::PhysicalDeviceMutableDescriptorTypeFeaturesEXT::default();
        let mut features = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut descriptor_features)
            .push_next(&mut buffer_address_features)
            .push_next(&mut mutable_descriptor_features);
        unsafe {
            // SAFETY: The pNext feature chain contains valid feature structs for query only.
            instance.get_physical_device_features2(physical_device, &mut features);
        }

        let mut descriptor_properties = vk::PhysicalDeviceDescriptorIndexingProperties::default();
        let mut properties2 =
            vk::PhysicalDeviceProperties2::default().push_next(&mut descriptor_properties);
        unsafe {
            // SAFETY: The pNext property chain contains valid property structs for query only.
            instance.get_physical_device_properties2(physical_device, &mut properties2);
        }

        descriptor_indexing.runtime_descriptor_array =
            bool32(descriptor_features.runtime_descriptor_array);
        descriptor_indexing.sampled_image_non_uniform_indexing =
            bool32(descriptor_features.shader_sampled_image_array_non_uniform_indexing);
        descriptor_indexing.storage_buffer_non_uniform_indexing =
            bool32(descriptor_features.shader_storage_buffer_array_non_uniform_indexing);
        descriptor_indexing.storage_image_non_uniform_indexing =
            bool32(descriptor_features.shader_storage_image_array_non_uniform_indexing);
        descriptor_indexing.partially_bound =
            bool32(descriptor_features.descriptor_binding_partially_bound);
        descriptor_indexing.update_unused_while_pending =
            bool32(descriptor_features.descriptor_binding_update_unused_while_pending);
        descriptor_indexing.variable_descriptor_count =
            bool32(descriptor_features.descriptor_binding_variable_descriptor_count);
        descriptor_indexing.mutable_descriptor_type = extensions.mutable_descriptor_type
            && bool32(mutable_descriptor_features.mutable_descriptor_type);
        descriptor_indexing.actual_heap_budget = if descriptor_indexing.runtime_descriptor_array {
            requested_heap_budget.clamp_to(PcuVulkanDescriptorHeapBudget {
                sampled_images: descriptor_properties
                    .max_descriptor_set_update_after_bind_sampled_images,
                storage_buffers: descriptor_properties
                    .max_descriptor_set_update_after_bind_storage_buffers,
                storage_images: descriptor_properties
                    .max_descriptor_set_update_after_bind_storage_images,
                samplers: descriptor_properties.max_descriptor_set_update_after_bind_samplers,
            })
        } else {
            PcuVulkanDescriptorHeapBudget::empty()
        };

        buffer_device_address.supported = extensions.buffer_device_address
            && bool32(buffer_address_features.buffer_device_address);
        buffer_device_address.capture_replay =
            bool32(buffer_address_features.buffer_device_address_capture_replay);
        buffer_device_address.multi_device =
            bool32(buffer_address_features.buffer_device_address_multi_device);
    }

    Ok(PcuVulkanRunnerCaps {
        api_version: instance_api_version,
        descriptor_indexing,
        buffer_device_address,
        push_constants,
        selected_storage_buffer_model: PcuResourceAddressingModel::FixedDescriptors,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct PcuVulkanDeviceExtensions {
    buffer_device_address: bool,
    mutable_descriptor_type: bool,
}

fn query_device_extensions(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<PcuVulkanDeviceExtensions, PcuVulkanError> {
    let properties = vk_try("enumerate Vulkan device extensions", unsafe {
        // SAFETY: The physical device handle was returned by this live instance.
        instance.enumerate_device_extension_properties(physical_device)
    })?;
    let mut extensions = PcuVulkanDeviceExtensions::default();
    for property in properties {
        let name = unsafe {
            // SAFETY: Vulkan extension names are fixed-size null-terminated C strings.
            CStr::from_ptr(property.extension_name.as_ptr())
        };
        if name == vk::KHR_BUFFER_DEVICE_ADDRESS_NAME {
            extensions.buffer_device_address = true;
        }
        if name == vk::EXT_MUTABLE_DESCRIPTOR_TYPE_NAME {
            extensions.mutable_descriptor_type = true;
        }
    }
    Ok(extensions)
}

fn create_shader_module<'a>(
    device: &'a ash::Device,
    words: &[u32],
) -> Result<VulkanShaderModule<'a>, PcuVulkanError> {
    let create_info = vk::ShaderModuleCreateInfo::default().code(words);
    let handle = vk_try(
        "create Vulkan shader module from fusion-pcu SPIR-V",
        unsafe {
            // SAFETY: `words` is SPIR-V word-aligned `u32` storage and lives for the duration of the call.
            device.create_shader_module(&create_info, None)
        },
    )?;
    Ok(VulkanShaderModule { device, handle })
}

fn create_descriptor_set_layout(
    device: &ash::Device,
) -> Result<VulkanDescriptorSetLayout<'_>, PcuVulkanError> {
    let bindings = [
        storage_buffer_layout_binding(0),
        storage_buffer_layout_binding(1),
        storage_buffer_layout_binding(2),
    ];
    let create_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let handle = vk_try("create Vulkan descriptor set layout", unsafe {
        // SAFETY: The create info references only stack-local binding descriptors for the duration of the call.
        device.create_descriptor_set_layout(&create_info, None)
    })?;
    Ok(VulkanDescriptorSetLayout { device, handle })
}

fn storage_buffer_layout_binding(binding: u32) -> vk::DescriptorSetLayoutBinding<'static> {
    vk::DescriptorSetLayoutBinding::default()
        .binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
}

fn create_pipeline_layout(
    device: &ash::Device,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<VulkanPipelineLayout<'_>, PcuVulkanError> {
    let set_layouts = [descriptor_set_layout];
    let create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let handle = vk_try("create Vulkan pipeline layout", unsafe {
        // SAFETY: The descriptor set layout handle is live and belongs to this device.
        device.create_pipeline_layout(&create_info, None)
    })?;
    Ok(VulkanPipelineLayout { device, handle })
}

fn create_compute_pipeline<'a>(
    device: &'a ash::Device,
    shader_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    entry_point: &CStr,
) -> Result<VulkanComputePipeline<'a>, PcuVulkanError> {
    let stage = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(entry_point);
    let create_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage)
        .layout(pipeline_layout);

    let pipelines = unsafe {
        // SAFETY: The shader module and pipeline layout are live and belong to this device.
        device.create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
    };
    match pipelines {
        Ok(mut pipelines) => {
            let Some(handle) = pipelines.pop() else {
                return Err(PcuVulkanError::NoComputePipeline);
            };
            Ok(VulkanComputePipeline { device, handle })
        }
        Err((partial, result)) => {
            for pipeline in partial {
                unsafe {
                    // SAFETY: Partial pipelines returned by Vulkan belong to this device and are not otherwise owned.
                    device.destroy_pipeline(pipeline, None);
                }
            }
            Err(PcuVulkanError::Vulkan {
                context: "create Vulkan compute pipeline",
                result,
            })
        }
    }
}

fn create_descriptor_pool(
    device: &ash::Device,
) -> Result<VulkanDescriptorPool<'_>, PcuVulkanError> {
    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3)];
    let create_info = vk::DescriptorPoolCreateInfo::default()
        .max_sets(1)
        .pool_sizes(&pool_sizes);
    let handle = vk_try("create Vulkan descriptor pool", unsafe {
        // SAFETY: The create info references only stack-local pool sizes for the duration of the call.
        device.create_descriptor_pool(&create_info, None)
    })?;
    Ok(VulkanDescriptorPool { device, handle })
}

fn allocate_descriptor_set(
    device: &ash::Device,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, PcuVulkanError> {
    let set_layouts = [descriptor_set_layout];
    let allocate_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
    let sets = vk_try("allocate Vulkan descriptor set", unsafe {
        // SAFETY: The descriptor pool and layout are live and belong to this device.
        device.allocate_descriptor_sets(&allocate_info)
    })?;
    sets.first().copied().ok_or(PcuVulkanError::NoDescriptorSet)
}

fn update_float_storage_descriptors(
    device: &ash::Device,
    descriptor_set: vk::DescriptorSet,
    source: &VulkanBuffer<'_>,
    bias: &VulkanBuffer<'_>,
    output: &VulkanBuffer<'_>,
) {
    let source_info = source.descriptor_info();
    let bias_info = bias.descriptor_info();
    let output_info = output.descriptor_info();
    let source_infos = [source_info];
    let bias_infos = [bias_info];
    let output_infos = [output_info];
    let writes = [
        storage_buffer_write(descriptor_set, 0, &source_infos),
        storage_buffer_write(descriptor_set, 1, &bias_infos),
        storage_buffer_write(descriptor_set, 2, &output_infos),
    ];
    unsafe {
        // SAFETY: Descriptor set, buffers, and buffer ranges are live for this update call.
        device.update_descriptor_sets(&writes, &[]);
    }
}

fn storage_buffer_write(
    descriptor_set: vk::DescriptorSet,
    binding: u32,
    buffer_info: &[vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'_> {
    vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(buffer_info)
}

fn create_command_pool(
    device: &ash::Device,
    queue_family_index: u32,
) -> Result<VulkanCommandPool<'_>, PcuVulkanError> {
    let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(queue_family_index)
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
    let handle = vk_try("create Vulkan command pool", unsafe {
        // SAFETY: The queue family index was selected from this physical device before logical device creation.
        device.create_command_pool(&create_info, None)
    })?;
    Ok(VulkanCommandPool { device, handle })
}

fn allocate_command_buffer(
    device: &ash::Device,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer, PcuVulkanError> {
    let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers = vk_try("allocate Vulkan command buffer", unsafe {
        // SAFETY: The command pool is live and belongs to this device.
        device.allocate_command_buffers(&allocate_info)
    })?;
    command_buffers
        .first()
        .copied()
        .ok_or(PcuVulkanError::NoCommandBuffer)
}

fn record_compute_commands(
    device: &ash::Device,
    command_buffer: vk::CommandBuffer,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set: vk::DescriptorSet,
    dispatch_groups_x: u32,
) -> Result<(), PcuVulkanError> {
    let begin_info = vk::CommandBufferBeginInfo::default();
    vk_try("begin Vulkan command buffer", unsafe {
        // SAFETY: The command buffer is allocated from a resettable pool and currently not recording.
        device.begin_command_buffer(command_buffer, &begin_info)
    })?;
    unsafe {
        // SAFETY: Pipeline, layout, and descriptor set are live and compatible by construction.
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            &[descriptor_set],
            &[],
        );
        device.cmd_dispatch(command_buffer, dispatch_groups_x, 1, 1);
    }
    vk_try("end Vulkan command buffer", unsafe {
        // SAFETY: The command buffer is in the recording state.
        device.end_command_buffer(command_buffer)
    })
}

fn create_fence(device: &ash::Device) -> Result<VulkanFence<'_>, PcuVulkanError> {
    let create_info = vk::FenceCreateInfo::default();
    let handle = vk_try("create Vulkan fence", unsafe {
        // SAFETY: The create info contains no borrowed extension data.
        device.create_fence(&create_info, None)
    })?;
    Ok(VulkanFence { device, handle })
}

fn submit_and_wait(
    device: &ash::Device,
    queue: vk::Queue,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
) -> Result<(), PcuVulkanError> {
    let command_buffers = [command_buffer];
    let submit_infos = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
    vk_try("submit Vulkan compute work", unsafe {
        // SAFETY: The queue and command buffer belong to this device, and the fence is unsignaled.
        device.queue_submit(queue, &submit_infos, fence)
    })?;
    vk_try("wait for Vulkan compute work", unsafe {
        // SAFETY: The fence belongs to this device and was submitted above.
        device.wait_for_fences(&[fence], true, u64::MAX)
    })
}

fn find_compute_queue_family(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
    let families = unsafe {
        // SAFETY: The physical device handle was returned by this live instance.
        instance.get_physical_device_queue_family_properties(physical_device)
    };

    families.iter().enumerate().find_map(|(index, family)| {
        if family.queue_count == 0 || !family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            return None;
        }
        u32::try_from(index).ok()
    })
}

fn physical_device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    let name = unsafe {
        // SAFETY: Vulkan guarantees `device_name` is a null-terminated string in this fixed array.
        CStr::from_ptr(properties.device_name.as_ptr())
    };
    name.to_string_lossy().into_owned()
}

fn vk_try<T>(context: &'static str, result: Result<T, vk::Result>) -> Result<T, PcuVulkanError> {
    result.map_err(|result| PcuVulkanError::Vulkan { context, result })
}

const fn bool32(value: vk::Bool32) -> bool {
    value == vk::TRUE
}

const fn min_u32(left: u32, right: u32) -> u32 {
    if left < right { left } else { right }
}

struct VulkanShaderModule<'a> {
    device: &'a ash::Device,
    handle: vk::ShaderModule,
}

impl Drop for VulkanShaderModule<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the shader module and destroys it before the device guard drops.
            self.device.destroy_shader_module(self.handle, None);
        }
    }
}

struct VulkanDescriptorSetLayout<'a> {
    device: &'a ash::Device,
    handle: vk::DescriptorSetLayout,
}

impl Drop for VulkanDescriptorSetLayout<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the descriptor set layout and destroys it exactly once.
            self.device.destroy_descriptor_set_layout(self.handle, None);
        }
    }
}

struct VulkanPipelineLayout<'a> {
    device: &'a ash::Device,
    handle: vk::PipelineLayout,
}

impl Drop for VulkanPipelineLayout<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the pipeline layout and destroys it exactly once.
            self.device.destroy_pipeline_layout(self.handle, None);
        }
    }
}

struct VulkanComputePipeline<'a> {
    device: &'a ash::Device,
    handle: vk::Pipeline,
}

impl Drop for VulkanComputePipeline<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the compute pipeline and destroys it exactly once.
            self.device.destroy_pipeline(self.handle, None);
        }
    }
}

struct VulkanDescriptorPool<'a> {
    device: &'a ash::Device,
    handle: vk::DescriptorPool,
}

impl Drop for VulkanDescriptorPool<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the descriptor pool and destroys it exactly once.
            self.device.destroy_descriptor_pool(self.handle, None);
        }
    }
}

struct VulkanCommandPool<'a> {
    device: &'a ash::Device,
    handle: vk::CommandPool,
}

impl Drop for VulkanCommandPool<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the command pool and destroys it exactly once.
            self.device.destroy_command_pool(self.handle, None);
        }
    }
}

struct VulkanFence<'a> {
    device: &'a ash::Device,
    handle: vk::Fence,
}

impl Drop for VulkanFence<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the fence and destroys it exactly once.
            self.device.destroy_fence(self.handle, None);
        }
    }
}

struct VulkanBuffer<'a> {
    device: &'a ash::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
}

impl<'a> VulkanBuffer<'a> {
    fn new_storage_f32_buffer(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        device: &'a ash::Device,
        element_count: usize,
    ) -> Result<Self, PcuVulkanError> {
        let size = f32_slice_byte_len(element_count)?;
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::STORAGE_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = vk_try("create Vulkan storage buffer", unsafe {
            // SAFETY: The create info contains no borrowed extension data.
            device.create_buffer(&create_info, None)
        })?;
        let requirements = unsafe {
            // SAFETY: The buffer was created on this device and is live.
            device.get_buffer_memory_requirements(buffer)
        };
        let memory_properties = unsafe {
            // SAFETY: The physical device handle was returned by this live instance.
            instance.get_physical_device_memory_properties(physical_device)
        };
        let memory_type_index = match find_memory_type(
            &memory_properties,
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) {
            Ok(index) => index,
            Err(error) => {
                unsafe {
                    // SAFETY: The buffer is live and not yet owned by a guard.
                    device.destroy_buffer(buffer, None);
                }
                return Err(error);
            }
        };
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let memory = match vk_try("allocate Vulkan buffer memory", unsafe {
            // SAFETY: The allocation info was built from this buffer's memory requirements.
            device.allocate_memory(&allocate_info, None)
        }) {
            Ok(memory) => memory,
            Err(error) => {
                unsafe {
                    // SAFETY: The buffer is live and not yet owned by a guard.
                    device.destroy_buffer(buffer, None);
                }
                return Err(error);
            }
        };
        if let Err(error) = vk_try("bind Vulkan buffer memory", unsafe {
            // SAFETY: The buffer and memory are live, compatible, and offset zero satisfies Vulkan requirements here.
            device.bind_buffer_memory(buffer, memory, 0)
        }) {
            unsafe {
                // SAFETY: The memory and buffer are live and not yet owned by a guard.
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
            }
            return Err(error);
        }

        Ok(Self {
            device,
            buffer,
            memory,
            size,
        })
    }

    fn write_f32s(&self, values: &[f32]) -> Result<(), PcuVulkanError> {
        let byte_len = f32_slice_byte_len(values.len())?;
        if byte_len > self.size {
            return Err(PcuVulkanError::BufferTooSmall);
        }
        let mapped = vk_try("map Vulkan buffer memory for write", unsafe {
            // SAFETY: The memory is host-visible and this runner maps the whole write range exclusively.
            self.device
                .map_memory(self.memory, 0, byte_len, vk::MemoryMapFlags::empty())
        })?;
        unsafe {
            // SAFETY: The mapped range is at least `values.len() * size_of::<f32>()` bytes and both regions are non-overlapping.
            ptr::copy_nonoverlapping(values.as_ptr(), mapped.cast::<f32>(), values.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    fn read_f32s(&self, values: &mut [f32]) -> Result<(), PcuVulkanError> {
        let byte_len = f32_slice_byte_len(values.len())?;
        if byte_len > self.size {
            return Err(PcuVulkanError::BufferTooSmall);
        }
        let mapped = vk_try("map Vulkan buffer memory for read", unsafe {
            // SAFETY: The memory is host-visible and GPU execution has completed before this read.
            self.device
                .map_memory(self.memory, 0, byte_len, vk::MemoryMapFlags::empty())
        })?;
        unsafe {
            // SAFETY: The mapped range is at least `values.len() * size_of::<f32>()` bytes and both regions are non-overlapping.
            ptr::copy_nonoverlapping(mapped.cast::<f32>(), values.as_mut_ptr(), values.len());
            self.device.unmap_memory(self.memory);
        }
        Ok(())
    }

    fn descriptor_info(&self) -> vk::DescriptorBufferInfo {
        vk::DescriptorBufferInfo::default()
            .buffer(self.buffer)
            .offset(0)
            .range(self.size)
    }
}

impl Drop for VulkanBuffer<'_> {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This guard owns the buffer and memory and destroys/frees them exactly once.
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

fn f32_slice_byte_len(count: usize) -> Result<vk::DeviceSize, PcuVulkanError> {
    let Some(bytes) = count.checked_mul(mem::size_of::<f32>()) else {
        return Err(PcuVulkanError::BufferTooLarge);
    };
    vk::DeviceSize::try_from(bytes).map_err(|_| PcuVulkanError::BufferTooLarge)
}

fn find_memory_type(
    properties: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    required: vk::MemoryPropertyFlags,
) -> Result<u32, PcuVulkanError> {
    for index in 0..properties.memory_type_count {
        let Some(bit) = 1_u32.checked_shl(index) else {
            continue;
        };
        if (memory_type_bits & bit) == 0 {
            continue;
        }
        let Some(memory_type) = usize::try_from(index)
            .ok()
            .and_then(|index| properties.memory_types.get(index))
        else {
            continue;
        };
        if memory_type.property_flags.contains(required) {
            return Ok(index);
        }
    }
    Err(PcuVulkanError::NoHostVisibleCoherentMemory)
}

#[cfg(test)]
mod tests {
    use super::{
        PcuVulkanDescriptorHeapBudget,
        PcuVulkanDescriptorIndexingCaps,
    };

    #[test]
    fn descriptor_heap_budget_clamps_to_device_limits() {
        let requested = PcuVulkanDescriptorHeapBudget::portable_default();
        let limits = PcuVulkanDescriptorHeapBudget {
            sampled_images: 64,
            storage_buffers: 32,
            storage_images: 16,
            samplers: 8,
        };

        let actual = requested.clamp_to(limits);

        assert_eq!(actual, limits);
    }

    #[test]
    fn descriptor_indexing_caps_are_class_specific() {
        let caps = PcuVulkanDescriptorIndexingCaps {
            runtime_descriptor_array: true,
            storage_buffer_non_uniform_indexing: true,
            sampled_image_non_uniform_indexing: false,
            ..PcuVulkanDescriptorIndexingCaps::default()
        };

        assert!(caps.supports_storage_buffer_heap());
        assert!(!caps.supports_sampled_image_heap());
    }
}
