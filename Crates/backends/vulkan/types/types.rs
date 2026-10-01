#[rustfmt::skip]
use std::{
    string::String,
};

/// Descriptor class used for heap budgeting and registration failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuVulkanDescriptorClass {
    SampledImage,
    StorageBuffer,
    StorageImage,
    Sampler,
}

/// Conservative descriptor-heap budget requested by the prototype.
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

/// Capabilities observed for the selected Vulkan device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuVulkanCaps {
    pub api_version: u32,
    /// Queried native Float64 support; activation enables it only when present.
    pub shader_float64: bool,
    pub descriptor_indexing: PcuVulkanDescriptorIndexingCaps,
    pub buffer_device_address: PcuVulkanBufferDeviceAddressCaps,
    pub push_constants: PcuVulkanPushConstantCaps,
    pub selected_storage_buffer_model: PcuVulkanResourceAddressingModel,
}

/// Result metadata returned by the current fixed-descriptor dispatch path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuVulkanExecutionReport {
    pub invocations: u32,
    pub dispatch_groups: [u32; 3],
    pub bound_byte_len: usize,
    pub resource_model: PcuVulkanResourceAddressingModel,
}

/// Resource addressing models reported by Vulkan capability inspection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuVulkanResourceAddressingModel {
    FixedDescriptors,
    DescriptorIndex,
    BufferDeviceAddress,
}

/// SPIR-V module passed to the private Vulkan executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuVulkanLoweredSpirvDispatch<'a> {
    pub words: &'a [u32],
    pub local_size: [u32; 3],
}

/// Reported properties of one actual native allocation, fixed during cold preparation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuVulkanMemoryRealization {
    pub native_memory_type: u32,
    /// Exact `VkMemoryPropertyFlagBits` returned for the selected memory type.
    pub native_property_flags: u32,
    pub allocation_bytes: u64,
}

impl PcuVulkanMemoryRealization {
    #[must_use]
    pub const fn host_cached(self) -> bool {
        self.native_property_flags & 8 != 0
    }
    #[must_use]
    pub const fn host_coherent(self) -> bool {
        self.native_property_flags & 4 != 0
    }
    #[must_use]
    pub const fn device_local(self) -> bool {
        self.native_property_flags & 1 != 0
    }
}

/// Metadata for one synchronous prototype dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuVulkanDispatchReport {
    pub device_name: String,
    pub spirv_words: usize,
    pub spirv_bound: u32,
    pub execution: PcuVulkanExecutionReport,
}

const fn min_u32(left: u32, right: u32) -> u32 {
    if left < right { left } else { right }
}

/// Optional successful-call diagnostic timings; ordinary calls read no measurement clock.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PcuVulkanCallMeasurements {
    pub upload: std::time::Duration,
    /// Fence reset, submit-info preparation and native queue submission.
    pub submission: std::time::Duration,
    pub completion: std::time::Duration,
    pub diagnostic_publication: std::time::Duration,
    /// Native host-boundary time; outer typed argument validation is excluded.
    pub wall: std::time::Duration,
}
