//! Errors from prototype admission, lowering, and Vulkan execution.
#[rustfmt::skip]
use core::{
    fmt,
};
#[rustfmt::skip]
use std::{
    error::Error,
};
#[rustfmt::skip]
use ash::{
    vk,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuError,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    PcuSpirvError,
};
#[rustfmt::skip]
use crate::types::{
    PcuVulkanDescriptorClass,
    PcuVulkanResourceAddressingModel,
};

#[derive(Debug)]
pub enum PcuVulkanError {
    DispatchAdmission {
        error: PcuError,
    },
    SpirvLowering {
        error: PcuSpirvError,
    },
    UnsupportedParameters,
    UnsupportedBindingProfile,
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
    InvalidInvocationBinding,
    BufferTooLarge,
    BufferTooSmall,
    DescriptorHeapFull {
        class: PcuVulkanDescriptorClass,
        capacity: u32,
    },
    UnsupportedResourceAddressing {
        requested: PcuVulkanResourceAddressingModel,
        available: PcuVulkanResourceAddressingModel,
    },
}

impl fmt::Display for PcuVulkanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DispatchAdmission { error } => {
                write!(formatter, "Vulkan dispatch admission failed: {error}")
            }
            Self::SpirvLowering { error } => {
                write!(formatter, "Vulkan SPIR-V lowering failed: {error}")
            }
            Self::UnsupportedParameters => {
                formatter.write_str("Vulkan prototype does not support scalar parameters")
            }
            Self::UnsupportedBindingProfile => formatter.write_str(
                "Vulkan prototype requires exactly set-zero bindings zero, one, and two",
            ),
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
            Self::InvalidInvocationBinding => {
                formatter.write_str("invalid Vulkan invocation binding")
            }
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
