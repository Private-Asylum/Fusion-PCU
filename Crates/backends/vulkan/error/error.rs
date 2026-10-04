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
    PcuBindingRef,
    PcuError,
    PcuExecutionFault,
    PcuTypedDispatchValidationError,
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
    ShaderArtifact(crate::PcuVulkanShaderArtifactError),
    DispatchAdmission {
        error: PcuError,
    },
    SpirvLowering {
        error: PcuSpirvError,
    },
    UnsupportedParameters,
    UnsupportedBindingProfile,
    UnsupportedPreparedProfile,
    InvalidArguments,
    /// A completed shader emitted a malformed raw or operation-incompatible fault record.
    InvalidStatus {
        code: u32,
        invocation_id: u64,
    },
    /// A completed ordered compound emitted an invalid physical kind or semantic location.
    InvalidCompoundStatus {
        code: u32,
        element_index: u64,
        reduction_index: u32,
        step: u32,
    },
    InvalidOfferShape,
    InvalidOfferBinding(PcuBindingRef),
    InvalidOfferValueFlow(PcuTypedDispatchValidationError),
    Fault(PcuExecutionFault),
    CompletionUnknown,
    Quarantined,
    DeviceLimitExceeded,
    InvalidDiscoveryReference,
    DiscoveryGenerationExhausted,
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
            Self::ShaderArtifact(error) => write!(formatter, "Vulkan shader artifact: {error}"),
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
            Self::UnsupportedPreparedProfile => {
                formatter.write_str("unsupported Vulkan prepared bit-map profile")
            }
            Self::InvalidArguments => formatter.write_str("invalid Vulkan prepared host arguments"),
            Self::InvalidStatus {
                code,
                invocation_id,
            } => write!(
                formatter,
                "invalid Vulkan shader status {code} at logical invocation {invocation_id}",
            ),
            Self::InvalidCompoundStatus {
                code,
                element_index,
                reduction_index,
                step,
            } => write!(
                formatter,
                "invalid Vulkan compound status {code} at output {element_index}, reduction {reduction_index}, step {step}"
            ),
            Self::InvalidOfferShape => formatter.write_str("invalid Vulkan offer logical shape"),
            Self::InvalidOfferBinding(binding) => {
                write!(formatter, "invalid Vulkan offer binding: {binding:?}")
            }
            Self::InvalidOfferValueFlow(error) => {
                write!(formatter, "invalid Vulkan offer value flow: {error:?}")
            }
            Self::Fault(fault) => write!(formatter, "Vulkan checked execution fault: {fault:?}"),
            Self::CompletionUnknown => {
                formatter.write_str("Vulkan completion unknown; native owners quarantined")
            }
            Self::Quarantined => {
                formatter.write_str("Vulkan session is quarantined after unknown completion")
            }
            Self::DeviceLimitExceeded => {
                formatter.write_str("Vulkan execution exceeds selected device limits")
            }
            Self::InvalidDiscoveryReference => {
                formatter.write_str("invalid, stale or changed Vulkan device reference")
            }
            Self::DiscoveryGenerationExhausted => {
                formatter.write_str("Vulkan discovery generation exhausted")
            }
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
