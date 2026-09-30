//! Limited Vulkan prototype for synchronous PCU SPIR-V dispatch.
//!
//! Enable `hosted` to opt into loader discovery and execution. The default crate loads no
//! Vulkan runtime. The prototype lowers legacy dispatch IR and executes exactly three fixed
//! storage-buffer descriptors (two inputs and one output), waiting for completion before return.
//! Capability inspection does not enable descriptor indexing or buffer-device-address execution.
//! Modern `#[pcu]` source, facade, and owned-backend integration remain deferred.

extern crate fusion_pcu_core as fusion_pcu;

#[cfg(feature = "hosted")]
#[path = "dispatch/dispatch.rs"]
mod dispatch;
#[cfg(feature = "hosted")]
#[path = "error/error.rs"]
mod error;
#[cfg(feature = "hosted")]
#[path = "ffi/ffi.rs"]
mod ffi;
#[cfg(feature = "hosted")]
#[path = "types/types.rs"]
mod types;

#[cfg(feature = "hosted")]
#[rustfmt::skip]
pub use crate::{
    dispatch::PcuVulkanBackend,
    error::PcuVulkanError,
    types::{
        PcuVulkanBufferDeviceAddressCaps,
        PcuVulkanCaps,
        PcuVulkanDescriptorClass,
        PcuVulkanDescriptorHeapBudget,
        PcuVulkanDescriptorIndexingCaps,
        PcuVulkanDispatchReport,
        PcuVulkanExecutionReport,
        PcuVulkanPushConstantCaps,
        PcuVulkanResourceAddressingModel,
    },
};
