//! Explicit synchronous Vulkan execution of bounded PCU SPIR-V compute profiles.
//!
//! Enable `hosted` to opt into loader discovery and execution. The default crate loads no
//! Vulkan runtime. Prepared typed host calls admit exact F32/F64 transport and checked sign-bit Neg,
//! retaining native storage, pipeline and submission objects. The legacy executor additionally
//! supports its three-buffer raw F32 dispatch profile, waiting for completion before return.
//! Capability inspection does not enable descriptor indexing or buffer-device-address execution.
//! Per-function `#[pcu]` source supports explicit preparation and ordinary facade calls through
//! its optional `vulkan` feature. Resident owned-backend integration is a separate slice.
//! External memory is not imported.

extern crate fusion_pcu_core as fusion_pcu;

#[cfg(feature = "hosted")]
#[path = "discovery/discovery.rs"]
mod discovery;
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
#[path = "offers/offers.rs"]
mod offers;
#[cfg(feature = "hosted")]
#[path = "prepared/prepared.rs"]
mod prepared;
#[cfg(feature = "hosted")]
#[path = "types/types.rs"]
mod types;

#[cfg(feature = "hosted")]
#[rustfmt::skip]
pub use crate::{
    dispatch::PcuVulkanBackend,
    error::PcuVulkanError,
    prepared::PcuVulkanPreparedBitMap,
    discovery::PcuVulkanDiscovery,
    types::{
        PcuVulkanBufferDeviceAddressCaps,
        PcuVulkanCaps,
        PcuVulkanDescriptorClass,
        PcuVulkanDescriptorHeapBudget,
        PcuVulkanDescriptorIndexingCaps,
        PcuVulkanDispatchReport,
        PcuVulkanExecutionReport,
        PcuVulkanPushConstantCaps,
        PcuVulkanMemoryRealization,
        PcuVulkanCallMeasurements,
        PcuVulkanResourceAddressingModel,
    },
};
