//! Explicit synchronous Vulkan execution of bounded PCU SPIR-V compute profiles.
//!
//! Enable `hosted` to opt into loader discovery and execution. The default crate loads no
//! Vulkan runtime. Prepared typed host calls admit twenty-two raw carriers, fourteen-width
//! checked integer maps, eight-width checked division/remainder, six-format checked binary/unary
//! maps and F32/F64 checked conversions, retaining native storage, pipeline and submission objects.
//! Exact numerical and publication contracts are checked independently of coarse capabilities.
//! The legacy executor additionally
//! supports its three-buffer raw F32 dispatch profile, waiting for completion before return.
//! Capability inspection does not enable descriptor indexing or buffer-device-address execution.
//! Per-function `#[pcu]` source supports explicit preparation and ordinary facade calls through
//! its optional `vulkan` feature. The optional `tensor` leaf table also backs ordinary annotated
//! identity source with retained native owners and exact originating-session affinity.
//! Constant/Uniform leaf APIs do not imply ordinary uniform source or tensor arithmetic support.
//! External memory is not imported.

extern crate fusion_pcu_core as fusion_pcu;

#[cfg(feature = "hosted")]
#[path = "arguments/arguments.rs"]
mod arguments;
#[cfg(feature = "hosted")]
pub use arguments::PcuVulkanArgument;

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
#[path = "owned/owned.rs"]
mod owned;
#[cfg(feature = "hosted")]
#[path = "prepared/prepared.rs"]
mod prepared;
#[cfg(feature = "hosted")]
#[path = "types/types.rs"]
mod types;
#[cfg(feature = "hosted")]
#[rustfmt::skip]
pub use owned::{
    PcuVulkanOwnedBuffer,
    PcuVulkanPreparedCarrierCopy,
};
#[cfg(feature = "tensor")]
#[path = "tensor/tensor.rs"]
mod tensor;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use tensor::{
    PcuVulkanPreparedTensorGraph,
    PcuVulkanPreparedScalarTensorGraph,
    PcuVulkanTensorBinding,
    PcuVulkanTensorError,
    PcuVulkanTensorInput,
};

#[cfg(feature = "hosted")]
#[rustfmt::skip]
pub use crate::{
    dispatch::PcuVulkanBackend,
    error::PcuVulkanError,
    prepared::PcuVulkanPreparedBitMap,
    prepared::PcuVulkanPreparedBinary,
    prepared::PcuVulkanBinaryMemoryRealizations,
    prepared::PcuVulkanIntegerMemoryRealizations,
    prepared::PcuVulkanPreparedInteger,
    prepared::PcuVulkanPreparedDivRem,
    prepared::PcuVulkanDivRemMemoryRealizations,
    prepared::PcuVulkanPreparedUnary,
    prepared::PcuVulkanPreparedUnaryRoles,
    prepared::PcuVulkanPreparedConversion,
    prepared::PcuVulkanPreparedScalarTransport,
    prepared::PcuVulkanPreparedHost,
    prepared::PcuVulkanPreparedComposed,
    prepared::PcuVulkanPreparedOrderedTransport,
    ffi::PcuVulkanComposedMemoryRealizations,
    prepared::PcuVulkanPreparedMixed,
    prepared::PcuVulkanPreparedMemoryRealizations,
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
