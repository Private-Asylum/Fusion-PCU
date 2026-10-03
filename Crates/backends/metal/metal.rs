//! Bounded Metal runtime with owned resources, checked integer maps and exact F32/F64 synthesis.
//!
//! The safe entry points compile only this crate's admitted kernels. Native floating arithmetic,
//! general dispatch IR, `MatMul` and training operations are not admitted. Other platforms return
//! an explicit unsupported error; they never evaluate device work on the CPU.

extern crate fusion_pcu_core as fusion_pcu;

#[path = "admission/admission.rs"]
mod admission;
#[rustfmt::skip]
pub use admission::{
    MetalPreparedCarrierKernel,
    MetalPreparedU32Kernel,
    MetalPreparedIntegerKernel,
    MetalPreparedF32Kernel,
    MetalPreparedF64Kernel,
    MetalPreparedFloatKernel,
    MetalCheckedUnaryPlan,
    MetalPortableUnaryPlan,
    MetalPreparedF32BinaryKernel,
    MetalPreparedF64BinaryKernel,
    MetalPreparedFloatBinaryKernel,
};

#[path = "discovery/discovery.rs"]
mod discovery;
pub use discovery::MetalDiscovery;

#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "host_kernel/host_kernel.rs"]
mod host_kernel;
#[rustfmt::skip]
pub use host_kernel::{
    MetalHostKernelError,
    MetalMixedHostArgument,
    MetalPreparedHostKernel,
};

#[path = "memory/memory.rs"]
mod memory;
#[rustfmt::skip]
pub use memory::{
    MetalMemoryProvider,
    MetalMemoryResource,
    MetalMemoryMapping,
    MetalMemoryImport,
};

#[path = "owned_dispatch/owned_dispatch.rs"]
mod owned_dispatch;
#[rustfmt::skip]
pub use owned_dispatch::{
    MetalOwnedDispatchBackend,
    MetalOwnedDispatchError,
    MetalOwnedResource,
    MetalOwnedCompletion,
    MetalPreparedDispatch,
    MetalPreparedDeviceKernel,
};

#[path = "runtime/runtime.rs"]
mod runtime;

#[rustfmt::skip]
pub use runtime::{
    MetalPreparedCarrierControl,
    MetalPreparedIntegerControl,
    MetalPreparedDivRemControl,
    MetalBuffer,
    MetalDeviceFacts,
    MetalError,
    MetalFault,
    MetalIntegerOp,
    MetalPreparedIntegerMap,
    MetalPreparedF32Unary,
    MetalPreparedF64Unary,
    MetalPreparedFloatUnary,
    MetalPreparedF32Binary,
    MetalPreparedF64Binary,
    MetalPreparedFloatBinary,
    MetalSession,
};

#[path = "div_rem/div_rem.rs"]
mod div_rem;
pub use div_rem::{MetalDivRemHostBackend, MetalPreparedDivRemHostKernel};
#[rustfmt::skip]
pub use div_rem::roles::{
    MetalDivRemRoleHostBackend,
    MetalDivRemRolePlan,
    MetalPreparedDivRemRoleHostKernel,
};

#[cfg(feature = "tensor")]
#[path = "tensor/tensor.rs"]
mod tensor;
#[cfg(feature="tensor")]
#[rustfmt::skip]
pub use tensor::{
    MetalTensorPlan,
    MetalTensorInput,
    MetalTensorOwner,
    MetalPreparedTensorProgram,
    MetalTensorBinaryPlan,
    MetalPreparedTensorBinaryProgram,
};

#[cfg(feature = "benchmark-control")]
#[rustfmt::skip]
pub use tensor::{
    MetalNativeTensorReluControl,
    MetalNativeTensorCarrierControl,
    MetalNativeTensorBinaryControl,
};

#[cfg(feature = "api-census")]
#[rustfmt::skip]
pub use ffi::{MetalApiCallCensus,api_call_census,reset_api_call_census};

#[rustfmt::skip]
pub use runtime::transport::{
    MetalTransportPlan,
    MetalPreparedTransportKernel,
    MetalTransportHostBackend,
    MetalPreparedTransportHostKernel,
};

#[path = "composed/composed.rs"]
mod composed;
pub use composed::{MetalCheckedMapPlan, MetalCheckedMapEffect};
#[rustfmt::skip]
pub use runtime::composed::{
    MetalComposedHostBackend,
    MetalPreparedCheckedMapKernel,
    MetalPreparedCheckedMapHostKernel,
};
