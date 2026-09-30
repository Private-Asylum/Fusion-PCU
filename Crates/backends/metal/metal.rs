//! Bounded Metal runtime with owned resources, checked integer maps and F32 bit maps.
//!
//! The safe entry points compile only this crate's admitted kernels. Native floating arithmetic,
//! general dispatch IR, `MatMul` and training operations are not admitted. Other platforms return
//! an explicit unsupported error; they never evaluate device work on the CPU.

extern crate fusion_pcu_core as fusion_pcu;

#[path = "admission/admission.rs"]
mod admission;
#[rustfmt::skip]
pub use admission::{
    MetalPreparedU32Kernel,
    MetalPreparedF32Kernel,
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
    MetalBuffer,
    MetalDeviceFacts,
    MetalError,
    MetalFault,
    MetalIntegerOp,
    MetalPreparedIntegerMap,
    MetalPreparedF32Unary,
    MetalSession,
};
