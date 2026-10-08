//! Explicit delegated Apple silicon MLX GPU tensor runtime for Fusion PCU.
//!
//! MLX is selected independently from native Metal. The initial tensor contract admits only
//! dense F32 matrix products with explicit native compound and optimized precision permissions.
//! Checked defaults, strict arithmetic, portable arithmetic and unproved precision policies
//! reject before device work. Other hosts remain SDK-free and return `UnsupportedPlatform`
//! before foreign loading; GPU requests never fall back to CPU tensor execution.

extern crate fusion_pcu_core as fusion_pcu;

#[path = "conversion/conversion.rs"]
mod conversion;
pub use conversion::{
    MlxCheckedConversionPlan, MlxConversionHostBackend, MlxPreparedConversionHostKernel,
};

#[path = "discovery/discovery.rs"]
mod discovery;
pub use discovery::MlxDiscovery;
/// Executor identity used by operation-specific MLX implementation offers.
pub use discovery::EXECUTOR as MLX_EXECUTOR;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use discovery::{
    MlxMatmulRequest,
    MlxCheckedTensorRequest,
    MlxTensorBinaryRequest,
    MlxTensorIntegerRequest,
};

#[path = "host_kernel/host_kernel.rs"]
mod host_kernel;
#[rustfmt::skip]
pub use host_kernel::{
    MlxCheckedDivRemRolePlan,
    MlxCheckedDivRemRoleBackend,
    MlxPreparedDivRemRoleHostKernel,
    MlxCheckedDivRemPlan,
    MlxCheckedDivRemControl,
    MlxCheckedDivRemBackend,
    MlxPreparedDivRemHostKernel,
    MlxCheckedIntegerControl,
    MlxCheckedIntegerPlan,
    MlxCheckedIntegerBackend,
    MlxPreparedIntegerHostKernel,
    MlxPreparedHostKernel,
    MlxPreparedDispatchKernel,
    MlxDispatchOutputLayout,
    MlxDispatchCompletion,
    MlxCarrierPlan,
    MlxPreparedCarrierHostKernel,
    MlxCarrierControl,
    MlxCheckedUnaryPlan,
    MlxHostKernelError,
    MlxCheckedUnaryControl,
    MlxCheckedBinaryControl,
    MlxCheckedBinaryPlan,
    MlxCheckedBinaryBackend,
    MlxPreparedBinaryHostKernel,
    MlxBinaryInput,
};

#[path = "transport/transport.rs"]
mod transport;
#[rustfmt::skip]
pub use transport::{
    MlxTransportPlan,
    MlxTransportHostBackend,
    MlxTransportInput,
    MlxPreparedTransportHostKernel,
    MlxPreparedTransportKernel,
    MlxTransportCompletion,
};

#[cfg(feature = "benchmark-control")]
#[rustfmt::skip]
pub use transport::{
    MlxNativeTransportControl,
    MlxNativeTransportWorkload,
};

#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "runtime/runtime.rs"]
mod runtime;
#[rustfmt::skip]
pub use runtime::{
    MlxArray,
    MlxEncodedArray,
    MlxEncodedCompletion,
    MlxPreparedEncodedPrefix,
    MlxArrayResidency,
    MlxDeviceFacts,
    MlxError,
    MlxGpuBackend,
    MlxRuntime,
    MlxSession,
    MlxPreparedFloatConversion,
};
#[cfg(feature = "benchmark-control")]
pub use runtime::MlxNativeMatmulControl;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use runtime::{
    MlxPreparedMatmul,
    MlxPreparedProgram,
    MlxPreparedCheckedProgram,
    MlxPreparedTensorBinaryProgram,
    MlxPreparedTensorIntegerProgram,
    MlxCheckedProgramInput,
    MlxProgramInput,
};

#[cfg(feature = "tensor")]
#[path = "admission/admission.rs"]
mod admission;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use admission::{
    MlxMatmulPlan,
    MlxCheckedTensorPlan,
    MlxCheckedTensorBinaryPlan,
    MlxCheckedTensorIntegerPlan,
    MlxTensorAssessor,
};

#[cfg(feature = "view-census")]
#[rustfmt::skip]
pub use ffi::{
    MlxViewCallCensus,
    view_call_census,
    reset_view_call_census,
};

#[cfg(feature = "division-census")]
#[rustfmt::skip]
pub use ffi::{
    MlxDivRemCallCensus,
    div_rem_call_census,
    reset_div_rem_call_census,
};

#[cfg(feature = "carrier-census")]
#[rustfmt::skip]
pub use ffi::{
    MlxCarrierCallCensus,
    carrier_call_census,
    reset_carrier_call_census,
};

#[cfg(feature = "binary-census")]
#[rustfmt::skip]
pub use ffi::{
    MlxBinaryCallCensus,
    binary_call_census,
    reset_binary_call_census,
};

#[cfg(feature = "integer-census")]
#[rustfmt::skip]
pub use ffi::{
    MlxIntegerCallCensus,
    integer_call_census,
    reset_integer_call_census,
};

#[path = "composed/composed.rs"]
mod composed;
pub use composed::{MlxCheckedMapPlan, MlxCheckedMapEffect};
pub use composed::{MlxPreparedCheckedMapKernel, MlxCheckedMapCompletion};

#[rustfmt::skip]
pub use composed::{
    MlxCheckedMapInput,
    MlxComposedHostBackend,
    MlxPreparedCheckedMapHostKernel,
};

pub use runtime::MlxPreparedReluBackward;

#[cfg(feature = "tensor")]
#[path = "admission/backward_tensor/backward_tensor.rs"]
mod backward_tensor;
#[cfg(feature = "tensor")]
pub use backward_tensor::MlxCheckedTensorBackwardPlan;

#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedTensorBackwardProgram;

#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedStrictMatMul;

#[cfg(feature = "tensor")]
#[path = "admission/matmul_tensor/matmul_tensor.rs"]
mod matmul_tensor;
#[cfg(feature = "tensor")]
pub use matmul_tensor::MlxCheckedTensorMatMulPlan;
#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedTensorMatMulProgram;

#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedStrictSgd;

#[path = "dispatch_shape/dispatch_shape.rs"]
mod dispatch_shape;

#[cfg(feature = "tensor")]
#[path = "admission/sgd_tensor/sgd_tensor.rs"]
mod sgd_tensor;
#[cfg(feature = "tensor")]
pub use sgd_tensor::MlxCheckedTensorSgdPlan;
#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedTensorSgdProgram;

#[cfg(feature = "tensor")]
pub use runtime::{
    MlxSelectedNumericalTensorPlan, MlxPreparedSelectedNumericalTensorProgram,
    MlxSelectedNumericalTensorOperation,
};

#[cfg(feature = "tensor")]
pub use discovery::MlxSelectedNumericalTensorRequest;

#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedStrictMse;

#[cfg(feature = "tensor")]
#[path = "admission/mse_tensor/mse_tensor.rs"]
mod mse_tensor;
#[cfg(feature = "tensor")]
pub use mse_tensor::MlxCheckedTensorMsePlan;
#[cfg(feature = "tensor")]
pub use runtime::MlxPreparedTensorMseProgram;

#[cfg(feature = "tensor")]
#[path = "runtime/selected_graph/selected_graph.rs"]
mod selected_graph;
#[cfg(feature = "tensor")]
pub use selected_graph::{
    MlxSelectedTensorGraphPlan, MlxPreparedSelectedTensorGraph, MlxTensorGraphError,
};

#[cfg(feature = "tensor")]
pub use discovery::MlxSelectedTensorGraphRequest;
