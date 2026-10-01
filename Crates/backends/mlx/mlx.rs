//! Explicit delegated Apple silicon MLX GPU tensor runtime for Fusion PCU.
//!
//! MLX is selected independently from native Metal. The initial tensor contract admits only
//! dense F32 matrix products with explicit native compound and optimized precision permissions.
//! Checked defaults, strict arithmetic, portable arithmetic and unproved precision policies
//! reject before device work. Other hosts remain SDK-free and return `UnsupportedPlatform`
//! before foreign loading; GPU requests never fall back to CPU tensor execution.

extern crate fusion_pcu_core as fusion_pcu;

#[path = "discovery/discovery.rs"]
mod discovery;
pub use discovery::MlxDiscovery;
#[cfg(feature = "tensor")]
pub use discovery::MlxMatmulRequest;

#[path = "ffi/ffi.rs"]
mod ffi;
#[cfg(feature = "c-api-evaluation")]
#[doc(hidden)]
pub use ffi::c_api_evaluation;
#[path = "runtime/runtime.rs"]
mod runtime;
#[rustfmt::skip]
pub use runtime::{
    MlxArray,
    MlxArrayResidency,
    MlxDeviceFacts,
    MlxError,
    MlxGpuBackend,
    MlxRuntime,
    MlxSession,
};
#[cfg(feature = "benchmark-control")]
pub use runtime::MlxNativeMatmulControl;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use runtime::{
    MlxPreparedMatmul,
    MlxPreparedProgram,
};

#[cfg(feature = "tensor")]
#[path = "admission/admission.rs"]
mod admission;
#[cfg(feature = "tensor")]
#[rustfmt::skip]
pub use admission::{
    MlxMatmulPlan,
    MlxTensorAssessor,
};
