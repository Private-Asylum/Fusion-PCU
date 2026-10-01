//! Public facade for the backend-neutral Fusion PCU core and optional providers.
//!
//! Most consumers can keep importing `fusion_pcu::*`. The dependency-free contract and IR
//! vocabulary lives in [`fusion_pcu_core`], while providers are exposed behind feature flags.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub use fusion_pcu_core::*;
pub use fusion_pcu_macros::*;

#[cfg(feature = "rocm")]
pub use fusion_pcu_rocm as rocm;
#[cfg(feature = "cuda")]
pub use fusion_pcu_cuda as cuda;

// `global` is supplied by the facade's provider-selection layer.
pub mod global;

#[doc(hidden)]
#[rustfmt::skip]
pub use global::{
    FixedArrayShape,
    FixedMatrixShape,
    PcuArgumentError,
    PcuCallArgument,
    PcuReadStorage,
    PcuResidentBufferOwner,
    PcuSourceShape,
    PcuTensorInput,
    PcuTensorSource,
    PcuWriteStorage,
    ScalarShape,
    SliceShape,
};

#[rustfmt::skip]
pub use global::{
    PcuExecutionError,
    PcuTensor,
};

#[cfg(feature = "metal")]
pub use fusion_pcu_metal as metal;

#[cfg(feature = "vulkan")]
pub use fusion_pcu_vulkan as vulkan;

#[cfg(feature = "cpu")]
pub use fusion_pcu_cpu as cpu;
