//! Optional PCU compiler backends.

#[cfg(feature = "backend-spirv")]
#[path = "spirv/spirv.rs"]
pub mod spirv;

#[cfg(feature = "backend-spirv")]
pub use spirv::*;
