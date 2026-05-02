//! SPIR-V lowering backend for PCU IR.
//!
//! This module is a compiler target, not a device runner. Vulkan, OpenCL, or any other runtime
//! may consume the generated words elsewhere; this backend only lowers PCU dispatch IR into
//! backend-neutral SPIR-V module bytes.

pub mod error;
pub mod lower;
pub mod module;
pub mod sink;
pub mod types;

pub use error::*;
pub use lower::*;
pub use module::*;
pub use sink::*;
pub use types::*;
