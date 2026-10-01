//! SPIR-V lowering backend for PCU IR.
//!
//! This module is a compiler target, not a device runner. Vulkan, `OpenCL`, or any other runtime
//! may consume the generated words elsewhere; this backend only lowers PCU dispatch IR into
//! backend-neutral SPIR-V module bytes.

#![no_std]

extern crate fusion_pcu_core as fusion_pcu;
extern crate alloc;

#[cfg(test)]
extern crate std;

#[path = "bit_map/bit_map.rs"]
mod bit_map;
#[path = "error/error.rs"]
pub mod error;
#[path = "lower/lower.rs"]
pub mod lower;
#[path = "module/module.rs"]
pub mod module;
#[path = "sink/sink.rs"]
pub mod sink;
#[path = "types/types.rs"]
pub mod types;

pub use error::*;
pub use lower::*;
pub use module::*;
pub use sink::*;
pub use types::*;
pub use bit_map::*;
