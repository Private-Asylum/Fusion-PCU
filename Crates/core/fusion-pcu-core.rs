//! Canonical Fusion PCU contract and IR crate.
//!
//! `fusion-pcu` owns:
//! - generic PCU contract law
//! - generic execution-profile IR law
//! - backend-neutral validation vocabulary
//!
//! It intentionally does not own:
//! - platform/provider selection
//! - transport protocol glue
//! - runtime dispatch policy or device orchestration
//! - graphics pipeline or shader-stage composition

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate std;
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod builder;
#[path = "contract/contract.rs"]
pub mod contract;
pub mod core;
pub mod dialect;
mod insight_macros;
pub use dialect::builder as dialect_builder;
pub mod dispatch;
pub mod function;
#[path = "hardware/hardware.rs"]
pub mod hardware;
#[cfg(feature = "insights")]
pub mod insights;
pub mod ir;
pub mod map_validation;
#[path = "model/model.rs"]
pub mod model;
#[path = "numerical/numerical.rs"]
pub mod numerical;
pub mod resource;
pub mod runtime;
pub mod scalar;
pub mod scalar_checked;
pub mod scalar_checked_conversion;
pub mod scalar_checked_float;
pub mod scalar_clamped;
pub mod scalar_float_underflow;
pub mod scalar_widen;
pub mod scalar_wrapping;
pub mod validation;

pub use contract::*;
pub use hardware::*;
#[rustfmt::skip]
pub use runtime::{
    activation,
    assessment,
    discovery,
    execution,
    registry,
    scalar_lowering,
};
pub use runtime::activation::*;
pub use runtime::assessment::*;
pub use runtime::discovery::*;
#[cfg(feature = "alloc")]
pub use runtime::device_tensor::*;
#[cfg(feature = "alloc")]
pub use runtime::owned_shape::*;
pub use runtime::device_kernel::*;
pub use runtime::execution::*;
pub use runtime::host_kernel::*;
pub use runtime::kernel::*;
pub use runtime::registry::*;
pub use runtime::scalar_lowering::*;
#[rustfmt::skip]
pub use resource::{
    borrowed,
    memory,
    owned,
};
pub use borrowed::*;
pub use builder::*;
pub use dispatch::*;
pub use function::*;
pub use map_validation::*;
pub use numerical::*;
pub use dialect::*;
pub use memory::*;
pub use owned::*;
pub use scalar::*;
pub use scalar_checked::*;
pub use scalar_clamped::*;
pub use scalar_checked_float::*;
pub use scalar_checked_conversion::*;
pub use scalar_float_underflow::*;
pub use scalar_widen::*;
pub use scalar_wrapping::*;
