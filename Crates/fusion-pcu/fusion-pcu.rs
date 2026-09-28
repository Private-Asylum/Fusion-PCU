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
#[cfg(feature = "tensor")]
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
#[cfg(feature = "insights")]
pub mod insights;
pub mod ir;
pub mod map_validation;
#[path = "model/model.rs"]
pub mod model;
pub mod resource;
pub mod runtime;
pub mod scalar;
pub mod scalar_widen;
pub mod scalar_wrapping;
pub mod validation;

pub use contract::*;
#[rustfmt::skip]
pub use runtime::{
    activation,
    assessment,
    discovery,
    execution,
    registry,
};
pub use runtime::activation::*;
pub use runtime::assessment::*;
pub use runtime::discovery::*;
pub use runtime::device_kernel::*;
pub use runtime::execution::*;
pub use runtime::host_kernel::*;
pub use runtime::kernel::*;
pub use runtime::registry::*;
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
pub use dialect::*;
pub use memory::*;
pub use owned::*;
pub use scalar::*;
pub use scalar_widen::*;
pub use scalar_wrapping::*;
