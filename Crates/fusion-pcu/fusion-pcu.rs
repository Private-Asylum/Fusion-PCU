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

pub mod builder;
#[path = "contract/contract.rs"]
pub mod contract;
pub mod core;
pub mod dialect;
pub use dialect::builder as dialect_builder;
pub mod dispatch;
pub mod ir;
pub mod map_validation;
#[path = "model/model.rs"]
pub mod model;
pub mod resource;
pub mod runtime;
pub mod scalar;
pub mod validation;

pub use contract::*;
pub use runtime::{
    activation,
    assessment,
    discovery,
    registry,
};
pub use runtime::activation::*;
pub use runtime::assessment::*;
pub use runtime::discovery::*;
pub use runtime::registry::*;
pub use resource::{
    borrowed,
    memory,
    owned,
};
pub use borrowed::*;
pub use builder::*;
pub use dispatch::*;
pub use map_validation::*;
pub use dialect::*;
pub use memory::*;
pub use owned::*;
pub use scalar::*;
