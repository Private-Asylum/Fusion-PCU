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

pub mod activation;
pub mod assessment;
pub mod builder;
#[path = "contract/contract.rs"]
pub mod contract;
pub mod core;
pub mod dialect;
pub mod dialect_builder;
pub mod discovery;
pub mod dispatch;
pub mod ir;
pub mod memory;
#[path = "model/model.rs"]
pub mod model;
pub mod owned;
pub mod registry;
pub mod validation;

pub use contract::*;
pub use activation::*;
pub use assessment::*;
pub use builder::*;
pub use dispatch::*;
pub use dialect::*;
pub use dialect_builder::*;
pub use discovery::*;
pub use memory::*;
pub use owned::*;
pub use registry::*;
