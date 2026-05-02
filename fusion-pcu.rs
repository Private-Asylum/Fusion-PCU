//! Canonical Fusion PCU contract and IR crate.
//!
//! `fusion-pcu` owns:
//! - generic PCU contract law
//! - generic execution-profile IR law
//! - backend-neutral validation vocabulary
//! - optional target-IR compiler backends
//! - optional hosted/device runners behind explicit feature gates
//!
//! It intentionally does not own:
//! - platform/provider selection
//! - transport protocol glue
//! - runtime dispatch policy or device orchestration
//! - graphics pipeline or shader-stage composition

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(test)]
extern crate std;

#[cfg(feature = "backend-spirv")]
#[path = "backends/backends.rs"]
pub mod backends;
#[path = "contract/contract.rs"]
pub mod contract;
pub mod core;
pub mod dispatch;
pub mod ir;
#[path = "model/model.rs"]
pub mod model;
#[cfg(feature = "runner-vulkan")]
#[path = "runner/runner.rs"]
pub mod runner;
pub mod validation;

pub use contract::*;
pub use dispatch::*;
