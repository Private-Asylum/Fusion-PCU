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
//! - backend lowering
//! - runtime dispatch policy

#![cfg_attr(not(feature = "std"), no_std)]

#[path = "contract/contract.rs"]
pub mod contract;
pub mod core;
pub mod dispatch;
pub mod ir;
#[path = "model/model.rs"]
pub mod model;
pub mod validation;

pub use contract::*;
pub use dispatch::*;
