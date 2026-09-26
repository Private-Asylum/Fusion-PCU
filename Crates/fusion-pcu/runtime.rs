//! Backend-neutral runtime discovery, assessment, and activation contracts.
//!
//! These modules describe runtime-facing operations without prescribing provider selection
//! policy or owning backend sessions.

pub mod activation;
pub mod assessment;
pub mod discovery;
pub mod registry;

pub use activation::*;
pub use assessment::*;
pub use discovery::*;
pub use registry::*;
