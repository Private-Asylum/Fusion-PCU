//! Backend-neutral runtime discovery, assessment, and activation contracts.
//!
//! These modules describe runtime-facing operations without prescribing provider selection
//! policy or owning backend sessions.

pub mod activation;
pub mod assessment;
pub mod device_kernel;
pub mod discovery;
pub mod execution;
pub mod host_kernel;
pub mod kernel;
pub mod registry;

pub use activation::*;
pub use assessment::*;
pub use discovery::*;
pub use device_kernel::*;
pub use execution::*;
pub use host_kernel::*;
pub use kernel::*;
pub use registry::*;
