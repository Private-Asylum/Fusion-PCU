//! Independent synchronous Driver copies; no PCU emitter, graph or resource owner.
#[path = "ffi/ffi.rs"]
mod ffi;
pub use ffi::Control;
