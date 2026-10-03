//! Exact unary execution; initialized carrier byte access stays isolated in ffi.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
pub(super) use ffi::{Executable,prepare};
