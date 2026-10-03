//! Cold-frozen checked arithmetic; native byte access is isolated in ffi.
#[path = "ffi/ffi.rs"]
mod ffi;
#[rustfmt::skip]
pub(super) use ffi::{
    Executable,
    prepare,
};
