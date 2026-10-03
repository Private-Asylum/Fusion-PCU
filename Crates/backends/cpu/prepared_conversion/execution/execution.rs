//! Cold-frozen checked conversion, with native representation access in bytes.
#[path = "bytes/bytes.rs"]
mod bytes;
#[rustfmt::skip]
pub(super) use bytes::{Executable,prepare};
