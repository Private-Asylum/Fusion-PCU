//! Entry to the isolated Rust foreign-runtime boundary.

#[path = "rust/rust.rs"]
mod rust;

#[rustfmt::skip]
pub use rust::{
    Api,
    Array,
    Session,
    as_f32,
    as_f32_mut,
};
#[cfg(feature = "tensor")]
pub use rust::PreparedMatmul;
