//! Same-SDK safety-patched official C versus owned C++ diagnostic.

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
};
#[path = "../compiled_matmul/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;

criterion_group!(comparison, support::comparison);
criterion_main!(comparison);
