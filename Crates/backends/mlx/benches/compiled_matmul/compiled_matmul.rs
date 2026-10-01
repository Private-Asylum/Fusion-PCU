//! Canonical authentic captured-source preparation/replay, explicit graph, and native MLX controls.
mod source;
#[path = "support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
};

criterion_group!(benches, support::comparison);
criterion_main!(benches);
