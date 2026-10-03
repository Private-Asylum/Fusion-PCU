//! Matching captured-source, explicit-graph and native immutable owner boundaries.
#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
};
#[path = "support/support.rs"]
mod support;
criterion_group!(benches, support::run);
criterion_main!(benches);
