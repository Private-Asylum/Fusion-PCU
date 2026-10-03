//! Matching fourteen-width captured, graph, native and ordinary source owner boundaries.
#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
};
#[path = "support/support.rs"]
mod support;
criterion_group!(benches, support::run);
criterion_main!(benches);
