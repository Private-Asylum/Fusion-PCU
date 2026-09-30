//! Thin Criterion entry for checked F32/F64 `ReLU` dispatch and its generated HIP comparator.
//! A nonmatching Criterion filter (e.g. `__paired_diagnostics_only__`) skips statistical
//! measurements while still running the 36-triple interleaved diagnostics and preflights.

#[path = "support/checked_relu.rs"]
mod checked_relu_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn checked_relu(criterion: &mut Criterion) {
    checked_relu_support::run(criterion).expect("checked ReLU benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = checked_relu
}
criterion_main!(benches);
