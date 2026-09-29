//! Thin Criterion entry for the fresh-output owned `ReLU` comparison.

#[path = "support/owned_relu.rs"]
mod owned_relu_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_relu(criterion: &mut Criterion) {
    owned_relu_support::run(criterion).expect("owned-ReLU benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_relu
}
criterion_main!(benches);
