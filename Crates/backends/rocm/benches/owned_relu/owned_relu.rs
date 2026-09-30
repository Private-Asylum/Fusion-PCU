//! Thin Criterion entry for the fresh-output owned `ReLU` comparison.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/owned_relu.rs"]
mod owned_relu_support;
#[path = "../support/support.rs"]
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
