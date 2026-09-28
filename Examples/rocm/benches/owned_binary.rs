//! Compare source, raw owned PCU, and native HIP binary tensor addition.

#[path = "support/owned_binary.rs"]
mod owned_binary_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_binary(criterion: &mut Criterion) {
    owned_binary_support::run(criterion).expect("owned binary benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_binary
}
criterion_main!(benches);
