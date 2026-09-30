//! Compare source checked integer addition with a matching native HIP kernel.

#[path = "support/owned_checked_integer.rs"]
mod owned_checked_integer_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_checked_integer(criterion: &mut Criterion) {
    owned_checked_integer_support::run(criterion).expect("owned checked integer benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_checked_integer
}
criterion_main!(benches);
