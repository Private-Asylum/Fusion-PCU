//! Measure selected owners in a consuming source call against native in-place Add.

#[path = "support/owned_selected_owners.rs"]
mod owned_selected_owners_support;
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_selected_owners(criterion: &mut Criterion) {
    owned_selected_owners_support::run(criterion).expect("selected-owner benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_selected_owners
}
criterion_main!(benches);
