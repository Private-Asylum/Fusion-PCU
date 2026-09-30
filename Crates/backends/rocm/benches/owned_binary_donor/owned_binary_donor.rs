//! Thin Criterion entry for terminal consuming binary donor reuse.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/owned_binary_donor.rs"]
mod owned_binary_donor_support;
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_binary_donor(criterion: &mut Criterion) {
    owned_binary_donor_support::run(criterion).expect("owned binary donor benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_binary_donor
}
criterion_main!(benches);
