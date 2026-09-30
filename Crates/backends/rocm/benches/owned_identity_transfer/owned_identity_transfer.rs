//! Thin Criterion entry for consuming identity ownership transfer.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/owned_identity_transfer.rs"]
mod owned_identity_transfer_support;
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_identity_transfer(criterion: &mut Criterion) {
    owned_identity_transfer_support::run(criterion)
        .expect("owned identity-transfer benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_identity_transfer
}
criterion_main!(benches);
