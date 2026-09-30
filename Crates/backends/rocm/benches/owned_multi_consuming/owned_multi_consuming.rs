//! Compare by-value source owners, raw owned PCU, and native HIP binary addition.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/owned_multi_consuming.rs"]
mod owned_multi_consuming_support;
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn owned_multi_consuming(criterion: &mut Criterion) {
    owned_multi_consuming_support::run(criterion).expect("owned multi-consuming benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = owned_multi_consuming
}
criterion_main!(benches);
