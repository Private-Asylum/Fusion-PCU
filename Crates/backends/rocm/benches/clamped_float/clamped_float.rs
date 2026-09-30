//! Matched complete-host-boundary comparison for explicit clamp continuation.
extern crate pcu_facade as fusion_pcu;

#[path = "../support/clamped_float.rs"]
mod clamped_float_support;
#[path = "../support/support.rs"]
mod support;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn clamped_float(criterion: &mut Criterion) {
    clamped_float_support::run(criterion).expect("clamp benchmark failed");
}
criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = clamped_float
}
criterion_main!(benches);
