//! Annotated native SGD, explicit graph, and matched handwritten HIP boundaries.
extern crate pcu_facade as fusion_pcu;

mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../native_mse/allocations.rs"]
#[allow(unsafe_code)] // Optional caller-thread System allocator census.
mod allocations;
mod driver;
mod native;
mod oracle;
#[path = "../native_mse/selection.rs"]
mod selection;
mod source;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn native_sgd(criterion: &mut Criterion) {
    driver::run(criterion).expect("native SGD benchmark failed");
}

criterion_group!(benches, native_sgd);
criterion_main!(benches);
