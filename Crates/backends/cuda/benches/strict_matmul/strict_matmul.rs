//! Canonical strict source, explicit graph and matched native checker measurements.
extern crate pcu_facade as fusion_pcu;

mod activity;
#[cfg(feature = "allocation-census")]
#[allow(unsafe_code)] // Opt-in allocator forwards valid caller layouts to System.
mod allocations;
mod correctness;
mod driver;
mod native;
mod oracle;
mod selection;
mod source;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn strict_matmul(criterion: &mut Criterion) {
    driver::run(criterion).expect("strict MatMul benchmark failed");
}

criterion_group!(benches, strict_matmul);
criterion_main!(benches);
