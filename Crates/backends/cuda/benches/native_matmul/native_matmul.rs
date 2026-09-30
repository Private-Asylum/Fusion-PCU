//! Explicit vendor compound permissions: actual annotated source, graph and matched cuBLAS work.
extern crate pcu_facade as fusion_pcu;

#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
mod allocations;
mod driver;
#[path = "heavy/heavy.rs"]
mod heavy;
mod native;
#[path = "../strict_matmul/oracle.rs"]
#[expect(
    dead_code,
    reason = "Shared nominal oracle's strict exceptional helpers are outside this native numerical contract."
)]
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
mod source;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn native_matmul(criterion: &mut Criterion) {
    driver::run(criterion);
}

criterion_group!(benches, native_matmul);
criterion_main!(benches);
