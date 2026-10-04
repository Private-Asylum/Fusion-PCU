//! Independent composed integer arithmetic beside genuine source and explicit typed IR.
extern crate pcu_facade as fusion_pcu;
#[path = "../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Allocator forwarding is isolated to separate census builds.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "ir/ir.rs"]
mod ir;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
use criterion::{Criterion, criterion_group, criterion_main};
fn composed_integer(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, composed_integer);
criterion_main!(benches);
