//! Canonical genuine four-format #[pcu] checked arithmetic with physical host/resident native peers.
extern crate pcu_facade as fusion_pcu;
#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)]
// System allocator forwarding and shared reporting are feature-gated.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "../checked_low_precision/native/native.rs"]
mod native;
#[path = "../checked_low_precision/oracle/oracle.rs"]
#[allow(dead_code)] // Exceptional oracle constants are exercised by the source integration proof.
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "../checked_low_precision/source/source.rs"]
#[allow(dead_code)] // Grid/Strict entries are exercised by the source integration proof.
mod source;
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
fn portable_low_precision(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, portable_low_precision);
criterion_main!(benches);
