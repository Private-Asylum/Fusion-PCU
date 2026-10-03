//! Canonical genuine wide-carrier #[pcu] identity transport with physical host/resident native peers.
extern crate pcu_facade as fusion_pcu;
#[path = "../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
#[allow(unsafe_code, dead_code)]
// System allocator forwarding and shared reporting are feature-gated.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
fn wide_transport(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, wide_transport);
criterion_main!(benches);
