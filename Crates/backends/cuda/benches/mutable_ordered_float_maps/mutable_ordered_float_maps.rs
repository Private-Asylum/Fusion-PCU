//! Actual mutable source locals with the original prepared/native ordered-map workload.
extern crate pcu_facade as fusion_pcu;
#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
#[allow(unsafe_code, dead_code)]
// System allocator forwarding and shared reporting are feature-gated.
mod allocations;
#[path = "../ordered_float_maps/driver/driver.rs"]
mod driver;
#[path = "../ordered_float_maps/native/native.rs"]
mod native;
#[path = "../ordered_float_maps/oracle/oracle.rs"]
#[allow(dead_code)] // Shared exceptional encoding constants are outside this finite corpus.
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
const WORKLOAD_LABEL: &str = "cuda_mutable_ordered_float_maps";
fn mutable_ordered_float_maps(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, mutable_ordered_float_maps);
criterion_main!(benches);
