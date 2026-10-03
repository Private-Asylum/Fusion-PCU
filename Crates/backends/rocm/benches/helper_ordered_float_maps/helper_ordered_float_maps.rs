//! Actual scalar helper locals and compound assignments with matched ordered-map work.
extern crate pcu_facade as fusion_pcu;
#[path = "activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)]
// System allocator forwarding and shared reporting are feature-gated.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "../ordered_float_maps/native/native.rs"]
mod native;
#[path = "../ordered_float_maps/oracle/oracle.rs"]
#[allow(dead_code)] // Shared exceptional encoding constants are outside this finite corpus.
mod oracle;
#[path = "raw/raw.rs"]
mod raw;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
const WORKLOAD_LABEL: &str = "rocm_helper_ordered_float_maps";
fn helper_ordered_float_maps(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, helper_ordered_float_maps);
criterion_main!(benches);
