//! Canonical six-format actual scalar locals and ordered stores with matched full-host native peers.
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
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
#[allow(dead_code)] // Shared exceptional encoding constants are outside this finite corpus.
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
const WORKLOAD_LABEL: &str = "rocm_ordered_float_maps";
fn ordered_float_maps(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, ordered_float_maps);
criterion_main!(benches);
