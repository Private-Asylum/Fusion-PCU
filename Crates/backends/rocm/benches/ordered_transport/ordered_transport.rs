//! Genuine ordered raw transport and matched two-output minimal native controls.
extern crate pcu_facade as fusion_pcu;
#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Feature-gated System allocator forwarding.
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
use criterion::{Criterion,criterion_group,criterion_main};
fn ordered_transport(criterion: &mut Criterion) {
    if !std::env::args().any(|arg| arg == "--test") {
        activity::guard();
    }
    driver::run(criterion);
}
criterion_group!(benches, ordered_transport);
criterion_main!(benches);
