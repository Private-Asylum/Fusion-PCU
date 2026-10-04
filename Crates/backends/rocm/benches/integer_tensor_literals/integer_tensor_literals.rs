//! Exact graph producers beside closest genuine source and handwritten SDK arithmetic.
extern crate pcu_facade as fusion_pcu;
#[path = "activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)]
// Separate allocator forwarding; common reporting helpers are shared.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
#[allow(dead_code)] // This producer target reuses exact integer formats, not staged Sub goldens.
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
#[allow(dead_code)] // Owner consumption is a separate source/lifetime companion.
mod source;
use criterion::{Criterion, criterion_group, criterion_main};
fn integer_tensor_literals(c: &mut Criterion) {
    driver::run(c);
}
criterion_group!(benches, integer_tensor_literals);
criterion_main!(benches);
