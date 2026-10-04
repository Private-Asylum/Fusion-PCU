//! Genuine immutable source producers beside explicit graph and handwritten SDK arithmetic.
extern crate pcu_facade as fusion_pcu;
#[path = "../integer_tensor_literals/activity/activity.rs"]
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
#[allow(dead_code)] // Midpoint helpers also serve the complete low-format golden corpus.
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
#[allow(dead_code)] // Owner consumption is a separate source/lifetime companion.
mod source;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn low_tensor_producers(c: &mut Criterion) {
    driver::run(c);
}
criterion_group!(benches, low_tensor_producers);
criterion_main!(benches);
