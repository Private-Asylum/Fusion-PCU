//! Genuine immutable wide encodings, explicit graph and independent Driver transport.
extern crate pcu_facade as fusion_pcu;
#[path = "../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Separate census allocator forwarding and shared reports.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/native.rs"]
mod native;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn raw_float_tensor_producers(c: &mut Criterion) {
    driver::run(c);
}
criterion_group!(benches, raw_float_tensor_producers);
criterion_main!(benches);
