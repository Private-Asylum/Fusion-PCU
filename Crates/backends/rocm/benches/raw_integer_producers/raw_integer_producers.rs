//! Genuine immutable integer encodings, explicit graph and independent HIP transport.
extern crate pcu_facade as fusion_pcu;
#[path = "../integer_tensor_literals/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Separate census allocator forwarding and shared reports.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "../raw_float_tensor_producers/native/native.rs"]
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
fn raw_integer_producers(c: &mut Criterion) {
    driver::run(c);
}
criterion_group!(benches, raw_integer_producers);
criterion_main!(benches);
