//! Canonical authored source / explicit graph / independent native native F64 optimizer benchmark.
extern crate pcu_facade as fusion_pcu;
#[path = "../support/activity/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Exact System forwarding; shared reporting helper is unused here.
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
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn native_sgd_f64(criterion: &mut Criterion) {
    driver::run(criterion).expect("native_sgd_f64 benchmark");
}
criterion_group!(benches, native_sgd_f64);
criterion_main!(benches);
