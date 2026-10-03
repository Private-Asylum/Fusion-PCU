//! Canonical genuine owned #[pcu] source, explicit graph and identical native low tensor work.
extern crate pcu_facade as fusion_pcu;
#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Exact System forwarding and unused common reporting helper.
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
#[allow(dead_code)]
// Shared legacy verifier allocates sentinel vectors; this benchmark caches its sentinel.
mod oracle;
#[path = "../strict_matmul/selection.rs"]
mod selection;
#[path = "source/source.rs"]
#[allow(dead_code)] // Consumption/discard fixtures use the remaining real source functions.
mod source;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn low_tensor(criterion: &mut Criterion) {
    driver::run(criterion).expect("low tensor benchmark");
}
criterion_group!(benches, low_tensor);
criterion_main!(benches);
