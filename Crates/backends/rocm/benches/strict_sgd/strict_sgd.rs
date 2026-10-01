//! Canonical authored source / explicit graph / native checker checked SGD benchmark.
extern crate pcu_facade as fusion_pcu;
#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../strict_matmul/allocations.rs"]
#[allow(unsafe_code, dead_code)] // Exact System forwarding; shared reporting helper is unused here.
mod allocations;
#[path = "correctness/correctness.rs"]
mod correctness;
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
fn strict_sgd(criterion: &mut Criterion) {
    driver::run(criterion).expect("strict SGD benchmark");
}
criterion_group!(benches, strict_sgd);
criterion_main!(benches);
