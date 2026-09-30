//! Native SGD permissions with actual authored source, frozen graph and matched private control.
extern crate pcu_facade as fusion_pcu;

#[path = "../strict_matmul/activity.rs"]
mod activity;
#[cfg(feature = "allocation-census")]
#[path = "../support/allocations/allocations.rs"]
mod allocations;
#[path = "driver/driver.rs"]
mod driver;
#[path = "native/native.rs"]
mod native;
#[path = "oracle/oracle.rs"]
mod oracle;
#[cfg(not(feature = "allocation-census"))]
#[path = "paired/paired.rs"]
mod paired;
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

fn native_sgd(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group!(benches, native_sgd);
criterion_main!(benches);
