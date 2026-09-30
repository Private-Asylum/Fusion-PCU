//! Native loss source, owned graph and typed rocBLAS synchronous boundary comparisons.
extern crate pcu_facade as fusion_pcu;

mod activity;
#[cfg(feature = "allocation-census")]
#[allow(unsafe_code)] // Optional allocator delegates valid layouts to System.
mod allocations;
mod driver;
mod native;
mod oracle;
mod selection;
mod source;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn native_mse(criterion: &mut Criterion) {
    driver::run(criterion).expect("native MSE benchmark failed");
}
criterion_group!(benches, native_mse);
criterion_main!(benches);
