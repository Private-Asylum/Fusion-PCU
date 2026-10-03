//! Actual requested-Portable unary source/prepared/native full-capacity peers.
#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};
#[path = "source/source.rs"]
mod source;
#[path = "../unary_prefix/support/support.rs"]
mod support;
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
