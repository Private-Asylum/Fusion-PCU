//! Genuine captured source, independent graph and direct native fresh-owner boundaries.
#[path = "source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
