//! Matched ordinary/captured actual source/independent graph/direct native fresh binary-owner peers.
#[path = "source/source.rs"]
mod source;
#[path = "support/support.rs"]
mod support;
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
