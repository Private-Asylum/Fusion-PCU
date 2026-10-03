//! Full-capacity encoded binary input to completed private output, terminal read and Drop peers.
#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};
#[path = "support/support.rs"]
mod support;
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
