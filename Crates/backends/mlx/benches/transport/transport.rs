//! Four physically matched fresh host-bank to two terminal host-prefix boundaries.

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
#[path = "support/support.rs"]
mod support;
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
