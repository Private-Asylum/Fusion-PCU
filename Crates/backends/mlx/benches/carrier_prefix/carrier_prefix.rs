//! Retained full-capacity input to terminal host-prefix publication peers.
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
