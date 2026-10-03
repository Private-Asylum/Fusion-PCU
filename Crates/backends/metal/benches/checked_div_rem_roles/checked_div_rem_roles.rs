//! Matched annotated role source, detached IR, direct native and joint host publication.
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use std::time::Duration;
#[path = "support/support.rs"]
mod support;
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group! {name=benches;config=Criterion::default().sample_size(30).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1));targets=benchmark}
criterion_main!(benches);
