//! Matched source/neutral/native semantic and caller-thread census boundary.
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use std::time::Duration;
#[path = "../support/low_precision/low_precision.rs"]
mod driver;
#[path = "source.rs"]
mod source;
fn benchmark(criterion: &mut Criterion) {
    driver::run(criterion, true);
}
criterion_group! {name = benches; config = Criterion::default().sample_size(30).confidence_level(0.95).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1)); targets = benchmark}
criterion_main!(benches);
