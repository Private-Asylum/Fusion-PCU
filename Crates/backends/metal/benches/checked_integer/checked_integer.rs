//! Exact fourteen-width source/neutral/native matched fresh staging and publication controls.
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use std::time::Duration;
#[path = "support/support.rs"]
mod support;
fn benchmark(criterion: &mut Criterion) {
    support::run(criterion);
}
criterion_group! {name=benches;config=Criterion::default().sample_size(30).confidence_level(0.95).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1));targets=benchmark}
criterion_main!(benches);
