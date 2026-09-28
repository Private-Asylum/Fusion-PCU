//! Independent strict 4x2 jobs with fresh inputs at increasing submission volumes.

#[path = "support/train_step.rs"]
#[allow(dead_code)] // Shared reference includes fixed-input and profiled routes.
mod native;
#[path = "support/train_volume_runner.rs"]
mod runner;
mod support;
#[path = "support/train_volume.rs"]
mod train_volume;

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};

fn bench(criterion: &mut Criterion) {
    runner::run(criterion).expect("strict training-volume benchmark failed");
}

criterion_group! {
    name = benches;
    config = support::criterion_config().sample_size(10);
    targets = bench
}
criterion_main!(benches);
