//! Matched fresh immutable MLX-owned result replacement, terminal prefix readback and old-owner drop.
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use std::time::Duration;
#[path = "support/support.rs"]
mod driver;
fn benchmark(criterion: &mut Criterion) {
    driver::run(criterion);
}
criterion_group! {name=benches;config=Criterion::default().sample_size(30).warm_up_time(Duration::from_millis(250)).measurement_time(Duration::from_secs(1));targets=benchmark}
criterion_main!(benches);
