//! Ordinary source, frozen generated source, explicit IR and detached conversion controls.
#[path = "../../../backends/mlx/benches/compiled_matmul/support/activity/activity.rs"]
mod activity;
#[path = "../cpu_source/census/allocator/allocator.rs"]
mod allocator;
#[path = "driver/driver.rs"]
mod driver;
#[path = "../../../backends/mlx/conversion/tests/graph/graph.rs"]
mod graph;
#[path = "native/native.rs"]
mod native;
#[path = "source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: allocator::CountingAllocator = allocator::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion, criterion_group, criterion_main};
fn comparisons(criterion: &mut Criterion) {
    activity::gpu_idle_guard();
    driver::run::<65>(criterion);
    driver::run::<4096>(criterion);
    activity::gpu_post_guard();
    fusion_pcu::global::use_defaults().unwrap();
}
criterion_group!(benches, comparisons);
criterion_main!(benches);
