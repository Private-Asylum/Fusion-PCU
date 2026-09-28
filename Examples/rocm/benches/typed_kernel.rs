//! Composition for typed host and resident-device invocation benchmarks.

mod support;
#[path = "support/typed_kernel.rs"]
mod typed_kernel_support;

use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

fn typed_kernel(criterion: &mut Criterion) {
    if let Err(error) = typed_kernel_support::run(criterion) {
        panic!("typed kernel benchmark setup failed: {error}");
    }
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = typed_kernel
}
criterion_main!(benches);
