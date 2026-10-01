//! Canonical captured-source/graph replay beside native retained, frontend, and public-C compile.
mod source;
#[path = "support/support.rs"]
mod support;

#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};

use fusion_pcu_mlx::MlxRuntime;

fn comparison(criterion: &mut Criterion) {
    support::gpu_idle_guard();
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    assert_eq!(runtime.version(), "0.32.3");
    if std::env::var_os("PCU_MLX_PAIRED_DIAGNOSTIC").is_some() {
        support::paired_run(&session);
        support::gpu_post_guard();
        return;
    }
    support::benchmark_size::<32>(criterion, &session);
    support::benchmark_size::<128>(criterion, &session);
    support::gpu_post_guard();
}

criterion_group!(benches, comparison);
criterion_main!(benches);
