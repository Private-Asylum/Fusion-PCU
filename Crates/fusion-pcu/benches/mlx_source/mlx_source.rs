//! Actual ordinary source call versus captured source, neutral graph and native MLX peer.
#[path = "../../../backends/mlx/benches/compiled_matmul/support/activity/activity.rs"]
mod activity;
#[cfg(feature = "mlx-view-census")]
#[path = "census/census.rs"]
mod census;
#[path = "../../tests/opaque/source.rs"]
#[allow(dead_code)] // Negative contract fixtures are exercised by the native integration test.
mod source;
#[path = "support/support.rs"]
#[cfg(not(feature = "mlx-view-census"))]
mod support;
#[path = "views/views.rs"]
mod views;
#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    Criterion,
};
fn benchmark(criterion: &mut Criterion) {
    #[cfg(not(feature = "mlx-view-census"))]
    {
        support::run::<16>(criterion);
        support::run::<128>(criterion);
    }
    views::run::<2>(criterion);
    views::run::<16>(criterion);
    views::run::<128>(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
