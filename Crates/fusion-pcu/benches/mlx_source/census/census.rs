//! Untimed caller heap/scorer and narrowly audited view protocol measurements.
#[path = "../../../../backends/mlx/benches/checked_unary/census/census.rs"]
#[allow(dead_code)]
// Reuse the transparent allocator; its one-call printer is not this 64-call report.
mod allocator;
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuDeviceDescriptor,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    reset_view_call_census,
    view_call_census,
};
use std::cell::Cell;

thread_local! {
    static SCORES: Cell<usize> = const { Cell::new(0) };
}

pub fn score_device(device: &PcuDeviceDescriptor<'_>, memory: u64) -> i128 {
    SCORES.with(|counter| counter.set(counter.get().saturating_add(1)));
    global::default_device_score(device, memory)
}

pub fn report(label: &str, first_input_views: bool, mut execute: impl FnMut(usize)) {
    let scores_before = SCORES.with(Cell::get);
    reset_view_call_census();
    let ((), heap) = allocator::measure(|| {
        for iteration in 0..64 {
            execute(iteration % 3);
        }
    });
    let views = view_call_census();
    let scores_after = SCORES.with(Cell::get);
    assert_eq!(
        scores_after, scores_before,
        "warm source must not invoke consumer scoring"
    );
    // Every call creates an output descriptor view; fresh inputs additionally
    // create two first views. All retained input banks were primed before capture.
    let expected = if first_input_views { 192 } else { 64 };
    for count in [
        views.input_clone_calls,
        views.construct_calls,
        views.eval_calls,
        views.synchronize_calls,
        views.wait_calls,
        views.available_calls,
        views.shared_storage_validation_calls,
        views.input_release_calls,
    ] {
        assert_eq!(
            count, expected,
            "unexpected first/cached view protocol at {label}"
        );
    }
    // Allocation capture has ended. Formatting and report allocations are excluded.
    eprintln!(
        "MLX view census/{label}: calls=64 alloc={} realloc={} dealloc={} requested_bytes={} scorer_before={} scorer_after={} first_view_protocol={views:?} (caller Rust heap and view-specific C API attempts only; excludes native allocation totals, other SDK calls and latency)",
        heap.alloc_calls,
        heap.realloc_calls,
        heap.dealloc_calls,
        heap.requested_bytes,
        scores_before,
        scores_after,
    );
}
