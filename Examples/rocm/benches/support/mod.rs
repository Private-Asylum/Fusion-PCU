//! Shared bench configuration, device policy, and untimed setup instrumentation.

use std::{
    error::Error,
    time::{
        Duration,
        Instant,
    },
};

use criterion::Criterion;
use fusion_pcu_rocm::RocmDiscovery;

#[path = "../../selection.rs"]
pub mod selection;

/// Use a short but statistically useful run for expensive GPU routes.
pub fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(30)
        .warm_up_time(Duration::from_millis(300))
        .measurement_time(Duration::from_secs(2))
}

/// Apply the example consumer's device policy without hiding it in the PCU runtime.
pub fn selected_candidates(
    discovery: &RocmDiscovery,
) -> Result<Vec<selection::Candidate>, Box<dyn Error>> {
    selection::ranked_devices(discovery, selection::preferred_device()?, false)
}

/// Report one preparation or diagnostic operation outside Criterion's repeated sample loop.
pub fn cold_once<T, E>(label: &str, f: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
    let start = Instant::now();
    let result = f();
    println!("{label}: {:?}", start.elapsed());
    result
}
