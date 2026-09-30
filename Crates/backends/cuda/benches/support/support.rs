//! Benchmark setup, native plumbing and correctness oracles live outside the bench entry point.
#[path = "activity/activity.rs"]
pub mod activity;
#[cfg(feature = "allocation-census")]
#[path = "allocations/allocations.rs"]
pub mod allocations;
#[path = "discovery/discovery.rs"]
pub mod discovery;
#[path = "fixture/fixture.rs"]
pub mod fixture;
#[path = "native/native.rs"]
pub mod native;
#[path = "oracle/oracle.rs"]
pub mod oracle;
#[path = "source/source.rs"]
pub mod source;

pub fn cold<T>(label: &str, operation: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = operation();
    eprintln!("diagnostic/{label}/process_wall={:?}", start.elapsed());
    result
}
