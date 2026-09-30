//! Instrumented diagnostics, separate from ordinary Criterion measurements.
extern crate pcu_facade as fusion_pcu;

#[path = "../support/cpu_counters.rs"]
#[allow(dead_code)] // Unsupported-target queries are shared diagnostics.
mod cpu_counters;
#[path = "../support/insights.rs"]
#[allow(dead_code)] // Unsupported-target queries are shared diagnostic utilities.
mod insights;
#[path = "../support/train_step.rs"]
#[allow(dead_code)] // Shared native reference includes other benchmark routes.
mod native;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[path = "../support/train_insights_runner.rs"]
mod runner;
#[allow(dead_code)] // Shared benchmark configuration is unused by this diagnostic entry.
#[path = "../support/support.rs"]
mod support;
#[path = "../support/trace_markers.rs"]
mod trace_markers;
#[path = "../support/train_volume.rs"]
mod train_volume;
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
fn main() {
    runner::run().expect("paired insights diagnostic failed");
}

#[cfg(not(all(target_arch = "x86_64", target_os = "linux")))]
fn main() {
    panic!("these TSC/perf diagnostics require Linux x86_64; no substitute clock selected");
}
