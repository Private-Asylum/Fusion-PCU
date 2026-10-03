//! Pure cold checked composition candidate; execution remains unadvertised.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::{MetalCheckedMapPlan, MetalCheckedMapEffect};

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

#[path = "shader/shader.rs"]
mod shader;

#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
pub mod control;
