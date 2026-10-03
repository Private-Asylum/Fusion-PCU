//! Pure cold checked composition candidate; execution remains unadvertised.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::{MlxCheckedMapPlan, MlxCheckedMapEffect};
#[path = "runtime/runtime.rs"]
mod runtime;
#[path = "shader/shader.rs"]
mod shader;
pub use runtime::{MlxPreparedCheckedMapKernel, MlxCheckedMapCompletion};

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

#[path = "host/host.rs"]
mod host;
#[rustfmt::skip]
pub use host::{
    MlxCheckedMapInput,
    MlxComposedHostBackend,
    MlxPreparedCheckedMapHostKernel,
};

#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
mod control;
