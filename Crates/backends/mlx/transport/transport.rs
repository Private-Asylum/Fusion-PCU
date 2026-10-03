//! Bounded representation-preserving transport through MLX-owned integer arrays.
#[path = "plan/plan.rs"]
mod plan;
pub use plan::MlxTransportPlan;
#[path = "runtime/runtime.rs"]
mod runtime;
#[rustfmt::skip]
pub use runtime::{
    MlxPreparedTransportKernel,
    MlxTransportCompletion,
};
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

#[path = "host/host.rs"]
mod host;
#[rustfmt::skip]
pub use host::{
    MlxTransportHostBackend,
    MlxTransportInput,
    MlxPreparedTransportHostKernel,
};

#[cfg(feature = "benchmark-control")]
#[path = "control/control.rs"]
mod control;
#[cfg(feature = "benchmark-control")]
#[rustfmt::skip]
pub use control::{
    MlxNativeTransportControl,
    MlxNativeTransportWorkload,
};
