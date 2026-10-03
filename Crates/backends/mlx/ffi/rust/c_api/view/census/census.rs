//! Benchmark-only caller-thread view protocol counts; never native allocation totals.
use std::cell::Cell;
/// Counts audited C API call attempts made specifically by the immutable F32 view protocol.
///
/// Cached descriptor returns bypass this protocol. Metadata validation outside this protocol,
/// MLX internal allocation, native primitives, compiler/driver work and other operations are
/// intentionally excluded. Counters saturate and are confined to the caller's current thread.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlxViewCallCensus {
    pub input_clone_calls: u64,
    pub construct_calls: u64,
    pub eval_calls: u64,
    pub synchronize_calls: u64,
    pub wait_calls: u64,
    pub available_calls: u64,
    pub shared_storage_validation_calls: u64,
    pub input_release_calls: u64,
}
impl MlxViewCallCensus {
    const ZERO: Self = Self {
        input_clone_calls: 0,
        construct_calls: 0,
        eval_calls: 0,
        synchronize_calls: 0,
        wait_calls: 0,
        available_calls: 0,
        shared_storage_validation_calls: 0,
        input_release_calls: 0,
    };
}
thread_local! {static COUNTS:Cell<MlxViewCallCensus>=const{Cell::new(MlxViewCallCensus::ZERO)};}
#[derive(Clone, Copy)]
pub enum Call {
    Clone,
    Construct,
    Eval,
    Synchronize,
    Wait,
    Available,
    Validate,
    Release,
}
pub fn record(call: Call) {
    COUNTS.with(|counter| {
        let mut counts = counter.get();
        let value = match call {
            Call::Clone => &mut counts.input_clone_calls,
            Call::Construct => &mut counts.construct_calls,
            Call::Eval => &mut counts.eval_calls,
            Call::Synchronize => &mut counts.synchronize_calls,
            Call::Wait => &mut counts.wait_calls,
            Call::Available => &mut counts.available_calls,
            Call::Validate => &mut counts.shared_storage_validation_calls,
            Call::Release => &mut counts.input_release_calls,
        };
        *value = value.saturating_add(1);
        counter.set(counts);
    });
}
/// Resets this caller thread's benchmark-only view protocol counters.
pub fn reset_view_call_census() {
    COUNTS.with(|counts| counts.set(MlxViewCallCensus::ZERO));
}
/// Returns this caller thread's benchmark-only view protocol call counts.
#[must_use]
pub fn view_call_census() -> MlxViewCallCensus {
    COUNTS.with(Cell::get)
}
