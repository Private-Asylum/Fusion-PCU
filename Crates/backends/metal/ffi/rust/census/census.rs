//! Explicit audited caller-thread Metal API attempts; never native heap allocation totals.
use std::cell::Cell;
/// Narrow explicit protocol counts. Internal SDK/driver calls, ARC retains/releases,
/// autorelease pools, getters and all native allocator activity are excluded.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalApiCallCensus {
    pub session_open_attempts: u64,
    pub shader_compile_attempts: u64,
    pub buffer_create_attempts: u64,
    pub command_buffer_attempts: u64,
    pub compute_encoder_attempts: u64,
    pub commit_attempts: u64,
    pub wait_attempts: u64,
    pub status_inspections: u64,
    pub payload_read_attempts: u64,
}
impl MetalApiCallCensus {
    const ZERO: Self = Self {
        session_open_attempts: 0,
        shader_compile_attempts: 0,
        buffer_create_attempts: 0,
        command_buffer_attempts: 0,
        compute_encoder_attempts: 0,
        commit_attempts: 0,
        wait_attempts: 0,
        status_inspections: 0,
        payload_read_attempts: 0,
    };
}
thread_local! {static COUNTS:Cell<MetalApiCallCensus>=const{Cell::new(MetalApiCallCensus::ZERO)};}
#[cfg(target_os = "macos")]
#[derive(Clone, Copy)]
pub(super) enum Call {
    Session,
    Compile,
    Buffer,
    Command,
    Encoder,
    Commit,
    Wait,
    Status,
    Read,
}
#[cfg(target_os = "macos")]
pub(super) fn record(call: Call) {
    COUNTS.with(|counter| {
        let mut counts = counter.get();
        let value = match call {
            Call::Session => &mut counts.session_open_attempts,
            Call::Compile => &mut counts.shader_compile_attempts,
            Call::Buffer => &mut counts.buffer_create_attempts,
            Call::Command => &mut counts.command_buffer_attempts,
            Call::Encoder => &mut counts.compute_encoder_attempts,
            Call::Commit => &mut counts.commit_attempts,
            Call::Wait => &mut counts.wait_attempts,
            Call::Status => &mut counts.status_inspections,
            Call::Read => &mut counts.payload_read_attempts,
        };
        *value = value.saturating_add(1);
        counter.set(counts);
    });
}
/// Resets only this caller thread's explicit Metal protocol attempt counters.
pub fn reset_api_call_census() {
    COUNTS.with(|counts| counts.set(MetalApiCallCensus::ZERO));
}
/// Reads only this caller thread's explicit Metal protocol attempt counters.
#[must_use]
pub fn api_call_census() -> MetalApiCallCensus {
    COUNTS.with(Cell::get)
}
