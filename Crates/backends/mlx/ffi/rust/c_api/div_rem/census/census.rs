//! Opt-in caller-thread adapter attempts, excluding all SDK-internal JIT and allocation work.
use std::cell::Cell;
/// Audited division adapter attempts on the current thread, with saturating counters.
///
/// This is not a count of native allocations, driver work or SDK-internal compilation.
/// Table lookup attempts cover the explicit loaded C function table, not upstream loader activity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlxDivRemCallCensus {
    pub exact_constructor_calls: u64,
    pub prefix_constructor_calls: u64,
    pub prime_calls: u64,
    pub apply_calls: u64,
    pub table_symbol_attempts: u64,
}
impl MlxDivRemCallCensus {
    const ZERO: Self = Self {
        exact_constructor_calls: 0,
        prefix_constructor_calls: 0,
        prime_calls: 0,
        apply_calls: 0,
        table_symbol_attempts: 0,
    };
}
thread_local! {static COUNTS:Cell<MlxDivRemCallCensus>=const{Cell::new(MlxDivRemCallCensus::ZERO)};}
#[derive(Clone, Copy)]
pub enum Call {
    ExactConstructor,
    PrefixConstructor,
    Prime,
    Apply,
    TableSymbol,
}
pub fn record(call: Call) {
    COUNTS.with(|counter| {
        let mut counts = counter.get();
        let value = match call {
            Call::ExactConstructor => &mut counts.exact_constructor_calls,
            Call::PrefixConstructor => &mut counts.prefix_constructor_calls,
            Call::Prime => &mut counts.prime_calls,
            Call::Apply => &mut counts.apply_calls,
            Call::TableSymbol => &mut counts.table_symbol_attempts,
        };
        *value = value.saturating_add(1);
        counter.set(counts);
    });
}
/// Resets only this caller thread's explicitly instrumented adapter counters.
pub fn reset_div_rem_call_census() {
    COUNTS.with(|counts| counts.set(MlxDivRemCallCensus::ZERO));
}
/// Returns only this caller thread's explicitly instrumented adapter counters.
#[must_use]
pub fn div_rem_call_census() -> MlxDivRemCallCensus {
    COUNTS.with(Cell::get)
}
