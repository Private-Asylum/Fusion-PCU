//! Caller-thread adapter attempts; SDK-internal compilation and allocations are unobserved.
use std::cell::Cell;
/// Explicit retained integer adapter attempts, with saturating caller-thread counters.
/// Table lookups cover this adapter's loaded function table, not upstream loader work.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MlxIntegerCallCensus {
    pub exact_constructor_calls: u64,
    pub prefix_constructor_calls: u64,
    pub prime_calls: u64,
    pub apply_calls: u64,
    pub table_symbol_attempts: u64,
}
impl MlxIntegerCallCensus {
    const ZERO: Self = Self {
        exact_constructor_calls: 0,
        prefix_constructor_calls: 0,
        prime_calls: 0,
        apply_calls: 0,
        table_symbol_attempts: 0,
    };
}
thread_local! {static COUNTS:Cell<MlxIntegerCallCensus>=const{Cell::new(MlxIntegerCallCensus::ZERO)};}
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
/// Resets only the current thread's explicitly instrumented adapter attempts.
pub fn reset_integer_call_census() {
    COUNTS.with(|counts| counts.set(MlxIntegerCallCensus::ZERO));
}
/// Returns only the current thread's explicitly instrumented adapter attempts.
#[must_use]
pub fn integer_call_census() -> MlxIntegerCallCensus {
    COUNTS.with(Cell::get)
}
