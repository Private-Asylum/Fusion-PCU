//! Caller-owned, allocation-free elapsed-tick accounting.
//!
//! Clock ticks are not necessarily retired CPU cycles. GPU time must be reported
//! independently from host waits. Closure scopes and [`InsightScopes`] guards close during panic
//! unwinding.

use core::cell::RefCell;

/// Stable core phases; backend points start after this vocabulary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(usize)]
pub enum InsightPoint {
    Prepare,
    Validate,
    Bind,
    AcquireStorage,
    BuildArguments,
    Submit,
    Dependency,
    Complete,
    Readback,
    Release,
    Execute,
}

impl InsightPoint {
    /// First numeric slot reserved for backend-specific measurements.
    pub const BACKEND_BASE: usize = 32;
    #[must_use]
    pub const fn index(self) -> usize {
        self as usize
    }
}

/// A clock sample and its opaque context (for example, a CPU identifier).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InsightStamp {
    pub ticks: u64,
    pub context: u64,
}

/// Sampling cost belongs to the observed measurement.
pub trait InsightClock {
    fn stamp(&mut self) -> InsightStamp;
}

/// Aggregate record indexed directly by a caller-defined numeric point.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InsightRecord {
    pub hits: u64,
    pub inclusive_ticks: u64,
    pub exclusive_ticks: u64,
    pub min_ticks: Option<u64>,
    pub max_ticks: u64,
    pub count: u64,
}

/// Invalid samples are excluded from timing aggregates.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct InsightStatus {
    pub invalid_samples: u64,
    pub depth_overflows: u64,
    pub invalid_points: u64,
    pub counter_overflow: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Frame {
    children: u64,
    invalid: bool,
    scope_id: u64,
}

#[derive(Clone, Copy)]
struct ScopeToken {
    epoch: u64,
    id: u64,
    index: usize,
    point: usize,
    start: InsightStamp,
}

/// Fixed memory bounds, with backend-defined numeric extension points.
pub struct InsightLedger<C, const POINTS: usize, const DEPTH: usize> {
    clock: C,
    records: [InsightRecord; POINTS],
    frames: [Frame; DEPTH],
    depth: usize,
    status: InsightStatus,
    scope_epoch: u64,
    next_scope_id: u64,
    scopes_poisoned: bool,
}

impl<C: InsightClock, const POINTS: usize, const DEPTH: usize> InsightLedger<C, POINTS, DEPTH> {
    pub fn new(clock: C) -> Self {
        Self {
            clock,
            records: [InsightRecord::default(); POINTS],
            frames: [Frame::default(); DEPTH],
            depth: 0,
            status: InsightStatus::default(),
            scope_epoch: 0,
            next_scope_id: 1,
            scopes_poisoned: false,
        }
    }

    pub const fn records(&self) -> &[InsightRecord; POINTS] {
        &self.records
    }
    pub const fn status(&self) -> InsightStatus {
        self.status
    }

    /// Count an event without sampling the clock.
    pub fn count(&mut self, point: usize, amount: u64) {
        if let Some(record) = self.records.get_mut(point) {
            self.status.counter_overflow |= add(&mut record.count, amount);
        } else {
            self.status.counter_overflow |= add(&mut self.status.invalid_points, 1);
        }
    }

    /// Measure a closure; its ledger argument permits safe nested accounting.
    ///
    /// The scope closes even if the operation unwinds.
    pub fn scope<R>(&mut self, point: usize, operation: impl FnOnce(&mut Self) -> R) -> R {
        let Some(token) = self.begin_scope(point) else {
            return operation(self);
        };
        let mut guard = LedgerScopeGuard {
            ledger: self,
            token: Some(token),
        };
        let result = operation(guard.ledger());
        drop(guard);
        result
    }

    fn begin_scope(&mut self, point: usize) -> Option<ScopeToken> {
        if point >= POINTS {
            self.status.counter_overflow |= add(&mut self.status.invalid_points, 1);
            if self.depth > 0 {
                self.frames[self.depth - 1].invalid = true;
            }
            return None;
        }
        if self.depth == DEPTH {
            self.status.counter_overflow |= add(&mut self.status.depth_overflows, 1);
            if self.depth > 0 {
                self.frames[self.depth - 1].invalid = true;
            }
            return None;
        }
        if self.scopes_poisoned || self.next_scope_id == 0 {
            self.status.counter_overflow |= add(&mut self.status.invalid_samples, 1);
            self.status.counter_overflow |= self.next_scope_id == 0;
            if self.depth > 0 {
                self.frames[self.depth - 1].invalid = true;
            }
            return None;
        }
        let index = self.depth;
        let id = self.next_scope_id;
        self.next_scope_id = id.checked_add(1).unwrap_or(0);
        self.frames[index] = Frame {
            scope_id: id,
            ..Frame::default()
        };
        self.depth += 1;
        Some(ScopeToken {
            epoch: self.scope_epoch,
            id,
            index,
            point,
            start: self.clock.stamp(),
        })
    }

    fn finish_scope(&mut self, token: ScopeToken) {
        let end = self.clock.stamp();
        if token.epoch != self.scope_epoch
            || self.depth == 0
            || self.depth - 1 != token.index
            || self.frames[token.index].scope_id != token.id
        {
            self.invalidate_open_scopes();
            return;
        }
        self.depth = token.index;
        let frame = self.frames[token.index];
        let elapsed = end
            .ticks
            .checked_sub(token.start.ticks)
            .filter(|_| token.start.context == end.context && !frame.invalid)
            .and_then(|ticks| ticks.checked_sub(frame.children).map(|own| (ticks, own)));
        if let Some((inclusive, exclusive)) = elapsed {
            let record = &mut self.records[token.point];
            self.status.counter_overflow |= add(&mut record.hits, 1)
                | add(&mut record.inclusive_ticks, inclusive)
                | add(&mut record.exclusive_ticks, exclusive);
            record.min_ticks = Some(record.min_ticks.map_or(inclusive, |v| v.min(inclusive)));
            record.max_ticks = record.max_ticks.max(inclusive);
            if token.index > 0 {
                let parent = &mut self.frames[token.index - 1];
                let overflow = add(&mut parent.children, inclusive);
                parent.invalid |= overflow;
                self.status.counter_overflow |= overflow;
            }
        } else {
            self.status.counter_overflow |= add(&mut self.status.invalid_samples, 1);
            if token.index > 0 {
                self.frames[token.index - 1].invalid = true;
            }
        }
    }

    const fn invalidate_open_scopes(&mut self) {
        self.status.counter_overflow |= add(&mut self.status.invalid_samples, 1);
        self.depth = 0;
        if let Some(epoch) = self.scope_epoch.checked_add(1) {
            self.scope_epoch = epoch;
        } else {
            self.scopes_poisoned = true;
            self.status.counter_overflow = true;
        }
    }
}

struct LedgerScopeGuard<'ledger, C: InsightClock, const POINTS: usize, const DEPTH: usize> {
    ledger: &'ledger mut InsightLedger<C, POINTS, DEPTH>,
    token: Option<ScopeToken>,
}

impl<C: InsightClock, const POINTS: usize, const DEPTH: usize>
    LedgerScopeGuard<'_, C, POINTS, DEPTH>
{
    const fn ledger(&mut self) -> &mut InsightLedger<C, POINTS, DEPTH> {
        self.ledger
    }
}

impl<C: InsightClock, const POINTS: usize, const DEPTH: usize> Drop
    for LedgerScopeGuard<'_, C, POINTS, DEPTH>
{
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.ledger.finish_scope(token);
        }
    }
}

/// Caller-owned, allocation-free scope manager whose guards close on every Rust exit path.
///
/// The interior borrow lasts only for individual counter/sample operations, so nested scope
/// guards can safely refer to the same manager. Read the aggregates with [`Self::records`] and
/// [`Self::status`] after the scopes have ended.
pub struct InsightScopes<C, const POINTS: usize, const DEPTH: usize> {
    ledger: RefCell<InsightLedger<C, POINTS, DEPTH>>,
}

impl<C: InsightClock, const POINTS: usize, const DEPTH: usize> InsightScopes<C, POINTS, DEPTH> {
    pub fn new(clock: C) -> Self {
        Self {
            ledger: RefCell::new(InsightLedger::new(clock)),
        }
    }

    #[must_use]
    pub fn records(&self) -> [InsightRecord; POINTS] {
        *self.ledger.borrow().records()
    }

    #[must_use]
    pub fn status(&self) -> InsightStatus {
        self.ledger.borrow().status()
    }

    pub fn count(&self, point: usize, amount: u64) {
        self.ledger.borrow_mut().count(point, amount);
    }

    pub fn enter(&self, point: usize) -> InsightScopeGuard<'_, C, POINTS, DEPTH> {
        let token = self.ledger.borrow_mut().begin_scope(point);
        InsightScopeGuard {
            scopes: self,
            token,
        }
    }
}

/// RAII scope handle produced by [`InsightScopes::enter`].
#[must_use]
pub struct InsightScopeGuard<'scopes, C: InsightClock, const POINTS: usize, const DEPTH: usize> {
    scopes: &'scopes InsightScopes<C, POINTS, DEPTH>,
    token: Option<ScopeToken>,
}

impl<C: InsightClock, const POINTS: usize, const DEPTH: usize>
    InsightScopeGuard<'_, C, POINTS, DEPTH>
{
    /// Shared manager handle for nested scopes and counters.
    #[must_use]
    pub const fn scopes(&self) -> &InsightScopes<C, POINTS, DEPTH> {
        self.scopes
    }
}

impl<C: InsightClock, const POINTS: usize, const DEPTH: usize> Drop
    for InsightScopeGuard<'_, C, POINTS, DEPTH>
{
    fn drop(&mut self) {
        if let Some(token) = self.token.take() {
            self.scopes.ledger.borrow_mut().finish_scope(token);
        }
    }
}

const fn add(value: &mut u64, amount: u64) -> bool {
    if let Some(sum) = value.checked_add(amount) {
        *value = sum;
        false
    } else {
        *value = u64::MAX;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Clock(u64);
    impl InsightClock for Clock {
        fn stamp(&mut self) -> InsightStamp {
            self.0 += 10;
            InsightStamp {
                ticks: self.0,
                context: 0,
            }
        }
    }
    #[test]
    fn nested_errors_close_and_reconcile() {
        let mut ledger = InsightLedger::<_, 2, 2>::new(Clock(0));
        let result: Result<(), ()> = ledger.scope(0, |l| {
            l.scope(1, |l| {
                l.count(1, 3);
                Err(())
            })
        });
        assert_eq!(result, Err(()));
        assert_eq!(ledger.records()[0].inclusive_ticks, 30);
        assert_eq!(ledger.records()[0].exclusive_ticks, 20);
        assert_eq!(ledger.records()[1].inclusive_ticks, 10);
        assert_eq!(ledger.records()[1].count, 3);
        assert_eq!(ledger.status(), InsightStatus::default());
    }
    #[test]
    fn overflow_executes_work_and_invalidates_parent() {
        let mut ledger = InsightLedger::<_, 1, 1>::new(Clock(0));
        assert_eq!(ledger.scope(0, |l| l.scope(0, |_| 42)), 42);
        assert_eq!(ledger.records()[0].hits, 0);
        assert_eq!(ledger.status().depth_overflows, 1);
        assert_eq!(ledger.status().invalid_samples, 1);
    }
    struct MigratingClock(u64);
    impl InsightClock for MigratingClock {
        fn stamp(&mut self) -> InsightStamp {
            self.0 += 1;
            InsightStamp {
                ticks: self.0,
                context: self.0,
            }
        }
    }
    #[test]
    fn migration_is_excluded_and_counter_overflow_is_reported() {
        let mut ledger = InsightLedger::<_, 1, 1>::new(MigratingClock(0));
        ledger.scope(0, |_| ());
        assert_eq!(ledger.records()[0].hits, 0);
        assert_eq!(ledger.status().invalid_samples, 1);
        ledger.count(0, u64::MAX);
        ledger.count(0, 1);
        assert!(ledger.status().counter_overflow);
        assert_eq!(ledger.records()[0].count, u64::MAX);
        ledger.count(1, 1);
        assert_eq!(ledger.status().invalid_points, 1);
    }

    #[test]
    fn out_of_order_guard_drop_invalidates_open_samples() {
        let scopes = InsightScopes::<_, 2, 3>::new(Clock(0));
        let outer = scopes.enter(0);
        let inner = scopes.enter(1);
        drop(outer);
        assert_eq!(scopes.status().invalid_samples, 1);
        drop(inner);
        let status = scopes.status();
        assert!(status.invalid_samples >= 2);
        assert_eq!(scopes.records()[0].hits, 0);
        assert_eq!(scopes.records()[1].hits, 0);
    }

    #[test]
    fn closure_scope_closes_during_panic_and_ledger_can_be_reused() {
        let mut ledger = InsightLedger::<_, 2, 3>::new(Clock(0));
        let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ledger.scope(0, |outer| {
                outer.scope(1, |_| panic!("closure scope panic test"));
            });
        }));
        assert!(panic.is_err());
        assert_eq!(ledger.records()[0].hits, 1);
        assert_eq!(ledger.records()[1].hits, 1);
        assert_eq!(ledger.status(), InsightStatus::default());

        ledger.scope(0, |outer| outer.count(0, 1));
        assert_eq!(ledger.records()[0].hits, 2);
        assert_eq!(ledger.records()[0].count, 1);
        assert_eq!(ledger.status(), InsightStatus::default());
    }
}
