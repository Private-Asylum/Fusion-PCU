//! Probes disappear during expansion when the insights feature is disabled.

/// Measure a block body with an RAII scope, binding its manager for nested probes.
///
/// With insights disabled, only the body remains; sink and point are not evaluated. The enabled
/// and disabled forms preserve the block's normal `return`, `?`, `break`, and panic behavior.
#[cfg(feature = "insights")]
#[macro_export]
macro_rules! insight_scope {
    ($sink:expr, $point:expr, |$ledger:ident| $body:block) => {{
        let __fusion_pcu_insight_guard = ($sink).enter($point);
        let $ledger = __fusion_pcu_insight_guard.scopes();
        $body
    }};
    ($sink:expr, $point:expr, $body:block) => {{
        let __fusion_pcu_insight_guard = ($sink).enter($point);
        $body
    }};
}

/// Execute only the body when instrumentation is compiled out.
#[cfg(not(feature = "insights"))]
#[macro_export]
macro_rules! insight_scope {
    ($sink:expr, $point:expr, |$ledger:ident| $body:block) => {
        $body
    };
    ($sink:expr, $point:expr, $body:block) => {
        $body
    };
}

/// Increment a fixed counter without sampling a clock.
#[cfg(feature = "insights")]
#[macro_export]
macro_rules! insight_count {
    ($sink:expr, $point:expr, $amount:expr) => {
        ($sink).count($point, $amount)
    };
}

/// Evaluate none of the instrumentation arguments when disabled.
#[cfg(not(feature = "insights"))]
#[macro_export]
macro_rules! insight_count {
    ($sink:expr, $point:expr, $amount:expr) => {
        ()
    };
}

#[cfg(all(test, not(feature = "insights")))]
mod tests {
    #[test]
    fn disabled_probes_do_not_resolve_or_evaluate_arguments() {
        let result = crate::insight_scope!(undefined_sink, undefined_point, |ledger| {
            crate::insight_count!(ledger, undefined_counter, undefined_amount);
            42
        });
        assert_eq!(result, 42);
    }

    #[test]
    fn disabled_scope_keeps_question_mark_return_semantics() {
        fn outer() -> Result<u32, ()> {
            crate::insight_scope!(undefined_sink, undefined_point, |ledger| {
                crate::insight_count!(ledger, undefined_counter, undefined_amount);
                let value = Some(42).ok_or(())?;
                Ok(value)
            })
        }
        assert_eq!(outer(), Ok(42));
    }

    #[test]
    fn disabled_leaf_scope_is_a_plain_block() {
        let result = crate::insight_scope!(undefined_sink, undefined_point, { 40 + 2 });
        assert_eq!(result, 42);
    }
}

#[cfg(all(test, feature = "insights"))]
mod enabled_tests {
    #[rustfmt::skip]
    use crate::insights::{
        InsightClock,
        InsightScopes,
        InsightStamp,
        InsightStatus,
    };

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

    fn exits_through_question_mark(scopes: &InsightScopes<Clock, 2, 4>) -> Result<(), ()> {
        crate::insight_scope!(scopes, 0, |outer| {
            crate::insight_scope!(outer, 1, |inner| {
                crate::insight_count!(inner, 1, 3);
                Err(())?;
                #[allow(unreachable_code)]
                Ok(())
            })?;
            Ok(())
        })?;
        Ok(())
    }

    #[allow(unreachable_code)]
    fn returns_from_outer(scopes: &InsightScopes<Clock, 1, 2>) -> u32 {
        crate::insight_scope!(scopes, 0, |ledger| {
            crate::insight_count!(ledger, 0, 1);
            return 42;
        });
        0
    }

    #[test]
    fn scopes_close_on_question_mark_and_nest() {
        let scopes = InsightScopes::<_, 2, 4>::new(Clock(0));
        assert_eq!(exits_through_question_mark(&scopes), Err(()));
        let records = scopes.records();
        assert_eq!(records[0].hits, 1);
        assert_eq!(records[1].hits, 1);
        assert_eq!(records[1].count, 3);
        assert_eq!(scopes.status(), InsightStatus::default());
    }

    #[test]
    fn scope_closes_on_return_from_outer_function() {
        let scopes = InsightScopes::<_, 1, 2>::new(Clock(0));
        assert_eq!(returns_from_outer(&scopes), 42);
        assert_eq!(scopes.records()[0].hits, 1);
    }

    #[test]
    fn loop_break_and_continue_keep_their_outer_targets() {
        let scopes = InsightScopes::<_, 1, 2>::new(Clock(0));
        let mut visits = 0;
        for index in 0..3 {
            crate::insight_scope!(&scopes, 0, |ledger| {
                crate::insight_count!(ledger, 0, 1);
                if index == 0 {
                    continue;
                }
                if index == 1 {
                    break;
                }
                visits += 1;
            });
        }
        assert_eq!(visits, 0);
        assert_eq!(scopes.records()[0].hits, 2);
        assert_eq!(scopes.records()[0].count, 2);
        assert_eq!(scopes.status(), InsightStatus::default());
    }

    #[test]
    fn panic_unwinding_closes_nested_scopes() {
        let scopes = InsightScopes::<_, 2, 3>::new(Clock(0));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::insight_scope!(&scopes, 0, |outer| {
                crate::insight_scope!(outer, 1, |inner| {
                    crate::insight_count!(inner, 1, 1);
                    panic!("scope panic test");
                });
            });
        }));
        assert!(result.is_err());
        let records = scopes.records();
        assert_eq!(records[0].hits, 1);
        assert_eq!(records[1].hits, 1);
        assert_eq!(records[1].count, 1);
        assert_eq!(scopes.status(), InsightStatus::default());
    }

    #[test]
    fn leaf_scope_returns_its_block_value() {
        let scopes = InsightScopes::<_, 1, 2>::new(Clock(0));
        let result = crate::insight_scope!(&scopes, 0, { 40 + 2 });
        assert_eq!(result, 42);
        assert_eq!(scopes.records()[0].hits, 1);
    }
}
