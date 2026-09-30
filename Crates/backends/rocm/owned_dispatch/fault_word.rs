//! Terminal sentinel proof shared by reusable synchronous executables.

use fusion_pcu::PcuCompletionOutcome;

#[allow(clippy::redundant_pub_crate)] // This physical-state proof is never a public backend API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FaultWordState {
    NeedsReset,
    Sentinel,
}

impl FaultWordState {
    const fn needs_reset(self) -> bool {
        matches!(self, Self::NeedsReset)
    }

    pub(crate) const fn begin_submission(&mut self) -> bool {
        let reset = self.needs_reset();
        // Once any launch is attempted, the old sentinel proof no longer applies.
        // Only a terminal successful readback restores it.
        *self = Self::NeedsReset;
        reset
    }

    pub(crate) const fn after_terminal(outcome: PcuCompletionOutcome) -> Self {
        match outcome {
            PcuCompletionOutcome::Succeeded => Self::Sentinel,
            PcuCompletionOutcome::Fault(_) | PcuCompletionOutcome::Failed => Self::NeedsReset,
        }
    }
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        FaultWordState,
        PcuCompletionOutcome,
    };
    #[rustfmt::skip]
    use super::super::{
        is_certain_checked_prelaunch_error,
        RocmOwnedDispatchError,
    };
    use crate::HipError;

    #[test]
    fn fault_word_state_reuses_only_a_terminally_proven_success_sentinel() {
        let mut state = FaultWordState::NeedsReset;
        assert!(
            state.begin_submission(),
            "first use must initialize device memory"
        );
        assert!(
            state.needs_reset(),
            "submission consumes any sentinel proof"
        );
        state = FaultWordState::after_terminal(PcuCompletionOutcome::Succeeded);
        assert!(
            !state.needs_reset(),
            "terminal success proves the word is still MAX"
        );

        assert!(
            !state.begin_submission(),
            "a consecutive successful call can reuse the retained MAX word"
        );
        assert!(
            state.needs_reset(),
            "in-flight submission is no longer proven clean"
        );
        state = FaultWordState::after_terminal(PcuCompletionOutcome::Succeeded);

        assert!(!state.begin_submission());
        state = FaultWordState::after_terminal(PcuCompletionOutcome::Fault(
            fusion_pcu::PcuExecutionFault {
                kind: fusion_pcu::PcuExecutionFaultKind::DivideByZero,
                invocation_id: 0,
                recovered: false,
            },
        ));
        assert!(
            state.needs_reset(),
            "terminal fault leaves a non-MAX word for retry"
        );
        assert!(
            state.begin_submission(),
            "retry after fault must reset status"
        );
        state = FaultWordState::after_terminal(PcuCompletionOutcome::Succeeded);
        assert!(!state.needs_reset());
    }

    #[test]
    fn rejected_or_uncertain_attempts_consume_the_sentinel_proof() {
        let mut state = FaultWordState::Sentinel;
        assert!(!state.begin_submission());
        // A known prelaunch rejection leaves the word untouched, but we intentionally require
        // another reset because the attempt cannot establish the next terminal state.
        assert!(state.needs_reset());
        assert!(state.begin_submission());

        // A wait or status-read error has no terminal success transition; the wrapper's existing
        // poison path retains/quarantines resources and the dirty state cannot be reused.
        assert!(state.needs_reset());
    }

    #[test]
    fn recovered_clamp_faults_require_reset_before_a_clean_retry() {
        let recovered = PcuCompletionOutcome::Fault(fusion_pcu::PcuExecutionFault {
            kind: fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow,
            invocation_id: 19,
            recovered: true,
        });
        // Clamp preserves useful output, but its observable status is still a non-sentinel word.
        let mut state = FaultWordState::after_terminal(recovered);
        assert!(state.begin_submission());
        assert!(state.needs_reset());
        state = FaultWordState::after_terminal(PcuCompletionOutcome::Succeeded);
        assert!(!state.begin_submission());
    }

    #[test]
    fn terminal_failure_does_not_establish_a_clean_sentinel() {
        let mut state = FaultWordState::after_terminal(PcuCompletionOutcome::Failed);
        assert!(state.begin_submission());
        // A second attempt without a terminal success cannot regain the sentinel proof.
        assert!(state.begin_submission());
    }

    #[test]
    fn hip_failures_never_qualify_for_unpoisoned_prelaunch_retry() {
        for operation in ["hipLaunchKernel", "hipEventSynchronize", "hipMemcpyDtoH"] {
            assert!(!is_certain_checked_prelaunch_error(
                &RocmOwnedDispatchError::Hip(HipError::Runtime {
                    operation,
                    code: 1,
                    detail: None,
                })
            ));
        }
        for error in [HipError::Busy, HipError::InvalidExecutionFaultWord(0)] {
            assert!(!is_certain_checked_prelaunch_error(
                &RocmOwnedDispatchError::Hip(error)
            ));
        }
        assert!(is_certain_checked_prelaunch_error(
            &RocmOwnedDispatchError::CheckedFaultWordSize { actual: 4 }
        ));
        assert!(!is_certain_checked_prelaunch_error(
            &RocmOwnedDispatchError::CheckedSequentialDispatchPoisoned
        ));
    }
}
