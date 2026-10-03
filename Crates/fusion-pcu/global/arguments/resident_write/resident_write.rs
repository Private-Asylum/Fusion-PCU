//! Shared logical publication law for mutable resident arguments.

use super::ResidentValidity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResidentWriteDisposition {
    NotSubmitted,
    MayHaveWritten,
    Discarded,
    Complete,
}

/// Tracks whether a mutable resident value remains readable across failure. The caller marks
/// possible submission only after preflight; Drop restores the prior state for prelaunch errors,
/// retains completed successful/recovered output, discards failed logical output even if its
/// bytes remain initialized, and blocks uncertain completion. Quiescence alone is not success.
pub(in crate::global) struct ResidentWriteGuard<'a> {
    validity: &'a mut ResidentValidity,
    prior: ResidentValidity,
    disposition: ResidentWriteDisposition,
}

impl<'a> ResidentWriteGuard<'a> {
    pub(super) const fn new(validity: &'a mut ResidentValidity) -> Self {
        let prior = *validity;
        Self {
            validity,
            prior,
            disposition: ResidentWriteDisposition::NotSubmitted,
        }
    }

    pub(in crate::global) const fn mark_may_have_written(&mut self) {
        self.disposition = ResidentWriteDisposition::MayHaveWritten;
    }

    pub(in crate::global) const fn mark_not_submitted(&mut self) {
        self.disposition = ResidentWriteDisposition::NotSubmitted;
    }

    pub(in crate::global) const fn mark_discarded(&mut self) {
        self.disposition = ResidentWriteDisposition::Discarded;
    }

    pub(in crate::global) const fn mark_complete(&mut self) {
        self.disposition = ResidentWriteDisposition::Complete;
    }
}

impl Drop for ResidentWriteGuard<'_> {
    fn drop(&mut self) {
        *self.validity = match self.disposition {
            ResidentWriteDisposition::NotSubmitted => self.prior,
            ResidentWriteDisposition::MayHaveWritten => ResidentValidity::Uncertain,
            ResidentWriteDisposition::Discarded => ResidentValidity::Discarded,
            ResidentWriteDisposition::Complete => ResidentValidity::Ready,
        };
    }
}

/// Completion proves safe release, not a successful logical value. Every provider uses this
/// classification instead of reconstructing launch state from its particular error variants.
pub(in crate::global) fn finish_resident_writes(
    guards: &mut [Option<ResidentWriteGuard<'_>>],
    may_have_written: bool,
    completion_uncertain: bool,
    result: &Result<(), crate::global::PcuExecutionError>,
) {
    if completion_uncertain {
        // Quarantine even when the backend has not yet reported an output writer. Keep this
        // failure-only safeguard local rather than relying on callers' pre-submission markers.
        for guard in guards.iter_mut().flatten() {
            guard.mark_may_have_written();
        }
        return;
    }
    let publishable = result.is_ok()
        || matches!(result,
        Err(crate::global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered);
    for guard in guards.iter_mut().flatten() {
        if !may_have_written {
            guard.mark_not_submitted();
        } else if publishable {
            guard.mark_complete();
        } else {
            guard.mark_discarded();
        }
    }
}
