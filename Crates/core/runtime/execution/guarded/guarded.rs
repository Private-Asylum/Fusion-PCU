//! Ordered user-effect admission for an already-enqueued execution chain.
//!
//! This is distinct from [`super::PcuExecutionSuccessGate`]: a guarded successor may be
//! submitted before its predecessor's outcome is observed by the host. The provider must
//! suppress every user resource load, arithmetic operation and output write after an earlier
//! blocking outcome. Private admission bookkeeping is permitted. The initial contract stops
//! on **any** arithmetic fault, including a recovered range notice; it does not authorize
//! containing-graph continuation or publication of partially completed outputs.
//!
//! These allocation-free contracts neither enqueue work nor establish physical quiescence.
//! Providers retain all resources/access leases until terminal completion is established,
//! validate their complete physical records under each stage's exact fault law and domain,
//! and quarantine uncertain work. A successful validation below is not a rollback guarantee
//! for caller-owned mutable resources and never replaces physical completion/access proofs.

use crate::PcuExecutionFault;

/// Host observation preferences, independent of arithmetic strictness and determinism.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PcuExecutionObservationPolicy {
    /// Providers may use a proved guarded chain within one logical execution scope.
    #[default]
    Automatic,
    /// Observe each checked stage on the host before submitting its successor.
    ///
    /// This is useful for diagnostic granularity or literal no-successor-enqueue requirements.
    /// It does not request payload readback, disable arithmetic checks, or turn internal stages
    /// into separately published Rust results.
    HostObservedStages,
}

/// Provider admission bookkeeping for one stage of an ordered guarded chain.
///
/// The integer values are a shared disposition vocabulary, not a prescribed device record
/// layout or arithmetic-fault encoding. Providers may use another physical representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u64)]
pub enum PcuGuardedExecutionDisposition {
    /// No terminal stage disposition has been observed; this is not success.
    Pending = 0,
    /// The stage's user body was admitted. Its validated fault record decides its outcome.
    Executed = 1,
    /// An earlier stage blocked this stage. No user body effects were permitted.
    Skipped = 2,
}

impl PcuGuardedExecutionDisposition {
    /// Decodes only the disposition tag; it does not validate completion or a fault record.
    #[must_use]
    pub const fn from_raw(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::Pending),
            1 => Some(Self::Executed),
            2 => Some(Self::Skipped),
            _ => None,
        }
    }
}

/// One physically completed and provider-validated stage outcome.
///
/// A provider must reject pending/unknown dispositions, malformed faults, a fault attached
/// to a skipped stage, and faults from stages whose admitted law cannot produce them before
/// constructing these values. A skipped stage is never represented as successful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuGuardedExecutionStageOutcome {
    Succeeded,
    Fault(PcuExecutionFault),
    Skipped,
}

/// The validated result of an ordered, stop-on-any-fault execution scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuGuardedExecutionOutcome {
    Succeeded,
    /// The first fault in authored stage order, retaining that stage's own fault selection.
    Fault {
        stage: usize,
        fault: PcuExecutionFault,
    },
}

/// A physical decoding failure or violation of ordered guarded-effect admission.
///
/// These are protocol errors, not arithmetic faults. They must block output publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuGuardedExecutionValidationError<E> {
    StageDecode { stage: usize, error: E },
    UnexpectedSkip { stage: usize },
    ExecutedAfterFault { stage: usize, predecessor: usize },
}

/// Validates a physically completed chain without allocating or hiding later protocol errors.
///
/// `decode` is called in stage order. Before returning an arithmetic fault, every stage is
/// decoded and checked, including successors after that fault. Decoding/protocol errors may
/// return immediately because publication is already rejected. The callback must validate
/// each physical record against its retained operation law, logical fault domain and actual
/// executed/skipped disposition. It must not inspect unconfirmed in-flight device storage.
///
/// Every stage preceding the first fault must succeed; every subsequent stage must be skipped.
/// A recovered notice also blocks successors. An empty chain succeeds without calling `decode`.
/// The stage index identifies an execution position, not a user source span; providers retain
/// their cold mapping to graph values/source locations separately.
///
/// # Errors
/// Returns the first decoding failure, an unblocked skipped stage, or user execution after an
/// earlier blocking fault. An early arithmetic fault cannot conceal a later malformed record.
pub fn validate_guarded_execution<E>(
    stage_count: usize,
    mut decode: impl FnMut(usize) -> Result<PcuGuardedExecutionStageOutcome, E>,
) -> Result<PcuGuardedExecutionOutcome, PcuGuardedExecutionValidationError<E>> {
    let mut first_fault = None;
    for stage in 0..stage_count {
        let outcome = decode(stage)
            .map_err(|error| PcuGuardedExecutionValidationError::StageDecode { stage, error })?;
        match (first_fault, outcome) {
            (None, PcuGuardedExecutionStageOutcome::Succeeded)
            | (Some(_), PcuGuardedExecutionStageOutcome::Skipped) => {}
            (None, PcuGuardedExecutionStageOutcome::Fault(fault)) => {
                first_fault = Some((stage, fault));
            }
            (None, PcuGuardedExecutionStageOutcome::Skipped) => {
                return Err(PcuGuardedExecutionValidationError::UnexpectedSkip { stage });
            }
            (Some((predecessor, _)), _) => {
                return Err(PcuGuardedExecutionValidationError::ExecutedAfterFault {
                    stage,
                    predecessor,
                });
            }
        }
    }
    Ok(
        first_fault.map_or(PcuGuardedExecutionOutcome::Succeeded, |(stage, fault)| {
            PcuGuardedExecutionOutcome::Fault { stage, fault }
        }),
    )
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
