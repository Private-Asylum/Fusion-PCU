use super::*;
use crate::PcuExecutionFaultKind;
use core::convert::Infallible;

fn fault(invocation_id: u64, recovered: bool) -> PcuExecutionFault {
    PcuExecutionFault {
        kind: PcuExecutionFaultKind::ArithmeticOverflow,
        invocation_id,
        recovered,
    }
}

#[test]
fn completed_success_and_empty_scope() {
    assert_eq!(
        validate_guarded_execution::<Infallible>(0, |_| unreachable!()),
        Ok(PcuGuardedExecutionOutcome::Succeeded)
    );
    assert_eq!(
        validate_guarded_execution::<Infallible>(4, |_| {
            Ok(PcuGuardedExecutionStageOutcome::Succeeded)
        }),
        Ok(PcuGuardedExecutionOutcome::Succeeded)
    );
}

#[test]
fn stage_order_preserves_original_fault_domain_and_decodes_successors() {
    let expected = fault(900, false);
    let stages = [
        PcuGuardedExecutionStageOutcome::Succeeded,
        PcuGuardedExecutionStageOutcome::Fault(expected),
        PcuGuardedExecutionStageOutcome::Skipped,
        PcuGuardedExecutionStageOutcome::Skipped,
    ];
    let mut decoded = 0;
    assert_eq!(
        validate_guarded_execution::<Infallible>(stages.len(), |index| {
            decoded += 1;
            Ok(stages[index])
        }),
        Ok(PcuGuardedExecutionOutcome::Fault {
            stage: 1,
            fault: expected,
        })
    );
    assert_eq!(decoded, stages.len());
}

#[test]
fn later_corruption_is_not_hidden_by_a_valid_earlier_fault() {
    let mut decoded = 0;
    assert_eq!(
        validate_guarded_execution(3, |index| {
            decoded += 1;
            match index {
                0 => Ok(PcuGuardedExecutionStageOutcome::Fault(fault(7, false))),
                1 => Ok(PcuGuardedExecutionStageOutcome::Skipped),
                _ => Err("malformed physical status"),
            }
        }),
        Err(PcuGuardedExecutionValidationError::StageDecode {
            stage: 2,
            error: "malformed physical status",
        })
    );
    assert_eq!(decoded, 3);
}

#[test]
fn recovered_notice_blocks_graph_continuation() {
    for recovered in [false, true] {
        for successor in [
            PcuGuardedExecutionStageOutcome::Succeeded,
            PcuGuardedExecutionStageOutcome::Fault(fault(2, false)),
        ] {
            assert_eq!(
                validate_guarded_execution::<Infallible>(2, |index| {
                    Ok(if index == 0 {
                        PcuGuardedExecutionStageOutcome::Fault(fault(900, recovered))
                    } else {
                        successor
                    })
                }),
                Err(PcuGuardedExecutionValidationError::ExecutedAfterFault {
                    stage: 1,
                    predecessor: 0,
                })
            );
        }
    }
}

#[test]
fn skipped_is_not_success_without_a_predecessor_fault() {
    for skipped in 0..3 {
        assert_eq!(
            validate_guarded_execution::<Infallible>(3, |index| {
                Ok(if index == skipped {
                    PcuGuardedExecutionStageOutcome::Skipped
                } else {
                    PcuGuardedExecutionStageOutcome::Succeeded
                })
            }),
            Err(PcuGuardedExecutionValidationError::UnexpectedSkip { stage: skipped })
        );
    }
}

#[test]
fn disposition_tags_do_not_admit_unknown_or_truncated_values() {
    assert_eq!(
        PcuGuardedExecutionDisposition::from_raw(0),
        Some(PcuGuardedExecutionDisposition::Pending)
    );
    assert_eq!(
        PcuGuardedExecutionDisposition::from_raw(1),
        Some(PcuGuardedExecutionDisposition::Executed)
    );
    assert_eq!(
        PcuGuardedExecutionDisposition::from_raw(2),
        Some(PcuGuardedExecutionDisposition::Skipped)
    );
    for invalid in [3, 256, u64::MAX] {
        assert_eq!(PcuGuardedExecutionDisposition::from_raw(invalid), None);
    }
}
