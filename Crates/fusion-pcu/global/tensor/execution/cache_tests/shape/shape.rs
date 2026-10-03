//! Selected source preflight precedes physical staging and omits unread declarations.
#[rustfmt::skip]
use super::super::{
    preflight_selected_inputs,
    PcuArgumentError,
    PcuExecutionError,
    PcuSourceShape,
    PcuTensorInput,
};

#[test]
fn empty_unread_host_argument_does_not_acquire_a_session_or_extent() {
    let empty: [u128; 0] = [];
    let selected = [3_u128];
    let inputs = [
        PcuTensorInput::host(&empty, PcuSourceShape::Slice { length: 0 }),
        PcuTensorInput::host(&selected, PcuSourceShape::FixedArray { length: 1 }),
    ];
    assert!(preflight_selected_inputs(&inputs, &[1]).unwrap().is_none());
    assert!(matches!(
        preflight_selected_inputs(&inputs, &[0]),
        Err(PcuExecutionError::EmptyTensorInput)
    ));
}

#[test]
fn selected_host_length_mismatch_rejects_before_staging() {
    let values = [3_u128];
    let inputs = [PcuTensorInput::host(
        &values,
        PcuSourceShape::FixedArray { length: 2 },
    )];
    assert!(matches!(
        preflight_selected_inputs(&inputs, &[0]),
        Err(PcuExecutionError::Argument(
            PcuArgumentError::SourceShapeMismatch {
                expected: PcuSourceShape::FixedArray { length: 2 },
                actual: PcuSourceShape::Slice { length: 1 },
            }
        ))
    ));
    assert!(preflight_selected_inputs(&inputs, &[]).unwrap().is_none());
}

#[test]
fn invalid_frozen_selection_rejects_without_touching_an_argument() {
    let values = [3_u128];
    let inputs = [PcuTensorInput::host(
        &values,
        PcuSourceShape::Slice { length: 1 },
    )];
    assert!(matches!(
        preflight_selected_inputs(&inputs, &[1]),
        Err(PcuExecutionError::InvalidTensorSourcePlan)
    ));
}
