//! Empty declarations never become fabricated storage; actual effects retain their inputs.
use super::*;
#[rustfmt::skip]
use crate::{
    PcuScalar,
    PcuI256,
    PcuI512,
    PcuU256,
    PcuU512,
};

#[crate::pcu(crate_path = crate)]
fn repeated<T: PcuScalar>(
    unused: &[T],
    input: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    pcu::mul(input, input)
}

#[crate::pcu(crate_path = crate)]
fn discarded<T: PcuScalar>(
    effect_input: &[T],
    result_input: &[T],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    let _effect = pcu::mul(effect_input, effect_input);
    pcu::identity(result_input)
}

#[crate::pcu(crate_path = crate)]
fn shaped_unused(
    unused: &[i128; 3],
    input: &[i128],
) -> Result<crate::PcuTensor<i128>, PcuExecutionError> {
    pcu::identity(input)
}

fn verify<T: PcuScalar>() {
    let captured = __pcu_capture_tensor_program::<T, 2, _>(
        [
            PcuSourceShape::Slice { length: 0 },
            PcuSourceShape::Slice { length: 7 },
        ],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        repeated::__pcu_capture_entry::<T>,
    )
    .unwrap();
    assert_eq!(captured.argument_indices(), [1]);
    assert_eq!(captured.input_values().len(), 1);
    assert_eq!(
        captured
            .program()
            .graph()
            .node(captured.input_values()[0])
            .unwrap()
            .shape,
        [7]
    );
    assert_eq!(
        captured
            .program()
            .graph()
            .node(captured.program().output_values()[0])
            .unwrap()
            .shape,
        [7]
    );
}

#[test]
fn all_fourteen_integer_sources_prune_zero_unread_declarations_truthfully() {
    verify::<u8>();
    verify::<i8>();
    verify::<u16>();
    verify::<i16>();
    verify::<u32>();
    verify::<i32>();
    verify::<u64>();
    verify::<i64>();
    verify::<u128>();
    verify::<i128>();
    verify::<PcuU256>();
    verify::<PcuI256>();
    verify::<PcuU512>();
    verify::<PcuI512>();
}

#[test]
fn used_empty_extent_and_discarded_checked_effect_still_reject() {
    for discarded_effect in [false, true] {
        let result = __pcu_capture_tensor_program::<i128, 2, _>(
            [
                PcuSourceShape::Slice { length: 0 },
                PcuSourceShape::Slice { length: 7 },
            ],
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuNumericalMode::Boundary,
            PcuNumericalOptions::default(),
            |context, values| {
                if discarded_effect {
                    discarded::__pcu_capture_entry::<i128>(context, values)
                } else {
                    repeated::__pcu_capture_entry::<i128>(context, [values[1], values[0]])
                }
            },
        );
        assert!(matches!(result, Err(PcuExecutionError::EmptyTensorInput)));
    }
}

#[test]
fn unused_fixed_shape_obligation_is_still_observable() {
    let result = __pcu_capture_tensor_program::<i128, 2, _>(
        [
            PcuSourceShape::FixedArray { length: 2 },
            PcuSourceShape::Slice { length: 7 },
        ],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
        shaped_unused::__pcu_capture_entry,
    );
    assert!(matches!(
        result,
        Err(PcuExecutionError::TensorSourceShapeMismatch { .. })
    ));
}
