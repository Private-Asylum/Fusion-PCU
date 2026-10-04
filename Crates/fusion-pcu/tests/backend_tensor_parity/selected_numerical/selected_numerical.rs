//! Bounded single-effect numerical lift; this does not claim a full training graph.
#[rustfmt::skip]
use fusion_pcu::{global,PcuScalar,PcuTensor,PcuExecutionError,PcuExecutionFaultKind,
    PcuNumericalMode,PcuFloatUnderflowPolicy,PcuPrecisionPolicy,PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    dialect::tensor::{TensorError,TensorArithmeticStep}};
use super::contract::POLICY_LOCK;
#[path = "source/source.rs"]
mod source;
fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 4], sentinel: T) {
    assert_eq!(owner.shape(), &[2, 2]);
    let mut stack = [sentinel; 6];
    owner.read_into(&mut stack).unwrap();
    for (actual, expected) in stack[..4].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
    for actual in &stack[4..] {
        assert_eq!(actual.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}
fn fault<T: PcuScalar>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
) {
    let error = result.unwrap_err();
    let fault = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("expected {kind:?}, received {error:?}"));
    assert_eq!(fault.kind, kind);
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
}
fn compound<T: PcuScalar>(
    result: Result<PcuTensor<T>, PcuExecutionError>,
    element_index: usize,
    reduction_index: usize,
    step: TensorArithmeticStep,
    kind: PcuExecutionFaultKind,
) {
    let error = result.unwrap_err();
    let common = error.arithmetic_fault().expect("common arithmetic fault");
    assert_eq!(common.invocation_id, element_index as u64);
    assert_eq!(common.kind, kind);
    assert!(!common.recovered);
    let PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
        element_index: actual_element,
        reduction_index: actual_reduction,
        step: actual_step,
        kind: actual_kind,
        ..
    }) = error
    else {
        panic!("lost compound arithmetic context: {error:?}");
    };
    assert_eq!(
        (actual_element, actual_reduction, actual_step, actual_kind),
        (element_index, reduction_index, step, kind)
    );
}
#[allow(clippy::too_many_lines)] // One retained cohort proves all role permutations and post-fault reuse.
fn format<T: PcuScalar>(
    value: impl Fn(f32) -> T + Copy,
    maximum: T,
    invalid: T,
    mean_tiny: T,
    underflow: PcuFloatUnderflowPolicy,
) {
    let left = [[1.0, 2.0], [3.0, 4.0]].map(|row| row.map(value));
    let right = [[1.0, 0.0], [0.0, 1.0]].map(|row| row.map(value));
    let resident_left = source::retain(&left).unwrap();
    let resident_right = source::retain(&right).unwrap();
    let expected_product = [1.0, 2.0, 3.0, 4.0].map(value);
    let expected_derivative = [1.0, 0.0, 0.0, 1.0].map(value);
    let expected_update = [0.5, 2.0, 3.0, 3.5].map(value);
    let sentinel = value(19.0);
    for mode in 0..4 {
        let outputs = match mode {
            0 => [
                source::product(&left, &right),
                source::derivative(&left, &right),
                source::update(&left, &right),
            ],
            1 => [
                source::product::<T>(&resident_left, &right),
                source::derivative::<T>(&resident_left, &right),
                source::update::<T>(&resident_left, &right),
            ],
            2 => [
                source::product::<T>(&left, &resident_right),
                source::derivative::<T>(&left, &resident_right),
                source::update::<T>(&left, &resident_right),
            ],
            _ => [
                source::product::<T>(&resident_left, &resident_right),
                source::derivative::<T>(&resident_left, &resident_right),
                source::update::<T>(&resident_left, &resident_right),
            ],
        };
        let loss = match mode {
            0 => source::loss(&left, &right),
            1 => source::loss::<T>(&resident_left, &right),
            2 => source::loss::<T>(&left, &resident_right),
            _ => source::loss::<T>(&resident_left, &resident_right),
        }
        .unwrap();
        assert!(loss.shape().is_empty());
        let mut scalar_stack = [sentinel; 3];
        loss.read_into(&mut scalar_stack).unwrap();
        assert_eq!(
            scalar_stack[0].encode_le().as_ref(),
            value(5.5).encode_le().as_ref()
        );
        for tail in &scalar_stack[1..] {
            assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
        }
        for (output, expected) in
            outputs
                .into_iter()
                .zip([expected_product, expected_derivative, expected_update])
        {
            read(&output.unwrap(), &expected, sentinel);
        }
    }
    for result in [
        source::discarded_product::<T>(&resident_left, &resident_right),
        source::discarded_derivative::<T>(&resident_left, &resident_right),
        source::discarded_update::<T>(&resident_left, &resident_right),
        source::discarded_loss::<T>(&resident_left, &resident_right),
    ] {
        read(&result.unwrap(), &[1.0, 0.0, 0.0, 1.0].map(value), sentinel);
    }
    let old = source::update::<T>(&resident_left, &resident_right).unwrap();
    // IEEE 754 range failure belongs to the separately rounded Strict operation.
    let overflow = [[maximum, value(0.0)], [value(0.0), value(1.0)]];
    let twice = [[value(2.0), value(0.0)], [value(0.0), value(1.0)]];
    fault(
        source::product(&overflow, &twice),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        source::discarded_product(&overflow, &twice),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    let invalid_left = [[invalid, value(0.0)], [value(0.0), value(1.0)]];
    fault(
        source::derivative(&invalid_left, &right),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    fault(
        source::discarded_derivative(&invalid_left, &right),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    fault(
        source::update(&invalid_left, &right),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    fault(
        source::discarded_update(&invalid_left, &right),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    // Raw provider ordinals differ from output cells. Preserve both the
    // common row-major coordinate and exact arithmetic step, including dead effects.
    let late_weight = [[value(1.0), value(2.0)], [invalid, value(4.0)]];
    let late_gradient = [[value(1.0), value(2.0)], [value(3.0), invalid]];
    for result in [
        source::update(&late_weight, &right),
        source::discarded_update(&late_weight, &right),
    ] {
        compound(
            result,
            2,
            0,
            TensorArithmeticStep::Subtract,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    for result in [
        source::update(&left, &late_gradient),
        source::discarded_update(&left, &late_gradient),
    ] {
        compound(
            result,
            3,
            0,
            TensorArithmeticStep::Multiply,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    let late_matrix = [[value(1.0), invalid], [value(0.0), value(1.0)]];
    for result in [
        source::product(&left, &late_matrix),
        source::discarded_product(&left, &late_matrix),
    ] {
        compound(
            result,
            1,
            0,
            TensorArithmeticStep::Multiply,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    let ones = [[value(1.0); 2]; 2];
    let large_column = [[maximum, value(0.0)]; 2];
    for result in [
        source::product(&ones, &large_column),
        source::discarded_product(&ones, &large_column),
    ] {
        compound(
            result,
            0,
            1,
            TensorArithmeticStep::Add,
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
    }
    let zero = [[value(0.0); 2]; 2];
    for result in [
        source::loss(&late_weight, &zero),
        source::discarded_loss(&late_weight, &zero),
    ] {
        compound(
            result,
            0,
            2,
            TensorArithmeticStep::Subtract,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    // A scalar result still has four reduction elements and thirteen ordered
    // event positions. IEEE after-rounding accepts the exact tiny square, but
    // rejects its inexact tiny mean. Tightened policy rejects the square first.
    let tiny = [[mean_tiny, value(0.0)], [value(0.0); 2]];
    for result in [
        source::loss(&tiny, &zero),
        source::discarded_loss(&tiny, &zero),
    ] {
        match underflow {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => compound(
                result,
                0,
                4,
                TensorArithmeticStep::Divide,
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ),
            PcuFloatUnderflowPolicy::RejectSubnormalResult => compound(
                result,
                0,
                0,
                TensorArithmeticStep::Multiply,
                PcuExecutionFaultKind::ArithmeticUnderflow,
            ),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => {
                let output = result.unwrap();
                assert!(matches!(output.len(), 1 | 4));
                let mut stack = [sentinel; 6];
                output.read_into(&mut stack).unwrap();
                for element in &stack[..output.len()] {
                    assert_eq!(
                        element.encode_le().as_ref(),
                        value(0.0).encode_le().as_ref()
                    );
                }
                for tail in &stack[output.len()..] {
                    assert_eq!(tail.encode_le().as_ref(), sentinel.encode_le().as_ref());
                }
            }
        }
    }
    read(&old, &expected_update, sentinel);
    read(
        &source::product::<T>(&resident_left, &resident_right).unwrap(),
        &expected_product,
        sentinel,
    );
    // Clearing cached preparation may not invalidate escaped device ownership.
    global::clear_thread_cache().unwrap();
    drop(resident_left);
    drop(resident_right);
    read(&old, &expected_update, sentinel);
}
pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for compound_arithmetic in [
        PcuCompoundArithmeticPolicy::Checked,
        PcuCompoundArithmeticPolicy::BackendDefined,
    ] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                global::configure(global::PcuExecutionPolicy {
                    backend,
                    numerical_mode: PcuNumericalMode::Boundary,
                    numerical_options: PcuNumericalOptions {
                        precision,
                        compound_arithmetic,
                        ..Default::default()
                    },
                    float_underflow,
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                format::<f32>(
                    |v| v,
                    f32::MAX,
                    f32::NAN,
                    f32::from_bits((127 - 74) << 23),
                    float_underflow,
                );
                format::<f64>(
                    f64::from,
                    f64::MAX,
                    f64::NAN,
                    f64::from_bits((1023 - 537) << 52),
                    float_underflow,
                );
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
