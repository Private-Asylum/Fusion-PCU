//! Common owned derivative contract. Selection preserves bits, including signed zero.
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCompoundArithmeticPolicy,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuRangePolicy,
    PcuScalar,
    PcuTensor,
};
use super::contract::POLICY_LOCK;
#[path = "sample/sample.rs"]
mod sample;
use sample::Sample;

#[pcu]
fn retain<T: PcuScalar>(input: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn derivative<T: PcuScalar>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}

#[pcu]
fn checked_unused<T: PcuScalar>(
    input: &[T],
    upstream: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _derivative = pcu::relu_backward(input, upstream)?;
    pcu::identity(input)
}

#[pcu(flag(reject_subnormal_result))]
fn tight<T: PcuScalar>(input: &[T], upstream: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(input, upstream)
}

fn bits<T: PcuScalar>(actual: &[T], expected: &[T]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), expected.encode_le().as_ref());
    }
}

fn read<T: Sample>(owner: &PcuTensor<T>, expected: &[T; 7]) {
    assert_eq!(owner.shape(), &[7]);
    let sentinel = T::finite(16.0);
    let mut stack = [sentinel; 9];
    owner.read_into(&mut stack).unwrap();
    bits(&stack[..7], expected);
    bits(&stack[7..], &[sentinel; 2]);
}

fn fault<T: PcuScalar>(
    outcome: Result<PcuTensor<T>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
) {
    let error = outcome.unwrap_err();
    let fault = error
        .arithmetic_fault()
        .unwrap_or_else(|| panic!("expected {kind:?} at lane2, received {error:?}"));
    assert_eq!(fault.kind, kind);
    assert_eq!(fault.invocation_id, 2);
    assert!(!fault.recovered);
}

fn exceptions<T: Sample>(policy: PcuFloatUnderflowPolicy) {
    let zero = T::finite(0.0);
    let one = T::finite(1.0);
    let mut upstream = [one; 7];
    upstream[2] = T::INVALID;
    upstream[6] = T::INVALID;
    // Both operands are validated even when the derivative's branch is inactive.
    for selected in [one, zero, T::finite(-1.0)] {
        fault(
            derivative(&[selected; 7], &upstream),
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        fault(
            checked_unused(&[selected; 7], &upstream),
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    upstream[2] = T::TINY;
    upstream[6] = T::TINY;
    let tiny = retain(&upstream).unwrap();
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        fault(
            derivative::<T>(&[one; 7], &tiny),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
    } else {
        // IEEE 754-2019 clause7.5: copying an exact subnormal is not underflow.
        read(&derivative::<T>(&[one; 7], &tiny).unwrap(), &upstream);
    }
    read(&derivative::<T>(&[zero; 7], &tiny).unwrap(), &[zero; 7]);
    read(&tiny, &upstream);
    fault(
        tight::<T>(&[one; 7], &tiny),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    // A local tight flag does not change the ambient policy captured by another call.
    read(&derivative::<T>(&[zero; 7], &tiny).unwrap(), &[zero; 7]);
}

fn format<T: Sample>(policy: PcuFloatUnderflowPolicy, checked: bool) {
    global::clear_thread_cache().unwrap();
    for phase in [1.0, 2.0, 3.0] {
        let input = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0].map(T::finite);
        let upstream = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 4.0].map(|value| T::finite(value * phase));
        let expected = [0.0, 0.0, 0.0, -1.0, -2.0, -3.0, 4.0].map(|value| T::finite(value * phase));
        let resident = retain(&input).unwrap();
        let gradient = retain(&upstream).unwrap();
        let output = derivative::<T>(&resident, &gradient).unwrap();
        let sibling = derivative::<T>(&input, &upstream).unwrap();
        read(&output, &expected);
        read(&resident, &input);
        read(&gradient, &upstream);
        drop(resident);
        drop(gradient);
        drop(sibling);
        global::clear_thread_cache().unwrap();
        read(&output, &expected);
    }
    let zero = T::finite(0.0);
    let one = T::finite(1.0);
    let input = [zero, T::NEGATIVE_ZERO, one, one, T::MAX, one, one];
    let upstream = [T::NEGATIVE_ZERO; 7];
    let expected = [
        zero,
        zero,
        T::NEGATIVE_ZERO,
        T::NEGATIVE_ZERO,
        T::NEGATIVE_ZERO,
        T::NEGATIVE_ZERO,
        T::NEGATIVE_ZERO,
    ];
    read(&derivative(&input, &upstream).unwrap(), &expected);
    if checked {
        exceptions::<T>(policy);
    }
}

pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    // Scalar Clamp does not authorize an owned graph to lose its fault or its
    // completed owner. Until that shared result contract exists, reject before
    // provider discovery, allocation or mutation, including globally set Clamp.
    global::configure(global::PcuExecutionPolicy {
        backend,
        range_policy: PcuRangePolicy::Clamp,
        ..Default::default()
    })
    .unwrap();
    assert!(matches!(
        derivative(&[1.0_f32; 7], &[2.0_f32; 7]),
        Err(PcuExecutionError::UnsupportedRangePolicy)
    ));
    global::clear_thread_cache().unwrap();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
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
                        numerical_mode,
                        float_underflow,
                        numerical_options: PcuNumericalOptions {
                            compound_arithmetic,
                            precision,
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                    .unwrap();
                    sample::low_formats(float_underflow);
                    let checked = compound_arithmetic == PcuCompoundArithmeticPolicy::Checked
                        || numerical_mode == PcuNumericalMode::Strict
                        || float_underflow != PcuFloatUnderflowPolicy::IeeeAfterRounding;
                    // Native permission can use a stronger checked route. Only
                    // Boundary/IEEE native selection omits arithmetic-status work.
                    format::<f32>(float_underflow, checked);
                    format::<f64>(float_underflow, checked);
                }
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
