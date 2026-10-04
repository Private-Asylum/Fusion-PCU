//! Ordinary six-format source graphs with exact bit witnesses and ordered faults.
#[rustfmt::skip]
use fusion_pcu::{global, pcu, PcuScalar, PcuTensor, PcuExecutionError,
    PcuNumericalOptions, PcuPrecisionPolicy, PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy, PcuExecutionFaultKind, PcuF16Bits, PcuBf16Bits,
    PcuF8E4M3FnBits, PcuF8E5M2Bits, dialect::tensor::TensorError};
use super::contract::POLICY_LOCK;

#[pcu(flag(strict))]
fn chain<T: PcuScalar>(
    a: &[[T; 2]; 2],
    b: &[[T; 2]; 2],
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = pcu::add(a, b)?;
    let prediction = pcu::relu(&sum)?;
    let gradient = pcu::sub(&prediction, target)?;
    pcu::relu_backward(&sum, &gradient)
}
#[pcu]
fn consume<T: PcuScalar>(input: PcuTensor<T>) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(&input)
}
#[pcu(flag(strict))]
fn discarded<T: PcuScalar>(
    a: &[[T; 2]; 2],
    b: &[[T; 2]; 2],
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let sum = pcu::add(a, b)?;
    let prediction = pcu::relu(&sum)?;
    let gradient = pcu::sub(&prediction, target)?;
    let _mandatory = pcu::relu_backward(&sum, &gradient)?;
    pcu::identity(a)
}
#[pcu(flag(strict))]
fn ordered<T: PcuScalar>(
    a: &[[T; 2]; 2],
    b: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    let _first = pcu::relu(a)?;
    let _second = pcu::relu(b)?;
    pcu::identity(a)
}
#[pcu]
fn retain<T: PcuScalar>(a: &[[T; 2]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(a)
}
fn read<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T; 4], sentinel: T) {
    let mut actual = [sentinel; 6];
    owner.read_into(&mut actual).unwrap();
    for (a, b) in actual[..4].iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
    for a in &actual[4..] {
        assert_eq!(a.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}
fn fault(error: &PcuExecutionError, expected: usize, kind: PcuExecutionFaultKind) {
    let view = error.arithmetic_fault().unwrap();
    assert_eq!(view.invocation_id, u64::try_from(expected).unwrap());
    assert_eq!(view.kind, kind);
    assert!(!view.recovered);
    let PcuExecutionError::TensorBuild(TensorError::ArithmeticFault {
        element_index,
        kind: actual,
        ..
    }) = error
    else {
        panic!("lost original pointwise fault: {error:?}")
    };
    assert_eq!(*element_index, expected);
    assert_eq!(*actual, kind);
}
#[allow(clippy::too_many_lines)] // All roles share the same escaped owners and independent bit witness.
fn format<T: PcuScalar>(values: [T; 7], policy: PcuFloatUnderflowPolicy) {
    let [zero, one, negative, two, three, invalid, tiny] = values;
    let left = [[negative, zero], [one, two]];
    let right = [[zero, one], [one, two]];
    let target = [[one; 2]; 2];
    let expected = [zero, zero, one, three];
    let left_owner = retain(&left).unwrap();
    let right_owner = retain(&right).unwrap();
    let target_owner = retain(&target).unwrap();
    let old = chain(&left, &right, &target).unwrap();
    let mut escaped = Vec::new();
    for role in 0..4 {
        let output = match role {
            0 => chain(&left, &right, &target),
            1 => chain(&left_owner, &right, &target_owner),
            2 => chain(&left, &right_owner, &target),
            _ => chain(&left_owner, &right_owner, &target_owner),
        }
        .unwrap();
        read(&output, &expected, negative);
        read(&old, &expected, negative);
        escaped.push(output);
        let output = match role {
            0 => discarded(&left, &right, &target),
            1 => discarded(&left_owner, &right, &target_owner),
            2 => discarded(&left, &right_owner, &target),
            _ => discarded(&left_owner, &right_owner, &target_owner),
        }
        .unwrap();
        read(&output, &[negative, zero, one, two], negative);
    }
    let mut bad_a = left;
    bad_a[1][1] = invalid;
    let mut bad_target = target;
    bad_target[0][0] = invalid;
    for (input, expected_lane) in [(bad_a, 3), (left, 0)] {
        fault(
            &chain(&input, &right, &bad_target).unwrap_err(),
            expected_lane,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        fault(
            &discarded(&input, &right, &bad_target).unwrap_err(),
            expected_lane,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
        let input_owner = retain(&input).unwrap();
        let bad_target_owner = retain(&bad_target).unwrap();
        for role in 1..4 {
            let error = match role {
                1 => chain(&input_owner, &right, &bad_target_owner),
                2 => chain(&input, &right_owner, &bad_target),
                _ => chain(&input_owner, &right_owner, &bad_target_owner),
            }
            .unwrap_err();
            fault(
                &error,
                expected_lane,
                PcuExecutionFaultKind::InvalidFloatingOperand,
            );
            let error = match role {
                1 => discarded(&input_owner, &right, &bad_target_owner),
                2 => discarded(&input, &right_owner, &bad_target),
                _ => discarded(&input_owner, &right_owner, &bad_target_owner),
            }
            .unwrap_err();
            fault(
                &error,
                expected_lane,
                PcuExecutionFaultKind::InvalidFloatingOperand,
            );
            read(&old, &expected, negative);
        }
        fault(
            &ordered(&input, &bad_target).unwrap_err(),
            expected_lane,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    let mut small = [[zero; 2]; 2];
    small[1][1] = tiny;
    let zeros = [[zero; 2]; 2];
    if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
        fault(
            &chain(&small, &zeros, &zeros).unwrap_err(),
            3,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        fault(
            &discarded(&small, &zeros, &zeros).unwrap_err(),
            3,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
    } else {
        read(
            &chain(&small, &zeros, &zeros).unwrap(),
            &[zero, zero, zero, tiny],
            negative,
        );
        read(
            &discarded(&small, &zeros, &zeros).unwrap(),
            &[zero, zero, zero, tiny],
            negative,
        );
    }
    read(&chain(&left, &right, &target).unwrap(), &expected, negative);
    let consumed = consume(chain(&left_owner, &right_owner, &target_owner).unwrap()).unwrap();
    drop(left_owner);
    drop(right_owner);
    drop(target_owner);
    global::clear_thread_cache().unwrap();
    read(&consumed, &expected, negative);
    read(&old, &expected, negative);
    for owner in escaped {
        read(&owner, &expected, negative);
    }
}
pub fn verify(backend: global::PcuBackendChoice) {
    let _guard = POLICY_LOCK.lock().unwrap();
    for precision in [
        PcuPrecisionPolicy::Preserve,
        PcuPrecisionPolicy::BackendOptimized,
    ] {
        for compound_arithmetic in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for float_underflow in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                global::configure(global::PcuExecutionPolicy {
                    backend,
                    float_underflow,
                    numerical_options: PcuNumericalOptions {
                        precision,
                        compound_arithmetic,
                        ..Default::default()
                    },
                    ..Default::default()
                })
                .unwrap();
                global::clear_thread_cache().unwrap();
                format(
                    [0.0f32, 1.0, -1.0, 2.0, 3.0, f32::NAN, f32::from_bits(1)],
                    float_underflow,
                );
                format(
                    [0.0f64, 1.0, -1.0, 2.0, 3.0, f64::NAN, f64::from_bits(1)],
                    float_underflow,
                );
                format(
                    [0, 0x3c00, 0xbc00, 0x4000, 0x4200, 0x7e01, 1].map(PcuF16Bits::from_bits),
                    float_underflow,
                );
                format(
                    [0, 0x3f80, 0xbf80, 0x4000, 0x4040, 0x7fc1, 1].map(PcuBf16Bits::from_bits),
                    float_underflow,
                );
                format(
                    [0, 0x38, 0xb8, 0x40, 0x44, 0x7f, 1].map(PcuF8E4M3FnBits::from_bits),
                    float_underflow,
                );
                format(
                    [0, 0x3c, 0xbc, 0x40, 0x42, 0x7f, 1].map(PcuF8E5M2Bits::from_bits),
                    float_underflow,
                );
            }
        }
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
