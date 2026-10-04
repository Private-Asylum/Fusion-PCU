//! Supported tensor primitives keep the same operation law when arguments move.
#![cfg(all(feature = "cpu", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuCheckedFloat,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    dialect::tensor::{TensorError, TensorArithmeticStep},
};

#[pcu]
fn retain<T: PcuScalar>(input: &[[T; 2]; 2]) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::identity(input)
}

#[pcu]
fn quotient<T: PcuScalar>(
    input: PcuTensor<T>,
    divisor: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::div(input, divisor)
}

#[pcu]
fn backward<T: PcuScalar>(
    activation: PcuTensor<T>,
    gradient: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::relu_backward(activation, gradient)
}

#[pcu(flag(strict))]
fn loss<T: PcuScalar>(
    prediction: PcuTensor<T>,
    target: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::mean_squared_error(prediction, target)
}

#[pcu(flag(strict))]
fn update<T: PcuScalar>(
    weights: PcuTensor<T>,
    gradient: &[[T; 2]; 2],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    pcu::sgd_update(weights, gradient, 0.5_f32)
}

fn check<T: PcuScalar>(owner: &PcuTensor<T>, expected: &[T], sentinel: T) {
    let mut output = vec![sentinel; expected.len() + 2];
    owner.read_into(&mut output).unwrap();
    for (actual, wanted) in output[..expected.len()].iter().zip(expected) {
        assert_eq!(actual.encode_le().as_ref(), wanted.encode_le().as_ref());
    }
    for actual in &output[expected.len()..] {
        assert_eq!(actual.encode_le().as_ref(), sentinel.encode_le().as_ref());
    }
}

fn scalar<T: PcuCheckedFloat>(zero: T, one: T, two: T, negative: T) {
    let input = [[negative, zero], [one, two]];
    let gradient = [[two; 2]; 2];
    let older = retain(&input).unwrap();
    let divided = quotient(retain(&gradient).unwrap(), &gradient).unwrap();
    let masked = backward(retain(&input).unwrap(), &gradient).unwrap();
    check(&divided, &[one; 4], negative);
    check(&masked, &[zero, zero, two, two], negative);
    // A moved logical input must stay live through failure; nothing publishes a partial owner.
    let error = quotient(retain(&input).unwrap(), &[[zero; 2]; 2]).unwrap_err();
    let fault = error.arithmetic_fault().unwrap();
    assert_eq!(fault.kind, PcuExecutionFaultKind::DivideByZero);
    assert_eq!(fault.invocation_id, 0);
    assert!(!fault.recovered);
    check(&older, &[negative, zero, one, two], negative);
    let retry = backward(retain(&input).unwrap(), &gradient).unwrap();
    global::clear_thread_cache().unwrap();
    for owner in [&masked, &retry] {
        check(owner, &[zero, zero, two, two], negative);
    }
    check(&divided, &[one; 4], negative);
    check(&older, &[negative, zero, one, two], negative);
}

#[test]
fn six_float_consuming_division_and_backward_keep_ownership_and_faults() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    scalar(0_f32, 1.0, 2.0, -1.0);
    scalar(0_f64, 1.0, 2.0, -1.0);
    scalar(
        PcuF16Bits::from_bits(0),
        PcuF16Bits::from_bits(0x3c00),
        PcuF16Bits::from_bits(0x4000),
        PcuF16Bits::from_bits(0xbc00),
    );
    scalar(
        PcuBf16Bits::from_bits(0),
        PcuBf16Bits::from_bits(0x3f80),
        PcuBf16Bits::from_bits(0x4000),
        PcuBf16Bits::from_bits(0xbf80),
    );
    scalar(
        PcuF8E4M3FnBits::from_bits(0),
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0x40),
        PcuF8E4M3FnBits::from_bits(0xb8),
    );
    scalar(
        PcuF8E5M2Bits::from_bits(0),
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0x40),
        PcuF8E5M2Bits::from_bits(0xbc),
    );
    global::clear_thread_cache().unwrap();
}

fn compound<T: PcuScalar>(one: T, two: T, negative: T, invalid: T) {
    let input = [[two, two], [two, two]];
    let target = [[one; 2]; 2];
    let gradient = [[two; 2]; 2];
    let old = retain(&input).unwrap();
    let mse = loss(retain(&input).unwrap(), &target).unwrap();
    let updated = update(retain(&input).unwrap(), &gradient).unwrap();
    assert!(mse.shape().is_empty()); // MSE is a rank-zero scalar, not a length-one vector.
    check(&mse, &[one], negative);
    check(&updated, &[one; 4], negative);
    let mut bad = input;
    bad[1][1] = invalid;
    for (error, output_index, reduction_index, expected_step) in [
        (
            loss(retain(&bad).unwrap(), &target).unwrap_err(),
            0,
            3,
            TensorArithmeticStep::Subtract,
        ),
        (
            update(retain(&bad).unwrap(), &gradient).unwrap_err(),
            3,
            0,
            TensorArithmeticStep::Subtract,
        ),
    ] {
        let fault = error.arithmetic_fault().unwrap();
        assert_eq!(fault.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
        assert_eq!(fault.invocation_id, output_index);
        assert!(!fault.recovered);
        let PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
            element_index,
            reduction_index: actual_reduction,
            step,
            ..
        }) = error
        else {
            panic!("lost the detailed compound fault: {error:?}");
        };
        assert_eq!(element_index, usize::try_from(output_index).unwrap());
        assert_eq!(actual_reduction, reduction_index);
        assert_eq!(step, expected_step);
        check(&old, &[two; 4], negative);
    }
    check(
        &loss(retain(&input).unwrap(), &target).unwrap(),
        &[one],
        negative,
    );
    check(
        &update(retain(&input).unwrap(), &gradient).unwrap(),
        &[one; 4],
        negative,
    );
    global::clear_thread_cache().unwrap();
    check(&old, &[two; 4], negative);
    check(&mse, &[one], negative);
    check(&updated, &[one; 4], negative);
}

#[test]
fn consuming_strict_mse_and_sgd_execute_the_actual_compound_operations() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    compound(1_f32, 2.0, -1.0, f32::NAN);
    compound(1_f64, 2.0, -1.0, f64::NAN);
    global::clear_thread_cache().unwrap();
}
