//! Owned source division lowers to checked F32/F64 division with scoped policy handling.
#![cfg(all(feature = "rocm", feature = "tensor"))]

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuScalar,
    PcuTensor,
};

#[pcu]
fn default_operator_div(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs / rhs)
}

#[pcu]
fn default_named_div(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::div(lhs, rhs)?)
}

#[pcu(flag(ieee_underflow))]
fn ieee_div(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs / rhs)
}

#[pcu(flag(allow_gradual_underflow))]
fn gradual_div(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::div(lhs, rhs)?)
}

#[pcu(flag(reject_subnormal_result))]
fn strict_div(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs / rhs)
}

#[pcu]
fn generic_div<T: PcuScalar>(lhs: &[T], rhs: &[T]) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(lhs / rhs)
}

#[pcu]
fn default_f64_operator_div(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs / rhs)
}
#[pcu]
fn default_f64_named_div(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(pcu::div(lhs, rhs)?)
}
#[pcu(flag(ieee_underflow))]
fn ieee_f64_div(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(pcu::div(lhs, rhs)?)
}
#[pcu(flag(allow_gradual_underflow))]
fn gradual_f64_div(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs / rhs)
}
#[pcu(flag(reject_subnormal_result))]
fn strict_f64_div(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(pcu::div(lhs, rhs)?)
}
#[pcu(flag(allow_gradual_underflow))]
fn generic_gradual_f64_div<T: PcuScalar>(
    lhs: &[T],
    rhs: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(lhs / rhs)
}

#[test]
fn integer_division_is_rejected_during_cold_capture() {
    assert!(matches!(
        generic_div(&[6_i32], &[2_i32]),
        Err(PcuExecutionError::TensorBuild(
            fusion_pcu::dialect::tensor::TensorError::UnsupportedScalarType {
                scalar_type: fusion_pcu::PcuScalarType::I32,
                ..
            }
        ))
    ));
    assert!(matches!(
        generic_gradual_f64_div(&[1_i32], &[1_i32]),
        Err(PcuExecutionError::TensorBuild(
            fusion_pcu::dialect::tensor::TensorError::UnsupportedScalarType {
                scalar_type: fusion_pcu::PcuScalarType::I32,
                ..
            }
        ))
    ));
}

fn expect_fault(result: Result<PcuTensor<f32>, PcuExecutionError>, kind: PcuExecutionFaultKind) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(fault)) => {
            assert_eq!(fault.kind, kind);
            assert_eq!(fault.invocation_id, 1);
        }
        other => panic!("expected {kind:?}, received {other:?}"),
    }
}

fn expect_bits(result: Result<PcuTensor<f32>, PcuExecutionError>, expected: [u32; 2]) {
    let result = result.unwrap();
    let mut host = [0.0; 2];
    result.read_into(&mut host).unwrap();
    assert_eq!(host.map(f32::to_bits), expected);
}

fn expect_f64_fault_at(
    result: Result<PcuTensor<f64>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
    invocation_id: u64,
) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(fault)) => {
            assert_eq!(fault.kind, kind);
            assert_eq!(fault.invocation_id, invocation_id);
        }
        other => panic!("expected {kind:?}, received {other:?}"),
    }
}

fn expect_f64_fault(
    result: Result<PcuTensor<f64>, PcuExecutionError>,
    kind: PcuExecutionFaultKind,
) {
    expect_f64_fault_at(result, kind, 1);
}

fn expect_f64_bits(result: Result<PcuTensor<f64>, PcuExecutionError>, expected: [u64; 2]) {
    let result = result.unwrap();
    let mut host = [0.0; 2];
    result.read_into(&mut host).unwrap();
    assert_eq!(host.map(f64::to_bits), expected);
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn checked_division_preserves_zero_faults_and_all_underflow_policies() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let ones = [1.0_f32, 1.0];
    let zero_divisors = [1.0_f32, 0.0];
    expect_fault(
        default_operator_div(&ones, &zero_divisors),
        PcuExecutionFaultKind::DivideByZero,
    );
    expect_fault(
        default_named_div(&ones, &zero_divisors),
        PcuExecutionFaultKind::DivideByZero,
    );

    let tiny = [1.0_f32, f32::from_bits(1)];
    let exact_half = [1.0_f32, 2.0];
    expect_fault(
        default_operator_div(&tiny, &exact_half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_fault(
        ieee_div(&tiny, &exact_half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_bits(gradual_div(&tiny, &exact_half), [1.0_f32.to_bits(), 0]);

    let min_normal = [1.0_f32, f32::MIN_POSITIVE];
    expect_fault(
        strict_div(&min_normal, &exact_half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );

    global::configure(global::PcuExecutionPolicy {
        float_underflow: fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    // A global policy change applies to the same unannotated operation on its next cold build.
    expect_bits(
        default_operator_div(&tiny, &exact_half),
        [1.0_f32.to_bits(), 0],
    );
    // An explicit source flag remains local even while the global default changes.
    expect_fault(
        ieee_div(&tiny, &exact_half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn checked_binary64_division_covers_boundaries_signs_and_fault_priority() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    // A generic helper remains executable for its supported F64 specialization.
    let generic_result = generic_gradual_f64_div(&[6.0_f64], &[2.0_f64]).unwrap();
    let mut generic_host = [0.0_f64];
    generic_result.read_into(&mut generic_host).unwrap();
    assert_eq!(generic_host.map(f64::to_bits), [3.0_f64.to_bits()]);

    let zeros = [0.0_f64, 0.0];
    expect_f64_fault_at(
        default_f64_operator_div(&zeros, &zeros),
        PcuExecutionFaultKind::DivideByZero,
        0,
    );
    expect_f64_fault_at(
        default_f64_named_div(&zeros, &zeros),
        PcuExecutionFaultKind::DivideByZero,
        0,
    );

    let min_subnormal = f64::from_bits(1);
    let min_normal_next = f64::from_bits(f64::MIN_POSITIVE.to_bits() + 1);
    // These quotients are intentionally asymmetric: the first rounds two ULPs below 2^-52;
    // the inverse is exactly 2^52 + 1.
    expect_f64_bits(
        default_f64_operator_div(
            &[min_subnormal, min_normal_next],
            &[min_normal_next, min_subnormal],
        ),
        [0x3caf_ffff_ffff_fffe, 0x4330_0000_0000_0001],
    );

    let tiny = [1.0_f64, min_subnormal];
    let half = [1.0_f64, 2.0];
    expect_f64_fault(
        default_f64_operator_div(&tiny, &half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_f64_fault(
        ieee_f64_div(&tiny, &half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_f64_bits(gradual_f64_div(&tiny, &half), [1.0_f64.to_bits(), 0]);

    let min_normal = [1.0_f64, f64::MIN_POSITIVE];
    // Exact subnormal results pass IEEE tininess and fail only under the strict policy.
    expect_f64_bits(
        default_f64_named_div(&min_normal, &half),
        [1.0_f64.to_bits(), 1_u64 << 51],
    );
    expect_f64_fault(
        strict_f64_div(&min_normal, &half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_f64_bits(
        generic_gradual_f64_div(&tiny, &half),
        [1.0_f64.to_bits(), 0],
    );

    // Zero results preserve the XOR of operand signs.
    expect_f64_bits(
        gradual_f64_div(&[-0.0, 0.0], &[2.0, -2.0]),
        [1_u64 << 63, 1_u64 << 63],
    );

    // Gradual underflow never masks overflow, invalid inputs, or zero divisors. Decode checks
    // nonfinite operands first, so infinity/zero reports InvalidFloatingOperand.
    expect_f64_fault_at(
        gradual_f64_div(&[f64::MAX, f64::INFINITY], &[min_subnormal, 0.0]),
        PcuExecutionFaultKind::ArithmeticOverflow,
        0,
    );
    expect_f64_fault_at(
        gradual_f64_div(&[f64::INFINITY, 1.0], &[0.0, 0.0]),
        PcuExecutionFaultKind::InvalidFloatingOperand,
        0,
    );

    global::configure(global::PcuExecutionPolicy {
        float_underflow: fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    expect_f64_bits(
        default_f64_operator_div(&tiny, &half),
        [1.0_f64.to_bits(), 0],
    );
    expect_f64_fault(
        ieee_f64_div(&tiny, &half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_f64_fault(
        strict_f64_div(&min_normal, &half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    global::use_defaults().unwrap();
    expect_f64_fault(
        default_f64_operator_div(&tiny, &half),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    global::clear_thread_cache().unwrap();
}
