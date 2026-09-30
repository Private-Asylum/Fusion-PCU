//! Function-scoped binary64 underflow policies survive capture, preparation and warm reuse.
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
fn default_mul(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs * rhs)
}
#[pcu(flag(ieee_underflow))]
fn pinned_ieee_mul(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs * rhs)
}
#[pcu(flag(allow_gradual_underflow))]
fn gradual_mul(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs * rhs)
}
#[pcu(flag(reject_subnormal_result))]
fn strict_mul(lhs: &[f64], rhs: &[f64]) -> Result<PcuTensor<f64>, PcuExecutionError> {
    Ok(lhs * rhs)
}
#[pcu(flag(allow_gradual_underflow))]
fn generic_gradual_mul<T: PcuScalar>(
    lhs: &[T],
    rhs: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(lhs * rhs)
}

fn expect_fault(result: Result<PcuTensor<f64>, PcuExecutionError>, kind: PcuExecutionFaultKind) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(fault)) => {
            assert_eq!(fault.kind, kind);
            assert_eq!(fault.invocation_id, 1);
        }
        other => panic!("expected {kind:?}, received {other:?}"),
    }
}

fn expect_bits(result: Result<PcuTensor<f64>, PcuExecutionError>, expected: [u64; 2]) {
    let result = result.unwrap();
    let mut host = [0.0; 2];
    result.read_into(&mut host).unwrap();
    assert_eq!(host.map(f64::to_bits), expected);
}

#[test]
fn generic_policy_flags_reject_integer_specializations_before_device_selection() {
    assert!(matches!(
        generic_gradual_mul(&[1_i32], &[1_i32]),
        Err(PcuExecutionError::TensorBuild(
            fusion_pcu::dialect::tensor::TensorError::UnsupportedScalarType {
                scalar_type: fusion_pcu::PcuScalarType::I32,
                ..
            }
        ))
    ));
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn scoped_binary64_policies_apply_all_modes_and_global_cache_rebuild() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let tiny = [1.0, f64::from_bits(1)];
    let exact = [1.0, f64::MIN_POSITIVE];
    let factor = [1.0, 0.5];
    let minnormal_edge = [1.0, f64::MIN_POSITIVE];
    let predecessor = [1.0, f64::from_bits(1.0_f64.to_bits() - 1)];

    for _ in 0..2 {
        expect_fault(
            default_mul(&tiny, &factor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        expect_bits(gradual_mul(&tiny, &factor), [1.0_f64.to_bits(), 0]);
        expect_bits(generic_gradual_mul(&tiny, &factor), [1.0_f64.to_bits(), 0]);
        expect_bits(
            default_mul(&exact, &factor),
            [1.0_f64.to_bits(), (1_u64 << 51)],
        );
        expect_fault(
            strict_mul(&exact, &factor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        expect_fault(
            default_mul(&minnormal_edge, &predecessor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        expect_bits(
            gradual_mul(&minnormal_edge, &predecessor),
            [1.0_f64.to_bits(), f64::MIN_POSITIVE.to_bits()],
        );
        expect_fault(
            gradual_mul(&[1.0, f64::MAX], &[1.0, 2.0]),
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        expect_fault(
            gradual_mul(&[1.0, f64::NAN], &factor),
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }

    global::configure(global::PcuExecutionPolicy {
        float_underflow: fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    expect_bits(default_mul(&tiny, &factor), [1.0_f64.to_bits(), 0]);
    expect_fault(
        pinned_ieee_mul(&tiny, &factor),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_fault(
        strict_mul(&exact, &factor),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    global::use_defaults().unwrap();
    expect_fault(
        default_mul(&tiny, &factor),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_bits(gradual_mul(&tiny, &factor), [1.0_f64.to_bits(), 0]);
    global::clear_thread_cache().unwrap();
}
