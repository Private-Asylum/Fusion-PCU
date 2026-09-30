//! Function-scoped arithmetic policy survives capture, preparation and warm reuse.
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
fn default_mul(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu(flag(ieee_underflow))]
fn pinned_ieee_mul(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu(flag(allow_gradual_underflow))]
fn gradual_mul(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu(flag(reject_subnormal_result))]
fn strict_mul(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[pcu(flag(allow_gradual_underflow))]
fn generic_gradual_mul<T: PcuScalar>(
    lhs: &[T],
    rhs: &[T],
) -> Result<PcuTensor<T>, PcuExecutionError> {
    Ok(lhs * rhs)
}

#[test]
fn unsupported_flag_specializations_fail_before_device_selection() {
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

#[pcu(flag(allow_gradual_underflow))]
fn gradual_caller_default_helper(
    lhs: &[f32],
    rhs: &[f32],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    default_mul(lhs, rhs)
}

#[pcu]
fn default_caller_gradual_helper(
    lhs: &[f32],
    rhs: &[f32],
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    gradual_mul(lhs, rhs)
}

#[pcu]
fn mixed_helper_policies(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let permitted = gradual_mul(lhs, rhs)?;
    let _strict_effect = strict_mul(lhs, rhs)?;
    Ok(permitted)
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

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn scoped_underflow_policies_keep_helper_contracts_and_warm_kernel_identity() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let tiny = [1.0, f32::from_bits(1)];
    let exact = [1.0, f32::MIN_POSITIVE];
    let factor = [1.0, 0.5];
    for _ in 0..2 {
        expect_fault(
            default_mul(&tiny, &factor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        expect_bits(gradual_mul(&tiny, &factor), [1.0_f32.to_bits(), 0]);
        expect_bits(generic_gradual_mul(&tiny, &factor), [1.0_f32.to_bits(), 0]);
        expect_bits(
            default_mul(&exact, &factor),
            [1.0_f32.to_bits(), 0x0040_0000],
        );
        expect_fault(
            strict_mul(&exact, &factor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        expect_fault(
            gradual_caller_default_helper(&tiny, &factor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        expect_bits(
            default_caller_gradual_helper(&tiny, &factor),
            [1.0_f32.to_bits(), 0],
        );
        expect_fault(
            mixed_helper_policies(&exact, &factor),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        // This flag changes only underflow. Other arithmetic faults remain mandatory.
        expect_fault(
            gradual_mul(&[1.0, f32::MAX], &[1.0, 2.0]),
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        expect_fault(
            gradual_mul(&[1.0, f32::INFINITY], &factor),
            PcuExecutionFaultKind::InvalidFloatingOperand,
        );
    }
    global::configure(global::PcuExecutionPolicy {
        float_underflow: fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
    // The same unannotated call site is rebuilt under the new global snapshot.
    expect_bits(default_mul(&tiny, &factor), [1.0_f32.to_bits(), 0]);
    expect_fault(
        pinned_ieee_mul(&tiny, &factor),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    expect_bits(
        pinned_ieee_mul(&exact, &factor),
        [1.0_f32.to_bits(), 0x0040_0000],
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
    expect_bits(gradual_mul(&tiny, &factor), [1.0_f32.to_bits(), 0]);
    global::clear_thread_cache().unwrap();
}
