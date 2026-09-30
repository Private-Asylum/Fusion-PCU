//! Strict checked binary32 source-to-device acceptance.
#![cfg(all(feature = "rocm", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuTensor,
};

#[pcu]
fn add(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs + rhs)
}
#[pcu]
fn sub(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs - rhs)
}
#[pcu]
fn mul(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs * rhs)
}
#[pcu]
fn discarded(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let _unused = lhs + rhs;
    pcu::identity(lhs)
}
#[pcu]
fn masked(lhs: &[f32], rhs: &[f32], zero: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::mul(&(lhs + rhs), zero)?)
}

fn fault(result: Result<PcuTensor<f32>, PcuExecutionError>, kind: PcuExecutionFaultKind) {
    match result {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, kind);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("expected terminal {kind:?}, got {other:?}"),
    }
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn owned_binary32_ieee_faults_preserve_effects_and_exact_subnormals() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
    let left = [1.0, f32::MAX];
    let right = [1.0, f32::MAX];
    fault(
        add(&left, &right),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        discarded(&left, &right),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        masked(&left, &right, &[0.0, 0.0]),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        sub(&[1.0, -f32::MAX], &right),
        PcuExecutionFaultKind::ArithmeticOverflow,
    );
    fault(
        mul(&[1.0, f32::from_bits(1)], &[1.0, 0.5]),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    fault(
        mul(
            &[1.0, f32::MIN_POSITIVE],
            &[1.0, f32::from_bits(0x3f7f_ffff)],
        ),
        PcuExecutionFaultKind::ArithmeticUnderflow,
    );
    fault(
        add(&[1.0, f32::INFINITY], &[1.0, 0.0]),
        PcuExecutionFaultKind::InvalidFloatingOperand,
    );
    let exact = mul(&[1.0, f32::MIN_POSITIVE], &[1.0, 0.5]).unwrap();
    let mut output = [0.0; 2];
    exact.read_into(&mut output).unwrap();
    assert_eq!(output.map(f32::to_bits), [1.0_f32.to_bits(), 0x0040_0000]);
    let retry = add(&[2.0, 3.0], &[1.0, 2.0]).unwrap();
    retry.read_into(&mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        [3.0_f32.to_bits(), 5.0_f32.to_bits()]
    );
    assert_eq!(
        left.map(f32::to_bits),
        [1.0_f32.to_bits(), f32::MAX.to_bits()]
    );
    global::clear_thread_cache().unwrap();
}
