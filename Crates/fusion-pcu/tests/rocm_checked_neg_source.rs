//! Actual annotated unary source against `ROCm`, including policy specialization and retry.
#![cfg(feature = "rocm")]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuRangePolicy,
};
#[pcu(invocations=N)]
fn negate_f32<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations=N)]
fn negate_f64<const N: usize>(input: &[f64; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result))]
fn reject_f32<const N: usize>(input: &[f32; N], output: &mut [f32; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
#[pcu(invocations=N,flag(reject_subnormal_result),flag(clamp_range))]
fn clamp_f64<const N: usize>(input: &[f64; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = -input[id];
}
fn configure(underflow: PcuFloatUnderflowPolicy, range: PcuRangePolicy) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Rocm,
        device: Some(0),
        block_size: 256,
        float_underflow: underflow,
        range_policy: range,
        ..Default::default()
    })
    .unwrap();
}
#[test]
#[ignore = "requires ROCm hardware; run serially"]
fn source_neg_preserves_bits_rejects_before_publish_recovers_and_retries() {
    configure(
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    global::clear_thread_cache().unwrap();
    let input = [
        0.0_f32,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::MAX,
        -f32::MAX,
    ];
    let mut output = [7.0_f32; 6];
    negate_f32(&input, &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        input.map(|value| value.to_bits() ^ 0x8000_0000)
    );
    output.fill(7.0);
    let error = reject_f32(&input, &mut output).unwrap_err();
    assert!(
        matches!(error,PcuExecutionError::ArithmeticFault(fault) if fault.kind==PcuExecutionFaultKind::ArithmeticUnderflow && fault.invocation_id==2 && !fault.recovered)
    );
    assert_eq!(output.map(f32::to_bits), [7.0_f32.to_bits(); 6]);
    // Explicit rejection specializes independently from the inherited default entry.
    negate_f32(&input, &mut output).unwrap();
    assert_eq!(
        output.map(f32::to_bits),
        input.map(|value| value.to_bits() ^ 0x8000_0000)
    );
    let finite = [
        0.0_f64,
        -0.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MAX,
        -f64::MAX,
    ];
    let mut output = [11.0_f64; 6];
    negate_f64(&finite, &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        finite.map(|value| value.to_bits() ^ (1 << 63))
    );
    let error = clamp_f64(&finite, &mut output).unwrap_err();
    let recovery = error.recovered_range_fault().unwrap();
    assert_eq!(recovery.kind, PcuExecutionFaultKind::ArithmeticUnderflow);
    assert_eq!(recovery.invocation_id, 2);
    assert!(recovery.recovered);
    assert_eq!(
        output.map(f64::to_bits),
        finite.map(|value| value.to_bits() ^ (1 << 63))
    );
    for invalid in [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff0_0000_0000_0001),
        f64::from_bits(0xfff8_0000_0000_0001),
    ] {
        let mut bad = finite;
        bad[4] = invalid;
        output.fill(11.0);
        let error = clamp_f64(&bad, &mut output).unwrap_err();
        assert!(
            matches!(error,PcuExecutionError::ArithmeticFault(fault) if fault.kind==PcuExecutionFaultKind::InvalidFloatingOperand && fault.invocation_id==4 && !fault.recovered)
        );
        assert_eq!(output.map(f64::to_bits), [11.0_f64.to_bits(); 6]);
        negate_f64(&finite, &mut output).unwrap();
        assert_eq!(
            output.map(f64::to_bits),
            finite.map(|value| value.to_bits() ^ (1 << 63))
        );
    }
    configure(
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Clamp,
    );
    let error = negate_f64(&finite, &mut output).unwrap_err();
    assert!(error.recovered_range_fault().unwrap().recovered);
    assert_eq!(
        output.map(f64::to_bits),
        finite.map(|value| value.to_bits() ^ (1 << 63))
    );
    configure(
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuRangePolicy::Reject,
    );
    negate_f64(&finite, &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        finite.map(|value| value.to_bits() ^ (1 << 63))
    );
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
}
