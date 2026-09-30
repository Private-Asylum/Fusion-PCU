//! Invocation-level clamped range recovery preserves completed host output and reports recovery.
#![cfg(feature = "rocm")]

#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
    PcuRangePolicy,
};

#[cfg(feature = "tensor")]
use fusion_pcu::PcuTensor;

#[pcu(invocations = N, flag(clamp_range))]
fn explicit_clamp<const N: usize>(input: &[f64; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0_f64 + 0.0_f64;
}

#[pcu(invocations = N)]
fn inherited_clamp<const N: usize>(input: &[f64; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0_f64 + 0.0_f64;
}

#[cfg(feature = "tensor")]
#[pcu]
fn owned_add(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(lhs + rhs)
}

#[cfg(feature = "tensor")]
#[test]
fn global_clamp_is_rejected_for_owned_tensor_results() {
    global::configure(global::PcuExecutionPolicy {
        range_policy: PcuRangePolicy::Clamp,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let error = owned_add(&[1.0], &[2.0]).unwrap_err();
    assert!(matches!(error, PcuExecutionError::UnsupportedRangePolicy));
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
}

#[test]
fn explicit_clamp_flag_is_captured_in_manual_ir() {
    let bindings = explicit_clamp_bindings();
    let kernel = explicit_clamp_ir::<2>(&bindings).unwrap();
    kernel.with_ir(|ir| {
        assert!(ir.ops.iter().any(|op| matches!(
            op,
            fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary {
                range_policy: PcuRangePolicy::Clamp,
                ..
            })
        )));
    });
}

#[test]
#[ignore = "requires a working ROCm device"]
fn clamp_range_returns_completed_output_and_recovery_metadata() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let overflow = [f64::MAX, 2.0_f64];
    let mut output = [-17.0_f64; 2];
    let error = explicit_clamp(&overflow, &mut output).unwrap_err();
    let recovery = error
        .recovered_range_fault()
        .expect("clamped range result is reported as a recovered error");
    assert_eq!(recovery.kind, PcuExecutionFaultKind::ArithmeticOverflow);
    assert_eq!(recovery.invocation_id, 0);
    assert!(recovery.recovered);
    assert_eq!(
        output.map(f64::to_bits),
        [f64::MAX.to_bits(), 4.0_f64.to_bits()]
    );

    global::configure(global::PcuExecutionPolicy {
        range_policy: PcuRangePolicy::Clamp,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let inherited_input = [3.0_f64, f64::MAX];
    let mut inherited_output = [0.0_f64; 2];
    let error = inherited_clamp(&inherited_input, &mut inherited_output).unwrap_err();
    assert_eq!(
        error.recovered_range_fault().map(|fault| fault.kind),
        Some(PcuExecutionFaultKind::ArithmeticOverflow)
    );
    assert_eq!(
        inherited_output.map(f64::to_bits),
        [6.0_f64.to_bits(), f64::MAX.to_bits()]
    );

    // A later fatal lane outranks the earlier recoverable lane, and staging is not published.
    let fatal_input = [f64::MAX, f64::INFINITY];
    let mut fatal_output = [-23.0_f64; 2];
    let error = inherited_clamp(&fatal_input, &mut fatal_output).unwrap_err();
    assert!(error.recovered_range_fault().is_none());
    match error {
        PcuExecutionError::ArithmeticFault(fault) => {
            assert_eq!(fault.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
            assert_eq!(fault.invocation_id, 1);
            assert!(!fault.recovered);
        }
        other => panic!("expected fatal arithmetic fault, found {other:?}"),
    }
    assert_eq!(fatal_output.map(f64::to_bits), [(-23.0_f64).to_bits(); 2]);

    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();
}
