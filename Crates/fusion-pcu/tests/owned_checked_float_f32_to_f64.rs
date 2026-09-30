//! Strict checked binary32-to-binary64 source-to-device acceptance.
#![cfg(all(feature = "rocm", feature = "tensor"))]
#[rustfmt::skip]
use fusion_pcu::{
    global,
    pcu,
    PcuExecutionError,
    PcuExecutionFaultKind,
};

#[pcu(invocations: N)]
fn checked_widen<const N: usize>(input: &[f32; N], output: &mut [f64; N]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] as f64;
}

#[test]
#[ignore = "requires a working ROCm device and HIPRTC"]
fn owned_f32_to_f64_checked_cast_preserves_exact_bits_faults_and_retry() {
    global::use_defaults().unwrap();
    global::clear_thread_cache().unwrap();

    let inputs = [
        0.0_f32,
        f32::from_bits(1 << 31),
        f32::from_bits(1),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        1.0,
        f32::MAX,
    ];
    let mut output = [0.0_f64; 7];
    checked_widen(&inputs, &mut output).unwrap();
    assert_eq!(
        output.map(f64::to_bits),
        inputs.map(|value| f64::from(value).to_bits())
    );

    let mut state = 0x9e37_79b9_u32;
    let mut random_inputs = [0.0_f32; 512];
    for value in &mut random_inputs {
        loop {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = state;
            if ((bits >> 23) & 0xff) != 0xff {
                *value = f32::from_bits(bits);
                break;
            }
        }
    }
    let mut random_output = [0.0_f64; 512];
    checked_widen(&random_inputs, &mut random_output).unwrap();
    assert_eq!(
        random_output.map(f64::to_bits),
        random_inputs.map(|value| f64::from(value).to_bits())
    );

    let mut fault_output = [0.0_f64; 2];
    match checked_widen(&[1.0, f32::INFINITY], &mut fault_output) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("expected non-finite widening fault, got {other:?}"),
    }
    match checked_widen(&[1.0, f32::NAN], &mut fault_output) {
        Err(PcuExecutionError::ArithmeticFault(record)) => {
            assert_eq!(record.kind, PcuExecutionFaultKind::InvalidFloatingOperand);
            assert_eq!(record.invocation_id, 1);
        }
        other => panic!("expected NaN widening fault, got {other:?}"),
    }

    let mut retry = [0.0_f64; 1];
    checked_widen(&[f32::from_bits(1)], &mut retry).unwrap();
    assert_eq!(retry[0].to_bits(), f64::from(f32::from_bits(1)).to_bits());
    global::clear_thread_cache().unwrap();
}
