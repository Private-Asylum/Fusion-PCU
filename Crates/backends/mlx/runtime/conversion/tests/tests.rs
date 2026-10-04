//! Genuine retained MLX conversion, independent core integer law and immutable input proof.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{PcuClampedError,PcuClampedFloatConversion,PcuCheckedFloatWidening,PcuExecutionFaultKind};
#[test]
#[ignore = "Requires actual official MLX0.32.3 GPU runtime."]
#[allow(clippy::too_many_lines)] // One retained-session oracle covers faults/retries/input ownership across the complete conversion tuple.
fn mlx_checked_conversion_recovery_fatal_retry_and_original_owner_oracle() {
    let runtime = crate::MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    let mut bits = vec![
        0,
        1,
        0x8000_0000_0000_0000,
        0x7ff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x47ef_ffff_f000_0000,
        0x47ef_ffff_ffff_ffff,
        0x380f_ffff_e000_0000,
        0x380f_ffff_f000_0000,
        0x3690_0000_0000_0000,
        0x36a0_0000_0000_0000,
        0x3810_0000_0000_0000,
    ];
    let mut state = 0x1234_5678_8765_4321_u64;
    for _ in 0..64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        bits.push(state);
    }
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            for broadcast in [false, true] {
                let count = if broadcast { 5 } else { 1 };
                let prepared = session
                    .prepare_float_conversion(
                        PcuDispatchCheckedFloatConversion::F64ToF32,
                        policy,
                        range,
                        count,
                        broadcast,
                    )
                    .unwrap();
                let wrong = foreign.upload_encoded(&[1.0_f64]).unwrap();
                assert!(matches!(
                    prepared.execute_resident(&wrong),
                    Err(MlxError::ForeignSession)
                ));
                let short = session.upload_encoded(&[1.0_f32]).unwrap();
                assert!(matches!(
                    prepared.execute_resident(&short),
                    Err(MlxError::InvalidExtent)
                ));
                for &raw in &bits {
                    let input = session.upload_encoded(&[f64::from_bits(raw)]).unwrap();
                    let result = prepared.execute_resident(&input);
                    match f64::from_bits(raw).pcu_clamped_to_f32_with_policy(policy) {
                        Ok(value) => {
                            let completion = result.unwrap();
                            assert!(completion.recovered_fault().is_none());
                            let mut output = vec![91.0_f32; count + 2];
                            completion.output().read_into(&mut output).unwrap();
                            assert!(
                                output[..count]
                                    .iter()
                                    .all(|actual| actual.to_bits() == value.to_bits())
                            );
                            assert_eq!(
                                output[count..]
                                    .iter()
                                    .map(|v| v.to_bits())
                                    .collect::<Vec<_>>(),
                                vec![91.0_f32.to_bits(); 2]
                            );
                        }
                        Err(PcuClampedError::Range(fault)) if range == PcuRangePolicy::Clamp => {
                            let completion = result.unwrap();
                            let notice = completion.recovered_fault().unwrap();
                            assert!(notice.recovered);
                            assert_eq!(notice.kind, fault.kind());
                            let expected = fault.clamped_value().to_bits();
                            let mut output = vec![0.0_f32; count];
                            completion.output().read_into(&mut output).unwrap();
                            assert!(output.iter().all(|actual| actual.to_bits() == expected));
                        }
                        Err(error) => assert!(
                            matches!(result,Err(MlxError::Arithmetic(fault)) if !fault.recovered&&fault.kind==error.kind())
                        ),
                    }
                    let mut preserved = [0.0_f64];
                    input.read_into(&mut preserved).unwrap();
                    assert_eq!(preserved[0].to_bits(), raw);
                }
            }
        }
    }
    let prepared = session
        .prepare_float_conversion(
            PcuDispatchCheckedFloatConversion::F32ToF64,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuRangePolicy::Clamp,
            1,
            false,
        )
        .unwrap();
    for raw in [0, 1, 0x8000_0000, 0x7f7f_ffff, 0x7f80_0000, 0x7fc0_0001] {
        let input = session.upload_encoded(&[f32::from_bits(raw)]).unwrap();
        match f32::from_bits(raw).pcu_checked_to_f64() {
            Ok(value) => {
                let completion = prepared.execute_resident(&input).unwrap();
                let mut output = [0.0_f64];
                completion.output().read_into(&mut output).unwrap();
                assert_eq!(output[0].to_bits(), value.to_bits());
            }
            Err(kind) => {
                assert_eq!(kind, PcuExecutionFaultKind::InvalidFloatingOperand);
                assert!(
                    matches!(prepared.execute_resident(&input),Err(MlxError::Arithmetic(fault)) if fault.kind==kind&&!fault.recovered)
                );
            }
        }
    }
}
