//! Actual aggregate host/resident mixed-width immutable completion.
use super::{graph, Policy, Range};
use crate::{MlxBinaryInput, MlxDispatchCompletion, MlxPreparedDispatchKernel, MlxRuntime};
use fusion_pcu::{
    PcuDispatchCheckedFloatConversion, PcuHostArgument, PcuHostKernelBackend,
    PcuPreparedHostKernel, PcuScalarType,
};
#[test]
#[ignore = "Requires actual official MLX0.32.3 GPU aggregate."]
#[allow(clippy::too_many_lines)] // One retained map checks mixed metadata, immutable input, private completion and host publication.
fn conversion_aggregate_mixed_input_metadata_and_private_publication() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    let foreign = runtime.open_gpu(0).unwrap();
    for broadcast in [false, true] {
        graph::with(
            PcuDispatchCheckedFloatConversion::F64ToF32,
            3,
            Range::Clamp,
            Policy::IeeeAfterRounding,
            broadcast,
            false,
            |kernel| {
                let mut prepared = session.prepare_host_kernel(kernel).unwrap();
                assert!(matches!(prepared, MlxPreparedDispatchKernel::Conversion(_)));
                assert_eq!(
                    prepared.input_scalar_type(graph::INPUT),
                    Some(PcuScalarType::F64)
                );
                assert_eq!(prepared.input_scalar_type(graph::OUTPUT), None);
                assert_eq!(prepared.scalar_type(), PcuScalarType::F32);
                let input = if broadcast {
                    vec![f64::MAX]
                } else {
                    vec![1.0_f64, f64::MAX, -2.0]
                };
                let resident = session.upload_encoded(&input).unwrap();
                let raw: Vec<u8> = input
                    .iter()
                    .flat_map(|value| value.to_bits().to_le_bytes())
                    .collect();
                for mode in 0..2 {
                    let input_arg = if mode == 0 {
                        MlxBinaryInput::HostBytes {
                            target: graph::INPUT,
                            scalar: PcuScalarType::F64,
                            bytes: &raw,
                        }
                    } else {
                        MlxBinaryInput::Resident {
                            target: graph::INPUT,
                            array: &resident,
                        }
                    };
                    let MlxDispatchCompletion::Single(completion) =
                        prepared.execute_inputs(&[input_arg]).unwrap()
                    else {
                        panic!("one exact conversion output");
                    };
                    assert!(completion.recovered_fault().unwrap().recovered);
                    assert_eq!(completion.output().scalar_type(), PcuScalarType::F32);
                    let mut output = [91.0_f32; 5];
                    completion.output().read_into(&mut output).unwrap();
                    assert_eq!(
                        output.map(f32::to_bits),
                        (if broadcast {
                            [f32::MAX, f32::MAX, f32::MAX, 91.0, 91.0]
                        } else {
                            [1.0, f32::MAX, -2.0, 91.0, 91.0]
                        })
                        .map(f32::to_bits)
                    );
                    assert!(!prepared.last_call_may_have_written());
                    assert!(!prepared.last_call_completion_uncertain());
                }
                let wrong = foreign.upload_encoded(&input).unwrap();
                assert!(
                    prepared
                        .execute_inputs(&[MlxBinaryInput::Resident {
                            target: graph::INPUT,
                            array: &wrong
                        }])
                        .is_err()
                );
                let mut original = vec![0.0_f64; input.len()];
                resident.read_into(&mut original).unwrap();
                assert_eq!(
                    original
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>(),
                    input
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                );
                let mut host_output = [91.0_f32; 5];
                prepared
                    .call(&mut [
                        PcuHostArgument::read(graph::INPUT, &input),
                        PcuHostArgument::read_write(graph::OUTPUT, &mut host_output),
                    ])
                    .unwrap_err();
                assert!(prepared.last_call_may_have_written());
                assert_eq!(&host_output[3..], &[91.0, 91.0]);
                let published = host_output.map(f32::to_bits);
                let invalid = vec![f64::INFINITY; input.len()];
                prepared
                    .call(&mut [
                        PcuHostArgument::read(graph::INPUT, &invalid),
                        PcuHostArgument::read_write(graph::OUTPUT, &mut host_output),
                    ])
                    .unwrap_err();
                assert!(!prepared.last_call_may_have_written());
                assert_eq!(host_output.map(f32::to_bits), published);
                assert_eq!(&host_output[3..], &[91.0, 91.0]);
            },
        );
    }
}

#[test]
#[ignore = "Requires actual official MLX0.32.3 GPU aggregate."]
fn conversion_aggregate_all_canonical_mixed_width_layouts() {
    let runtime = MlxRuntime::load_default().unwrap();
    let session = runtime.open_gpu(0).unwrap();
    for direction in [
        PcuDispatchCheckedFloatConversion::F32ToF64,
        PcuDispatchCheckedFloatConversion::F64ToF32,
    ] {
        for range in [Range::Reject, Range::Clamp] {
            for policy in [
                Policy::IeeeAfterRounding,
                Policy::RejectSubnormalResult,
                Policy::AllowGradualUnderflow,
            ] {
                for broadcast in [false, true] {
                    for grid in [false, true] {
                        graph::with(direction, 3, range, policy, broadcast, grid, |kernel| {
                            if direction == PcuDispatchCheckedFloatConversion::F64ToF32 {
                                let input = if broadcast {
                                    vec![-0.0_f64]
                                } else {
                                    vec![-0.0_f64, 1.0, -2.0]
                                };
                                let expected = if broadcast {
                                    [-0.0_f32; 3]
                                } else {
                                    [-0.0_f32, 1.0, -2.0]
                                };
                                verify_layout(&session, kernel, &input, &expected, 91.0);
                            } else {
                                let input = if broadcast {
                                    vec![-0.0_f32]
                                } else {
                                    vec![-0.0_f32, 1.0, -2.0]
                                };
                                let expected = if broadcast {
                                    [-0.0_f64; 3]
                                } else {
                                    [-0.0_f64, 1.0, -2.0]
                                };
                                verify_layout(&session, kernel, &input, &expected, 91.0);
                            }
                        });
                    }
                }
            }
        }
    }
}
fn bytes<T: fusion_pcu::PcuScalar>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn verify_layout<I: fusion_pcu::PcuScalar, O: fusion_pcu::PcuScalar>(
    session: &crate::MlxSession,
    kernel: &fusion_pcu::PcuDispatchKernelIr<'_>,
    input: &[I],
    expected: &[O],
    sentinel: O,
) {
    let mut prepared = session.prepare_host_kernel(kernel).unwrap();
    let mut output = vec![sentinel; 5];
    prepared
        .call(&mut [
            PcuHostArgument::read(graph::INPUT, input),
            PcuHostArgument::read_write(graph::OUTPUT, &mut output),
        ])
        .unwrap();
    assert_eq!(bytes(&output[..3]), bytes(expected));
    assert_eq!(bytes(&output[3..]), bytes(&[sentinel; 2]));
    assert!(prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    let raw = bytes(input);
    let resident = session.upload_encoded(input).unwrap();
    for mode in 0..2 {
        let arg = if mode == 0 {
            MlxBinaryInput::HostBytes {
                target: graph::INPUT,
                scalar: I::TYPE,
                bytes: &raw,
            }
        } else {
            MlxBinaryInput::Resident {
                target: graph::INPUT,
                array: &resident,
            }
        };
        let MlxDispatchCompletion::Single(completion) = prepared.execute_inputs(&[arg]).unwrap()
        else {
            panic!("single conversion output");
        };
        assert!(completion.recovered_fault().is_none());
        let mut values = vec![sentinel; 5];
        completion.output().read_into(&mut values).unwrap();
        assert_eq!(bytes(&values[..3]), bytes(expected));
        assert_eq!(bytes(&values[3..]), bytes(&[sentinel; 2]));
        assert!(!prepared.last_call_may_have_written());
        assert!(!prepared.last_call_completion_uncertain());
    }
}
