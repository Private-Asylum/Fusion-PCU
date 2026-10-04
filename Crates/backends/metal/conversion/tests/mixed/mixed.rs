//! Native aggregate mixed-width resource and terminal publication proof.
use super::{graph, Policy, Range};
use crate::{
    MetalMemoryProvider, MetalMemoryResource, MetalMixedHostArgument, MetalPreparedHostKernel,
    MetalSession,
};
use fusion_pcu::{
    PcuDeviceArgument, PcuDeviceBuffer, PcuDispatchCheckedFloatConversion, PcuHostArgument,
    PcuHostKernelBackend, PcuMemoryAccess, PcuMemoryAllocationRequest, PcuMemoryHostAccess,
    PcuMemoryPoolId, PcuMemoryProvider, PcuScalar,
};
fn bytes<T: PcuScalar>(values: &[T]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect()
}
fn owner<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    values: &[T],
) -> PcuDeviceBuffer<T, MetalMemoryResource> {
    let data = bytes(values);
    let mut resource = provider
        .allocate(PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(121),
            size_bytes: u64::try_from(data.len()).unwrap(),
            alignment_bytes: 1,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        })
        .unwrap();
    provider.transfer_to(&mut resource, 0, &data).unwrap();
    PcuDeviceBuffer::new(resource, values.len())
}
fn read<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    value: &PcuDeviceBuffer<T, MetalMemoryResource>,
) -> Vec<u8> {
    let mut data = vec![0; value.len() * usize::from(T::TYPE.bit_width()) / 8];
    provider
        .transfer_from(value.resource(), 0, &mut data)
        .unwrap();
    data
}
#[test]
#[ignore = "Requires actual macOS Metal mixed-width publication."]
#[allow(clippy::too_many_lines)] // One retained prepared map verifies preflight, fatal publication and all four host/resident roles.
fn conversion_aggregate_mixed_roles_preflight_tail_and_fatal_publication() {
    let session = MetalSession::open(0).unwrap();
    let foreign = MetalSession::open(0).unwrap();
    let mut provider = session.memory_provider(PcuMemoryPoolId(121));
    let mut other = foreign.memory_provider(PcuMemoryPoolId(121));
    for grid in [false, true] {
        graph::with(
            PcuDispatchCheckedFloatConversion::F64ToF32,
            3,
            Range::Clamp,
            Policy::IeeeAfterRounding,
            false,
            grid,
            |kernel| {
                let mut prepared = session.prepare_host_kernel(kernel).unwrap();
                assert!(matches!(prepared, MetalPreparedHostKernel::Conversion(_)));
                let input = [1.0_f64, f64::MAX, -2.0, 777.0];
                let resident_input = owner(&mut provider, &input);
                for mode in 0..4 {
                    let mut host_output = [91.0_f32; 5];
                    let mut resident_output = owner(&mut provider, &host_output);
                    let input_arg = if mode & 1 == 0 {
                        MetalMixedHostArgument::Host(PcuHostArgument::read(graph::INPUT, &input))
                    } else {
                        MetalMixedHostArgument::Resident(PcuDeviceArgument::read(
                            graph::INPUT,
                            &resident_input,
                        ))
                    };
                    let output_arg = if mode & 2 == 0 {
                        MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                            graph::OUTPUT,
                            &mut host_output,
                        ))
                    } else {
                        MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                            graph::OUTPUT,
                            &mut resident_output,
                        ))
                    };
                    assert!(prepared.call_mixed(&mut [output_arg, input_arg]).is_err());
                    assert!(prepared.last_call_may_have_written());
                    assert!(!prepared.last_call_completion_uncertain());
                    let output = if mode & 2 == 0 {
                        bytes(&host_output)
                    } else {
                        read(&mut provider, &resident_output)
                    };
                    assert_eq!(output, bytes(&[1.0_f32, f32::MAX, -2.0, 91.0, 91.0]));
                    assert_eq!(read(&mut provider, &resident_input), bytes(&input));
                }
                let wrong = owner(&mut other, &[1.0_f64, 2.0, 3.0]);
                let mut output = [91.0_f32; 5];
                assert!(
                    prepared
                        .call_mixed(&mut [
                            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(
                                graph::INPUT,
                                &wrong
                            )),
                            MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                                graph::OUTPUT,
                                &mut output
                            )),
                        ])
                        .is_err()
                );
                assert!(!prepared.last_call_may_have_written());
                assert_eq!(bytes(&output), bytes(&[91.0_f32; 5]));
                let invalid = [1.0_f64, f64::INFINITY, -2.0];
                assert!(
                    prepared
                        .call_mixed(&mut [
                            MetalMixedHostArgument::Host(PcuHostArgument::read(
                                graph::INPUT,
                                &invalid
                            )),
                            MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                                graph::OUTPUT,
                                &mut output
                            )),
                        ])
                        .is_err()
                );
                assert!(!prepared.last_call_may_have_written());
                assert_eq!(bytes(&output), bytes(&[91.0_f32; 5]));
                let mut resident_output = owner(&mut provider, &output);
                assert!(
                    prepared
                        .call_mixed(&mut [
                            MetalMixedHostArgument::Host(PcuHostArgument::read(
                                graph::INPUT,
                                &invalid
                            )),
                            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                                graph::OUTPUT,
                                &mut resident_output
                            )),
                        ])
                        .is_err()
                );
                assert!(!prepared.last_call_may_have_written());
                assert!(!prepared.last_call_completion_uncertain());
                assert_eq!(read(&mut provider, &resident_output), bytes(&output));
                prepared
                    .call_mixed(&mut [
                        MetalMixedHostArgument::Host(PcuHostArgument::read(
                            graph::INPUT,
                            &[1.0_f64, 2.0, 3.0],
                        )),
                        MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                            graph::OUTPUT,
                            &mut resident_output,
                        )),
                    ])
                    .unwrap();
                assert!(prepared.last_call_may_have_written());
                assert_eq!(
                    read(&mut provider, &resident_output),
                    bytes(&[1.0_f32, 2.0, 3.0, 91.0, 91.0]),
                );
            },
        );
    }
}

fn verify_healthy<I: PcuScalar, O: PcuScalar>(
    session: &MetalSession,
    provider: &mut MetalMemoryProvider,
    kernel: &fusion_pcu::PcuDispatchKernelIr<'_>,
    input: &[I],
    expected: &[O],
    sentinel: O,
) {
    let mut prepared = session.prepare_host_kernel(kernel).unwrap();
    let resident_input = owner(provider, input);
    for mode in 0..4 {
        let mut host_output = vec![sentinel; 5];
        let mut resident_output = owner(provider, &host_output);
        let input_arg = if mode & 1 == 0 {
            MetalMixedHostArgument::Host(PcuHostArgument::read(graph::INPUT, input))
        } else {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(graph::INPUT, &resident_input))
        };
        let output_arg = if mode & 2 == 0 {
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(
                graph::OUTPUT,
                &mut host_output,
            ))
        } else {
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                graph::OUTPUT,
                &mut resident_output,
            ))
        };
        prepared.call_mixed(&mut [input_arg, output_arg]).unwrap();
        assert!(prepared.last_call_may_have_written());
        let actual = if mode & 2 == 0 {
            bytes(&host_output)
        } else {
            read(provider, &resident_output)
        };
        assert_eq!(
            actual,
            bytes(
                &expected
                    .iter()
                    .copied()
                    .chain([sentinel; 2])
                    .collect::<Vec<_>>()
            )
        );
        assert_eq!(read(provider, &resident_input), bytes(input));
    }
}
#[test]
#[ignore = "Requires actual macOS Metal mixed-width publication."]
fn conversion_aggregate_all_canonical_mixed_width_layouts() {
    let session = MetalSession::open(0).unwrap();
    let mut provider = session.memory_provider(PcuMemoryPoolId(121));
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
                            if direction == PcuDispatchCheckedFloatConversion::F32ToF64 {
                                let input = if broadcast {
                                    vec![-0.0_f32]
                                } else {
                                    vec![-0.0_f32, 1.0, -2.0]
                                };
                                let expected = if broadcast {
                                    vec![-0.0_f64; 3]
                                } else {
                                    vec![-0.0_f64, 1.0, -2.0]
                                };
                                verify_healthy(
                                    &session,
                                    &mut provider,
                                    kernel,
                                    &input,
                                    &expected,
                                    91.0,
                                );
                            } else {
                                let input = if broadcast {
                                    vec![-0.0_f64]
                                } else {
                                    vec![-0.0_f64, 1.0, -2.0]
                                };
                                let expected = if broadcast {
                                    vec![-0.0_f32; 3]
                                } else {
                                    vec![-0.0_f32, 1.0, -2.0]
                                };
                                verify_healthy(
                                    &session,
                                    &mut provider,
                                    kernel,
                                    &input,
                                    &expected,
                                    91.0,
                                );
                            }
                        });
                    }
                }
            }
        }
    }
}
