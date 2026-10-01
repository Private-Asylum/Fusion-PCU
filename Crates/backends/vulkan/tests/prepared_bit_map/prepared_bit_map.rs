//! Actual per-function source acceptance on a selected Vulkan compute device.

#[path = "../support/support.rs"]
mod support;
#[rustfmt::skip]
use pcu_facade::{
    pcu,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuScalarType,
    PcuCostBoundary,
    PcuDeviceActivation,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuExecutorId,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuReproducibility,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanDiscovery,
};

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn transport<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

fn compare<const N: usize>(backend: &PcuVulkanBackend) {
    let mut state = 0x243f_6a88_u32;
    let input: Vec<f32> = (0..N)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            let bits = if index < 8 {
                [
                    0,
                    0x8000_0000,
                    1,
                    0x8000_0001,
                    0x7f7f_ffff,
                    0xff7f_ffff,
                    0x3f80_0000,
                    0xbf80_0000,
                ][index]
            } else if state & 0x7f80_0000 == 0x7f80_0000 {
                state ^ 0x0080_0000
            } else {
                state
            };
            f32::from_bits(bits)
        })
        .collect();
    let mut output = vec![17.0; N + 3];
    let mut call = negate_prepare::<N, _>(backend).expect("source preparation");
    call(&input, &mut output).expect("checked exact Neg");
    for (value, result) in input.iter().zip(&output) {
        assert_eq!(result.to_bits(), value.pcu_checked_neg().unwrap().to_bits());
    }
    assert_eq!(&output[N..], &[17.0; 3]);
    call(&vec![2.0; N], &mut output).unwrap();
    assert_eq!(&output[..N], &vec![-2.0; N]);
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn actual_source_matches_core_bits_reuses_storage_and_preserves_padded_tails() {
    let backend = PcuVulkanBackend::new().expect("Vulkan device");
    compare::<1>(&backend);
    compare::<3>(&backend);
    compare::<63>(&backend);
    compare::<64>(&backend);
    compare::<65>(&backend);
    compare::<257>(&backend);
    compare::<1025>(&backend);
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn actual_source_reports_first_fault_preserves_output_and_retries() {
    let backend = PcuVulkanBackend::new().unwrap();
    let mut call = negate_prepare::<257, _>(&backend).unwrap();
    let mut input = [1.0_f32; 257];
    let mut output = [19.0_f32; 260];
    for bits in [0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001] {
        input[7] = f32::from_bits(bits);
        input[256] = f32::NAN;
        assert!(matches!(
            call(&input, &mut output),
            Err(PcuVulkanError::Fault(PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: 7,
            }))
        ));
        assert_eq!(output.map(f32::to_bits), [19.0_f32; 260].map(f32::to_bits));
    }
    input.fill(1.0);
    call(&input, &mut output).unwrap();
    assert_eq!(&output[..257], &[-1.0; 257]);
    assert_eq!(&output[257..], &[19.0; 3]);
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn exact_transport_preserves_all_patterns_and_retains_backend_after_drop() {
    let backend = PcuVulkanBackend::new().unwrap();
    let mut call = transport_prepare::<8, _>(&backend).unwrap();
    drop(backend);
    let input = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0001,
        0x7f80_0001,
    ]
    .map(f32::from_bits);
    let mut output = [23.0; 11];
    call(&input, &mut output).unwrap();
    assert_eq!(
        output[..8]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        input.map(f32::to_bits)
    );
    assert_eq!(&output[8..], &[23.0; 3]);
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn detached_policy_grid_schema_and_short_call_admission_are_transactional() {
    let backend = PcuVulkanBackend::new().unwrap();
    for grid in [false, true] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let mut prepared =
                support::prepare_graph(&backend, 257, policy, PcuRangePolicy::Reject, grid)
                    .unwrap();
            let mut input = [1.0_f32; 257];
            input[9] = f32::from_bits(1);
            let mut output = [31.0_f32; 260];
            let result = prepared.call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
            ]);
            if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                assert!(matches!(
                    result,
                    Err(PcuVulkanError::Fault(PcuExecutionFault {
                        recovered: false,
                        kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                        invocation_id: 9,
                    }))
                ));
                assert_eq!(output.map(f32::to_bits), [31.0_f32; 260].map(f32::to_bits));
            } else {
                result.unwrap();
                assert_eq!(output[9].to_bits(), 0x8000_0001);
                assert_eq!(&output[257..], &[31.0; 3]);
            }
            assert!(
                support::prepare_graph(&backend, 257, policy, PcuRangePolicy::Clamp, grid).is_err()
            );
        }
    }
    let mut call = negate_prepare::<8, _>(&backend).unwrap();
    let mut output = [37.0; 8];
    assert!(matches!(
        call(&[1.0; 7], &mut output),
        Err(PcuVulkanError::InvalidArguments)
    ));
    assert_eq!(output.map(f32::to_bits), [37.0_f32; 8].map(f32::to_bits));
}

#[test]
#[ignore = "requires native Vulkan discovery and explicit physical device activation"]
fn discovery_offer_admission_is_bounded_exact_and_generation_scoped() {
    let discovery = PcuVulkanDiscovery::discover().unwrap();
    let empty = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };
    let readiness = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness,
    }];
    assert_eq!(discovery.providers(&mut []).unwrap(), 1);
    discovery.providers(&mut providers).unwrap();
    let mut targets = [PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [PcuDeviceDescriptor {
        reference: empty,
        target: empty,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }];
    assert!(
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap()
            > 0
    );
    let selected = devices[0].reference;
    let facts = discovery.device_facts(selected).unwrap();
    assert!(facts.stable_identity.is_some());
    assert!(facts.max_workgroup_invocations.is_some());
    assert!(discovery.native_caps(selected).unwrap().api_version > 0);
    for wrong in [
        PcuObjectRef {
            provider: PcuProviderId(3),
            ..selected
        },
        PcuObjectRef {
            generation: selected.generation + 1,
            ..selected
        },
        PcuObjectRef {
            kind: PcuObjectKind::Context,
            ..selected
        },
        PcuObjectRef {
            id: u32::MAX,
            ..selected
        },
    ] {
        assert!(discovery.device_facts(wrong).is_err());
        assert!(discovery.open_device(wrong).is_err());
    }
    let backend = PcuVulkanBackend::open(&discovery, selected).unwrap();
    assert_offers(&backend);
    let new_snapshot = PcuVulkanDiscovery::discover().unwrap();
    assert!(new_snapshot.open_device(selected).is_err());
}

fn assert_offers(backend: &PcuVulkanBackend) {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        if scalar == PcuScalarType::F64 && !backend.caps().shader_float64 {
            continue;
        }
        support::with_typed_graph(
            257,
            PcuFloatUnderflowPolicy::default(),
            PcuRangePolicy::Reject,
            false,
            scalar,
            |kernel| {
                for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    let request = PcuImplementationRequest {
                        device: backend.device_identity().unwrap(),
                        executor: PcuExecutorId(0),
                        boundary: PcuCostBoundary::Host,
                        operation: kernel,
                        requirements: PcuImplementationRequirements {
                            numerical_mode,
                            ..PcuImplementationRequirements::default()
                        },
                    };
                    assert_eq!(backend.implementation_offers(&request, &mut []).unwrap(), 1);
                    let mut output = [None];
                    assert_eq!(
                        backend
                            .implementation_offers(&request, &mut output)
                            .unwrap(),
                        1
                    );
                    let offer = output[0].unwrap();
                    assert_eq!(offer.implementation.revision, 3);
                    assert_eq!(
                        offer.implementation.local_id,
                        if scalar == PcuScalarType::F64 { 3 } else { 1 }
                    );
                    assert_eq!(offer.workspace_bytes, None);
                    assert!(offer.cost.completion.is_none());
                    offer.validate_request(&request).unwrap();
                    let deterministic = PcuImplementationRequest {
                        requirements: PcuImplementationRequirements {
                            numerical_options: PcuNumericalOptions {
                                reproducibility: PcuReproducibility::PortableV1,
                                ..PcuNumericalOptions::default()
                            },
                            ..request.requirements
                        },
                        ..request
                    };
                    assert_eq!(
                        backend
                            .implementation_offers(&deterministic, &mut [])
                            .unwrap(),
                        0
                    );
                }
                let resident = PcuImplementationRequest {
                    device: backend.device_identity().unwrap(),
                    executor: PcuExecutorId(0),
                    boundary: PcuCostBoundary::Resident,
                    operation: kernel,
                    requirements: PcuImplementationRequirements::default(),
                };
                assert_eq!(
                    backend.implementation_offers(&resident, &mut []).unwrap(),
                    0
                );
            },
        );
    }
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn transport_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = input[id];
}

fn compare_f64<const N: usize>(backend: &PcuVulkanBackend) {
    let mut state = 0x243f_6a88_85a3_08d3_u64;
    let input: Vec<f64> = (0..N)
        .map(|index| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let bits = if index < 10 {
                [
                    0,
                    0x8000_0000_0000_0000,
                    1,
                    0x8000_0000_0000_0001,
                    0x000f_ffff_ffff_ffff,
                    0x0010_0000_0000_0000,
                    0x7fef_ffff_ffff_ffff,
                    0xffef_ffff_ffff_ffff,
                    0x3ff0_0000_0000_0000,
                    0xbff0_0000_0000_0000,
                ][index]
            } else if state & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 {
                state ^ 0x0010_0000_0000_0000
            } else {
                state
            };
            f64::from_bits(bits)
        })
        .collect();
    let mut output = vec![17.0_f64; N + 3];
    let mut call = negate_f64_prepare::<N, _>(backend).unwrap();
    call(&input, &mut output).unwrap();
    for (value, actual) in input.iter().zip(&output) {
        assert_eq!(actual.to_bits(), value.to_bits() ^ 0x8000_0000_0000_0000);
    }
    assert_eq!(&output[N..], &[17.0; 3]);
    call(&vec![2.0; N], &mut output).unwrap();
    assert_eq!(&output[..N], &vec![-2.0; N]);
}

#[test]
#[ignore = "requires an explicitly available Vulkan shaderFloat64 compute device"]
fn f64_source_exact_edges_random_bits_tails_fault_rollback_and_retry() {
    let backend = PcuVulkanBackend::new().unwrap();
    assert!(backend.caps().shader_float64);
    compare_f64::<1>(&backend);
    compare_f64::<3>(&backend);
    compare_f64::<63>(&backend);
    compare_f64::<64>(&backend);
    compare_f64::<65>(&backend);
    compare_f64::<257>(&backend);
    compare_f64::<1025>(&backend);
    let mut call = negate_f64_prepare::<257, _>(&backend).unwrap();
    let mut input = [1.0_f64; 257];
    let mut output = [19.0_f64; 260];
    for bits in [
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7ff0_0000_0000_0001,
        0xfff0_0000_0000_0001,
    ] {
        input[7] = f64::from_bits(bits);
        input[256] = f64::NAN;
        assert!(matches!(
            call(&input, &mut output),
            Err(PcuVulkanError::Fault(PcuExecutionFault {
                recovered: false,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: 7,
            }))
        ));
        assert_eq!(output.map(f64::to_bits), [19.0_f64; 260].map(f64::to_bits));
    }
    input.fill(1.0);
    call(&input, &mut output).unwrap();
    assert_eq!(&output[..257], &[-1.0; 257]);
    assert_eq!(&output[257..], &[19.0; 3]);
}

#[test]
#[ignore = "requires an explicitly available Vulkan shaderFloat64 compute device"]
fn f64_direct_grid_all_underflow_policies_and_transport_are_exact() {
    let backend = PcuVulkanBackend::new().unwrap();
    for grid in [false, true] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let mut prepared = support::with_typed_graph(
                257,
                policy,
                PcuRangePolicy::Reject,
                grid,
                pcu_facade::PcuScalarType::F64,
                |kernel| backend.prepare_host_kernel(kernel),
            )
            .unwrap();
            let mut input = [1.0_f64; 257];
            let mut output = [31.0_f64; 260];
            // A nonzero low word alone must trigger the F64 subnormal test.
            for bits in [
                1,
                0x8000_0000_0000_0001,
                0x0000_0001_0000_0000,
                0x000f_ffff_ffff_ffff,
            ] {
                input[9] = f64::from_bits(bits);
                let result = prepared.call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                ]);
                if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                    assert!(matches!(
                        result,
                        Err(PcuVulkanError::Fault(PcuExecutionFault {
                            recovered: false,
                            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                            invocation_id: 9,
                        }))
                    ));
                    assert_eq!(output.map(f64::to_bits), [31.0_f64; 260].map(f64::to_bits));
                } else {
                    result.unwrap();
                    assert_eq!(output[9].to_bits(), bits ^ 0x8000_0000_0000_0000);
                    assert_eq!(&output[257..], &[31.0; 3]);
                }
            }
            input.fill(1.0);
            prepared
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                ])
                .unwrap();
            assert_eq!(&output[..257], &[-1.0; 257]);
            assert!(
                support::with_typed_graph(
                    257,
                    policy,
                    PcuRangePolicy::Clamp,
                    grid,
                    pcu_facade::PcuScalarType::F64,
                    |kernel| backend.prepare_host_kernel(kernel)
                )
                .is_err()
            );
        }
    }
    let mut call = transport_f64_prepare::<8, _>(&backend).unwrap();
    drop(backend);
    let input = [
        0,
        0x8000_0000_0000_0000,
        1,
        0x8000_0000_0000_0001,
        0x7ff0_0000_0000_0000,
        0xfff0_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7ff0_0000_0000_0001,
    ]
    .map(f64::from_bits);
    let mut output = [23.0_f64; 11];
    call(&input, &mut output).unwrap();
    assert_eq!(
        output[..8]
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        input.map(f64::to_bits)
    );
    assert_eq!(&output[8..], &[23.0; 3]);
}
