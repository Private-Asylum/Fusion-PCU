//! Actual checked F32 binary source, explicit diagnostic and ordinary invocation parity.
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuCostBoundary,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuExecutorId,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuCompoundArithmeticPolicy,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuRuntimeDiscovery,
    PcuObjectRef,
    PcuObjectKind,
    PcuProviderId,
    PcuProviderDescriptor,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanDiscovery,
};

const OPS: [PcuDispatchFloatBinaryOp; 4] = [
    PcuDispatchFloatBinaryOp::Add,
    PcuDispatchFloatBinaryOp::Sub,
    PcuDispatchFloatBinaryOp::Mul,
    PcuDispatchFloatBinaryOp::Div,
];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
];
fn oracle(
    op: PcuDispatchFloatBinaryOp,
    a: f32,
    b: f32,
    policy: PcuFloatUnderflowPolicy,
) -> Result<f32, PcuExecutionFaultKind> {
    match op {
        PcuDispatchFloatBinaryOp::Add => a.pcu_checked_add_with_policy(b, policy),
        PcuDispatchFloatBinaryOp::Sub => a.pcu_checked_sub_with_policy(b, policy),
        PcuDispatchFloatBinaryOp::Mul => a.pcu_checked_mul_with_policy(b, policy),
        PcuDispatchFloatBinaryOp::Div => a.pcu_checked_div_with_policy(b, policy),
    }
}
fn call(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuVulkanError>,
    left: &[f32],
    right: &[f32],
    output: &mut [f32],
) -> Result<(), PcuVulkanError> {
    prepared.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
    ])
}
const fn random(state: &mut u32) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 17;
    *state ^= *state << 5;
    f32::from_bits(*state)
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn actual_shader_random_bits_all_operations_policies_and_rounded_results() {
    const COUNT: usize = 8192;
    let backend = PcuVulkanBackend::new().unwrap();
    let edges = [
        0,
        0x8000_0000,
        1,
        0x8000_0001,
        0x007f_ffff,
        0x0080_0000,
        0x0080_0001,
        0x3f7f_ffff,
        0x3f80_0000,
        0x3f80_0001,
        0x7f7f_ffff,
        0xff7f_ffff,
    ];
    for op in OPS {
        for policy in POLICIES {
            let mut state = 0x243f_6a88;
            let mut left = Vec::new();
            let mut right = Vec::new();
            let mut expected = Vec::new();
            for a in edges {
                for b in edges {
                    let a = f32::from_bits(a);
                    let b = f32::from_bits(b);
                    if let Ok(value) = oracle(op, a, b, policy) {
                        left.push(a);
                        right.push(b);
                        expected.push(value.to_bits());
                    }
                }
            }
            while left.len() < COUNT {
                let a = random(&mut state);
                let b = random(&mut state);
                if let Ok(value) = oracle(op, a, b, policy) {
                    left.push(a);
                    right.push(b);
                    expected.push(value.to_bits());
                }
            }
            let mut prepared = graph::Graph::new(u32::try_from(COUNT).unwrap(), op, policy)
                .with(|kernel| backend.prepare_host_kernel(kernel))
                .unwrap();
            let mut output = vec![91.0; COUNT + 3];
            call(&mut prepared, &left, &right, &mut output).unwrap();
            for (index, (actual, expected)) in output.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    *expected,
                    "op{op:?} policy{policy:?} index{index} a{:08x} b{:08x}",
                    left[index].to_bits(),
                    right[index].to_bits()
                );
            }
            assert_eq!(&output[COUNT..], &[91.0; 3]);
        }
    }
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn actual_edge_faults_are_exact_transactional_and_retryable() {
    let backend = PcuVulkanBackend::new().unwrap();
    // Includes signed zeros, finite scans, exact/tiny-inexact subnormals and the normal boundary.
    let cases = [
        (0, 0),
        (0x8000_0000, 0),
        (0x8000_0000, 0x8000_0000),
        (1, 0x3f00_0000),
        (0x0080_0000, 0x3eff_ffff),
        (0x0080_0000, 0x3f7f_ffff),
        (0x0080_0001, 0x3f7f_ffff),
        (0x007f_ffff, 1),
        (0x7f7f_ffff, 0x4000_0000),
        (0xff7f_ffff, 0xbf80_0000),
        (0x7f80_0000, 0),
        (0x7f80_0001, 0x3f80_0000),
        (0x3f80_0000, 0x7fc0_0001),
        (0x3f80_0000, 0),
    ];
    for op in OPS {
        for policy in POLICIES {
            let mut prepared = graph::Graph::new(65, op, policy)
                .with(|kernel| backend.prepare_host_kernel(kernel))
                .unwrap();
            for (a, b) in cases {
                let mut left = [1.0; 65];
                let mut right = [1.0; 65];
                let mut output = [37.0_f32; 68];
                left[7] = f32::from_bits(a);
                right[7] = f32::from_bits(b);
                left[64] = f32::NAN;
                let expected = oracle(op, left[7], right[7], policy).err().map_or(
                    (64, PcuExecutionFaultKind::InvalidFloatingOperand),
                    |kind| (7, kind),
                );
                let Err(PcuVulkanError::Fault(fault)) =
                    call(&mut prepared, &left, &right, &mut output)
                else {
                    panic!("expected exact checked fault");
                };
                assert_eq!(
                    fault,
                    PcuExecutionFault {
                        recovered: false,
                        kind: expected.1,
                        invocation_id: expected.0
                    }
                );
                assert_eq!(output.map(f32::to_bits), [37.0_f32; 68].map(f32::to_bits));
                left[64] = 1.0;
                if let Ok(value) = oracle(op, left[7], right[7], policy) {
                    call(&mut prepared, &left, &right, &mut output).unwrap();
                    assert_eq!(output[7].to_bits(), value.to_bits());
                } else {
                    left[7] = 1.0;
                    right[7] = 1.0;
                    call(&mut prepared, &left, &right, &mut output).unwrap();
                }
                assert_eq!(&output[65..], &[37.0; 3]);
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn actual_broadcast_repeated_swapped_ssa_grid_and_complete_schema() {
    let backend = PcuVulkanBackend::new().unwrap();
    for op in OPS {
        for operands in [[1, 2], [2, 1], [1, 1], [2, 2]] {
            for broadcast in [[false, false], [true, false], [false, true], [true, true]] {
                let mut graph = graph::Graph::new(257, op, PcuFloatUnderflowPolicy::default());
                graph.grid = true;
                graph.reverse_loads = operands[0] == 2;
                graph.operands = operands;
                graph.broadcast = broadcast;
                let mut prepared = graph
                    .with(|kernel| backend.prepare_host_kernel(kernel))
                    .unwrap();
                let left = vec![
                    if operands == [2, 2] { f32::NAN } else { 6.0 };
                    if broadcast[0] { 1 } else { 257 }
                ];
                let right = vec![
                    if operands == [1, 1] { f32::NAN } else { 2.0 };
                    if broadcast[1] { 1 } else { 257 }
                ];
                let mut output = [41.0; 260];
                call(&mut prepared, &left, &right, &mut output).unwrap();
                let values = [6.0, 2.0];
                let expected = oracle(
                    op,
                    values[usize::from(operands[0] - 1)],
                    values[usize::from(operands[1] - 1)],
                    PcuFloatUnderflowPolicy::default(),
                )
                .unwrap();
                assert_eq!(output[..257], [expected; 257]);
                assert_eq!(output[257..], [41.0; 3]);
                let before = output.map(f32::to_bits);
                assert!(matches!(
                    call(&mut prepared, &[], &right, &mut output),
                    Err(PcuVulkanError::InvalidArguments)
                ));
                assert_eq!(output.map(f32::to_bits), before);
            }
        }
    }
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
#[allow(clippy::cognitive_complexity)] // Ordered fatal/recovered publication and retry assertions preserve one native owner lifecycle.
fn genuine_source_explicit_and_ordinary_calls_and_f64_admission() {
    let backend = PcuVulkanBackend::new().unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    macro_rules! check {
        ($entry:ident,$prepare:ident,$op:ident) => {{
            let mut explicit = source::f32_source::$prepare::<65, _>(&backend).unwrap();
            let left = [6.0; 65];
            let right = [2.0; 65];
            let mut output = [43.0; 68];
            explicit(&left, &right, &mut output).unwrap();
            let expected = oracle(
                PcuDispatchFloatBinaryOp::$op,
                6.0,
                2.0,
                PcuFloatUnderflowPolicy::default(),
            )
            .unwrap();
            assert_eq!(output[..65], [expected; 65]);
            assert_eq!(output[65..], [43.0; 3]);
            source::f32_source::$entry::<65>(&left, &right, &mut output).unwrap();
            assert_eq!(output[..65], [expected; 65]);
            assert!(source::f64_source::$prepare::<65, _>(&backend).is_ok());
            let mut invalid = left;
            invalid[3] = f32::NAN;
            let before = output.map(f32::to_bits);
            assert!(matches!(
                source::f32_source::$entry::<65>(&invalid, &right, &mut output),
                Err(global::PcuExecutionError::ArithmeticFault(
                    PcuExecutionFault {
                        kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                        invocation_id: 3,
                        ..
                    }
                ))
            ));
            assert_eq!(output.map(f32::to_bits), before);
            source::f32_source::$entry::<65>(&left, &right, &mut output).unwrap();
        }};
    }
    check!(add, add_prepare, Add);
    check!(sub, sub_prepare, Sub);
    check!(mul, mul_prepare, Mul);
    check!(div, div_prepare, Div);
    let mut clamp = graph::Graph::new(
        65,
        PcuDispatchFloatBinaryOp::Mul,
        PcuFloatUnderflowPolicy::default(),
    );
    clamp.range = PcuRangePolicy::Clamp;
    assert!(
        clamp
            .with(|kernel| backend.prepare_host_kernel(kernel))
            .is_ok()
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

fn discovered_backend() -> PcuVulkanBackend {
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
    PcuVulkanBackend::open(&discovery, selected).unwrap()
}

#[test]
#[ignore = "requires native Vulkan discovery and activation"]
#[allow(clippy::too_many_lines)] // Actual-device cold matrix retains exact IDs, policies and negative requests together.
fn exact_binary_offers_keep_numerical_axes_ranges_and_f64_independent() {
    let backend = discovered_backend();
    for scalar in [
        pcu_facade::PcuScalarType::F32,
        pcu_facade::PcuScalarType::F64,
        pcu_facade::PcuScalarType::F16,
        pcu_facade::PcuScalarType::BF16,
        pcu_facade::PcuScalarType::F8E4M3FN,
        pcu_facade::PcuScalarType::F8E5M2,
    ] {
        for (op_code, op) in OPS.into_iter().enumerate() {
            for policy in POLICIES {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    let mut graph = graph::Graph::new(65, op, policy);
                    graph.range = range;
                    graph.scalar = scalar;
                    graph.with(|kernel| {
                for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                    let mut kernel = *kernel;
                    kernel.numerical_requirements.numerical_mode = numerical_mode;
                    kernel.numerical_requirements.float_underflow = policy;
                    let kernel = &kernel;
                    let request = PcuImplementationRequest {
                        device: backend.device_identity().unwrap(),
                        executor: PcuExecutorId(0),
                        boundary: PcuCostBoundary::Host,
                        operation: kernel,
                        requirements: PcuImplementationRequirements {
                            numerical_mode,
                            range_policy:range,
                            float_underflow: policy,
                            ..Default::default()
                        },
                    };
                    let mut offers = [None];
                    assert_eq!(backend.implementation_offers(&request, &mut []).unwrap(), 1);
                    assert_eq!(
                        backend
                            .implementation_offers(&request, &mut offers)
                            .unwrap(),
                        1
                    );
                    let offer = offers[0].unwrap();
                    assert_eq!(
                        offer.implementation.local_id,
                        match (scalar,range) {(pcu_facade::PcuScalarType::F32,PcuRangePolicy::Reject)=>4,(pcu_facade::PcuScalarType::F32,PcuRangePolicy::Clamp)=>8,(pcu_facade::PcuScalarType::F64,PcuRangePolicy::Reject)=>12,(pcu_facade::PcuScalarType::F64,PcuRangePolicy::Clamp)=>16,(pcu_facade::PcuScalarType::F16,PcuRangePolicy::Reject)=>32,(pcu_facade::PcuScalarType::BF16,PcuRangePolicy::Reject)=>36,(pcu_facade::PcuScalarType::F8E4M3FN,PcuRangePolicy::Reject)=>40,(pcu_facade::PcuScalarType::F8E5M2,PcuRangePolicy::Reject)=>44,(pcu_facade::PcuScalarType::F16,PcuRangePolicy::Clamp)=>48,(pcu_facade::PcuScalarType::BF16,PcuRangePolicy::Clamp)=>52,(pcu_facade::PcuScalarType::F8E4M3FN,PcuRangePolicy::Clamp)=>56,(pcu_facade::PcuScalarType::F8E5M2,PcuRangePolicy::Clamp)=>60,_=>unreachable!("closed binary offers")} + u32::try_from(op_code).unwrap()
                    );
                    assert_eq!(offer.implementation.revision, 2);
                    assert_eq!(offer.workspace_bytes, None);
                    assert!(offer.cost.completion.is_none());
                    offer.validate_request(&request).unwrap();
                    for options in [
                        PcuNumericalOptions {
                            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                            ..Default::default()
                        },
                        PcuNumericalOptions {
                            precision: PcuPrecisionPolicy::BackendOptimized,
                            ..Default::default()
                        },
                        PcuNumericalOptions {
                            reproducibility: PcuReproducibility::PortableV1,
                            ..Default::default()
                        },
                    ] {
                        let different = PcuImplementationRequest {
                            requirements: PcuImplementationRequirements {
                                numerical_options: options,
                                ..request.requirements
                            },
                            ..request
                        };
                        assert_eq!(
                            backend.implementation_offers(&different, &mut []).unwrap(),
                            0
                        );
                        let matched_kernel = pcu_facade::PcuDispatchKernelIr {
                            numerical_requirements: different.requirements,
                            ..*kernel
                        };
                        let matched = PcuImplementationRequest {
                            operation: &matched_kernel,
                            ..different
                        };
                        assert_eq!(
                            backend.implementation_offers(&matched, &mut []).unwrap(),
                            usize::from(options.reproducibility != PcuReproducibility::PortableV1 || pcu_facade::describe_portable_v1_map(&matched_kernel).is_ok())
                        );
                        if options.reproducibility == PcuReproducibility::PortableV1 && pcu_facade::describe_portable_v1_map(&matched_kernel).is_err() {
                            assert!(matches!(
                                backend.prepare_host_kernel(&matched_kernel),
                                Err(PcuVulkanError::SpirvLowering {
                                    error: fusion_pcu_spirv::PcuSpirvError::UnsupportedNumericalRequirements
                                })
                            ));
                        }
                    }
                    let resident = PcuImplementationRequest {
                        boundary: PcuCostBoundary::Resident,
                        ..request
                    };
                    assert_eq!(
                        backend.implementation_offers(&resident, &mut []).unwrap(),
                        0
                    );
                    let mismatched = PcuImplementationRequest {
                        requirements: PcuImplementationRequirements {
                            float_underflow: if policy == PcuFloatUnderflowPolicy::IeeeAfterRounding
                            {
                                PcuFloatUnderflowPolicy::AllowGradualUnderflow
                            } else {
                                PcuFloatUnderflowPolicy::IeeeAfterRounding
                            },
                            ..request.requirements
                        },
                        ..request
                    };
                    assert!(matches!(
                        backend.implementation_offers(&mismatched, &mut []),
                        Ok(0)
                    ));
                }
            });
                }
                let mut unsupported = graph::Graph::new(65, op, policy);
                unsupported.scalar = pcu_facade::PcuScalarType::F128;
                unsupported.with(|kernel| {
                    let request = PcuImplementationRequest {
                        device: backend.device_identity().unwrap(),
                        executor: PcuExecutorId(0),
                        boundary: PcuCostBoundary::Host,
                        operation: kernel,
                        requirements: PcuImplementationRequirements::default(),
                    };
                    assert_eq!(backend.implementation_offers(&request, &mut []).unwrap(), 0);
                });
            }
        }
    }
}
