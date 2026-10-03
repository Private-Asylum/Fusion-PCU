//! Actual six-format unary oracle, detached ownership, exact policies and source publication.
#[path = "../../benches/checked_unary/ffi/ffi.rs"]
#[allow(dead_code)]
mod ffi;
#[path = "../../../cpu/tests/low_unary/graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)]
mod source;
use oracle::{Float, same};
#[rustfmt::skip]
use pcu_facade::{global,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDispatchFloatUnaryOp,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuExecutionFault,PcuExecutionFaultKind};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanDiscovery,PcuVulkanError};
#[rustfmt::skip]
use pcu_facade::{PcuDeviceClass,PcuDeviceDescriptor,PcuObjectKind,PcuObjectRef,PcuProviderDescriptor,PcuProviderId,PcuProviderReadiness,PcuProviderStatus,PcuRuntimeDiscovery,PcuStableDeviceIdentity,PcuTargetDescriptor};
fn selected() -> (PcuVulkanBackend, PcuStableDeviceIdentity) {
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
    (
        PcuVulkanBackend::open(&discovery, selected).unwrap(),
        discovery
            .device_facts(selected)
            .unwrap()
            .stable_identity
            .unwrap(),
    )
}

const OPS: [PcuDispatchFloatUnaryOp; 2] =
    [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
];
fn call<T: Float>(
    plan: &mut impl PcuPreparedHostKernel<Error = PcuVulkanError>,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuVulkanError> {
    plan.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
    ])
}
fn fault(error: PcuVulkanError) -> PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected {other:?}"),
    }
}
fn encoding<T: Float, const PORTABLE: bool>(identity: &PcuStableDeviceIdentity) -> usize {
    let count = if T::BITS <= 16 {
        1usize << T::BITS
    } else {
        131_072
    };
    let mut state = 0x243f_6a88_85a3_08d3u64;
    let edges = [
        0,
        T::SIGN,
        1,
        T::SIGN | 1,
        (1 << T::FRACTION) - 1,
        1 << T::FRACTION,
        T::MAX,
        T::SIGN | T::MAX,
        T::MAX + 1,
        T::SIGN | (T::MAX + 1),
        T::SIGN - 1,
    ];
    let input: Vec<T> = (0..count)
        .map(|i| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            T::from_raw(if T::BITS <= 16 {
                u64::try_from(i).unwrap()
            } else if i < edges.len() {
                edges[i]
            } else if T::BITS == 32 {
                state & u64::from(u32::MAX)
            } else {
                state
            })
        })
        .collect();
    let mut output = vec![T::from_raw(17); count];
    let mut statuses = vec![0u32; count];
    let mut comparisons = 0;
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let g =
                    graph::Graph::new(T::TYPE, u32::try_from(count).unwrap(), op, policy, range);
                let words = g.with(|kernel| {
                    let mut kernel = *kernel;
                    if PORTABLE {
                        kernel
                            .numerical_requirements
                            .numerical_options
                            .reproducibility = pcu_facade::PcuReproducibility::PortableV1;
                    }
                    let kernel = &kernel;
                    let mut words = Vec::new();
                    fusion_pcu_spirv::lower_checked_float_unary_to_spirv(
                        kernel,
                        fusion_pcu_spirv::PcuSpirvLoweringOptions::default(),
                        &mut words,
                    )
                    .unwrap();
                    words
                });
                let mut native = ffi::NativeUnary::from_words(
                    *identity,
                    u32::try_from(count).unwrap(),
                    T::FORMAT,
                    &words,
                )
                .unwrap();
                native
                    .diagnostics(
                        ffi::bytes(&input),
                        ffi::bytes_mut(&mut output),
                        &mut statuses,
                    )
                    .unwrap();
                for lane in 0..count {
                    let (bits, code) = match oracle::evaluate::<T>(input[lane].raw(), op, policy) {
                        Ok((bits, false)) => (bits, 0),
                        Ok((bits, true)) => (bits, 2),
                        Err(PcuExecutionFaultKind::InvalidFloatingOperand) => (0, 1),
                        Err(other) => panic!("unary oracle unexpected {other:?}"),
                    };
                    assert_eq!(
                        (output[lane].raw(), statuses[lane]),
                        (bits, code),
                        "{:?} {op:?} {policy:?} {range:?} input{:#x}",
                        T::TYPE,
                        input[lane].raw()
                    );
                    comparisons += 1;
                }
            }
        }
    }
    comparisons
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn complete_low_encoding_and_native_full_bit_private_oracle() {
    let (_, identity) = selected();
    let count = encoding::<PcuF16Bits, false>(&identity)
        + encoding::<PcuBf16Bits, false>(&identity)
        + encoding::<PcuF8E4M3FnBits, false>(&identity)
        + encoding::<PcuF8E5M2Bits, false>(&identity)
        + encoding::<f32, false>(&identity)
        + encoding::<f64, false>(&identity);
    assert_eq!(count, 4_724_736);
}
#[allow(clippy::too_many_lines)] // Every policy/layout/fault ordering shares one independent bit oracle and lifecycle.
fn publication<T: Float, const PORTABLE: bool>(backend: &PcuVulkanBackend) {
    let sentinel = T::from_raw(17);
    let one = T::from_raw(1 << T::FRACTION);
    let tiny = T::from_raw(1);
    let nan = T::from_raw(T::MAX + 1);
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        let mut g = graph::Graph::new(T::TYPE, 7, op, policy, range);
                        g.grid = grid;
                        g.broadcast = broadcast;
                        let mut plan = g
                            .with(|kernel| {
                                let mut kernel = *kernel;
                                if PORTABLE {
                                    kernel
                                        .numerical_requirements
                                        .numerical_options
                                        .reproducibility =
                                        pcu_facade::PcuReproducibility::PortableV1;
                                }
                                backend.prepare_host_kernel(&kernel)
                            })
                            .unwrap();
                        let mut output = [sentinel; 10];
                        let finite = [
                            T::from_raw(0),
                            T::from_raw(T::SIGN),
                            tiny,
                            T::from_raw(T::SIGN | 1),
                            one,
                            T::from_raw(T::SIGN | (1 << T::FRACTION)),
                            T::from_raw(T::MAX),
                        ];
                        let input = &finite[..if broadcast { 1 } else { 7 }];
                        let result = call(&mut plan, input, &mut output);
                        let mut expected = [sentinel; 10];
                        let mut notice = None;
                        let mut fatal = None;
                        for (lane, value) in expected[..7].iter_mut().enumerate() {
                            let input = finite[if broadcast { 0 } else { lane }];
                            let (bits, underflow) =
                                oracle::evaluate::<T>(input.raw(), op, policy).unwrap();
                            *value = T::from_raw(bits);
                            if underflow {
                                let fault = PcuExecutionFault {
                                    recovered: range == PcuRangePolicy::Clamp,
                                    invocation_id: u64::try_from(lane).unwrap(),
                                    kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                                };
                                if fault.recovered {
                                    notice.get_or_insert(fault);
                                } else {
                                    fatal.get_or_insert(fault);
                                }
                            }
                        }
                        if let Some(fatal) = fatal {
                            assert_eq!(fault(result.unwrap_err()), fatal);
                            same(&output, &[sentinel; 10]);
                        } else {
                            assert_eq!(result.map_err(fault), notice.map_or(Ok(()), Err));
                            same(&output, &expected);
                        }
                        for bad_lane in [0, 6] {
                            let mut input = [tiny; 7];
                            input[bad_lane] = nan;
                            let before = output;
                            let input = &input[..if broadcast { 1 } else { 7 }];
                            let result = call(&mut plan, input, &mut output);
                            if broadcast && bad_lane == 6 {
                                if policy == PcuFloatUnderflowPolicy::RejectSubnormalResult {
                                    assert!(result.is_err());
                                } else {
                                    result.unwrap();
                                }
                            } else {
                                let error = fault(result.unwrap_err());
                                let first = if range == PcuRangePolicy::Reject
                                    && policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                                    && bad_lane != 0
                                {
                                    0
                                } else {
                                    bad_lane
                                };
                                assert_eq!(
                                    error.invocation_id,
                                    u64::try_from(if broadcast { 0 } else { first }).unwrap()
                                );
                                assert!(!error.recovered);
                                same(&output, &before);
                            }
                        }
                        let before = output;
                        assert!(call(&mut plan, &[], &mut output).is_err());
                        same(&output, &before);
                        call(
                            &mut plan,
                            &[one; 7][..if broadcast { 1 } else { 7 }],
                            &mut output,
                        )
                        .unwrap();
                        same(&output[7..], &[sentinel; 3]);
                    }
                }
            }
        }
    }
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn six_format_full_policies_grid_broadcast_fatal_precedence_tails_and_retry() {
    let backend = PcuVulkanBackend::new().unwrap();
    publication::<PcuF16Bits, false>(&backend);
    publication::<PcuBf16Bits, false>(&backend);
    publication::<PcuF8E4M3FnBits, false>(&backend);
    publication::<PcuF8E5M2Bits, false>(&backend);
    publication::<f32, false>(&backend);
    publication::<f64, false>(&backend);
}

macro_rules! genuine {
    ($backend:ident,$entry:ident,$prepare:ident,$op:ident,$policy:ident,$clamp:literal) => {{
        let mut prepared = source::$prepare::<T, 7, _>($backend).unwrap();
        let tiny = T::from_raw(1);
        let normal = T::from_raw(1 << T::FRACTION);
        let sentinel = T::from_raw(17);
        let mut input = [tiny; 7];
        let mut output = [sentinel; 10];
        for ordinary in [false, true] {
            output.fill(sentinel);
            let result = if ordinary {
                source::$entry::<T, 7>(&input, &mut output).map_err(|e| match e {
                    global::PcuExecutionError::ArithmeticFault(f) => f,
                    other => panic!("unexpected {other:?}"),
                })
            } else {
                prepared(&input, &mut output).map_err(fault)
            };
            let (bits, underflow) = oracle::evaluate::<T>(
                1,
                PcuDispatchFloatUnaryOp::$op,
                PcuFloatUnderflowPolicy::$policy,
            )
            .unwrap();
            assert_eq!(
                result,
                if underflow {
                    Err(PcuExecutionFault {
                        recovered: $clamp,
                        invocation_id: 0,
                        kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                    })
                } else {
                    Ok(())
                }
            );
            same(
                &output[..7],
                &[T::from_raw(if underflow && !$clamp { 17 } else { bits }); 7],
            );
            same(&output[7..], &[sentinel; 3]);
            input[6] = T::from_raw(T::MAX + 1);
            let before = output;
            let result = if ordinary {
                source::$entry::<T, 7>(&input, &mut output).map_err(|e| match e {
                    global::PcuExecutionError::ArithmeticFault(f) => f,
                    other => panic!("unexpected {other:?}"),
                })
            } else {
                prepared(&input, &mut output).map_err(fault)
            };
            let first = if underflow && !$clamp { 0 } else { 6 };
            assert_eq!(result.unwrap_err().invocation_id, first);
            same(&output, &before);
            input.fill(normal);
            if ordinary {
                source::$entry::<T, 7>(&input, &mut output).unwrap();
            } else {
                prepared(&input, &mut output).unwrap();
            }
            same(&output[7..], &[sentinel; 3]);
            input.fill(tiny);
        }
    }};
}
fn genuine_source<T: Float>(backend: &PcuVulkanBackend) {
    genuine!(
        backend,
        neg_reject_ieee,
        neg_reject_ieee_prepare,
        Neg,
        IeeeAfterRounding,
        false
    );
    genuine!(
        backend,
        neg_clamp_ieee,
        neg_clamp_ieee_prepare,
        Neg,
        IeeeAfterRounding,
        true
    );
    genuine!(
        backend,
        neg_reject_gradual,
        neg_reject_gradual_prepare,
        Neg,
        AllowGradualUnderflow,
        false
    );
    genuine!(
        backend,
        neg_clamp_gradual,
        neg_clamp_gradual_prepare,
        Neg,
        AllowGradualUnderflow,
        true
    );
    genuine!(
        backend,
        neg_reject_strict,
        neg_reject_strict_prepare,
        Neg,
        RejectSubnormalResult,
        false
    );
    genuine!(
        backend,
        neg_clamp_strict,
        neg_clamp_strict_prepare,
        Neg,
        RejectSubnormalResult,
        true
    );
    genuine!(
        backend,
        relu_reject_ieee,
        relu_reject_ieee_prepare,
        Relu,
        IeeeAfterRounding,
        false
    );
    genuine!(
        backend,
        relu_clamp_ieee,
        relu_clamp_ieee_prepare,
        Relu,
        IeeeAfterRounding,
        true
    );
    genuine!(
        backend,
        relu_reject_gradual,
        relu_reject_gradual_prepare,
        Relu,
        AllowGradualUnderflow,
        false
    );
    genuine!(
        backend,
        relu_clamp_gradual,
        relu_clamp_gradual_prepare,
        Relu,
        AllowGradualUnderflow,
        true
    );
    genuine!(
        backend,
        relu_reject_strict,
        relu_reject_strict_prepare,
        Relu,
        RejectSubnormalResult,
        false
    );
    genuine!(
        backend,
        relu_clamp_strict,
        relu_clamp_strict_prepare,
        Relu,
        RejectSubnormalResult,
        true
    );
}
#[test]
#[ignore = "requires explicitly available Vulkan compute device"]
fn six_format_genuine_ordinary_and_prepared_fault_publication() {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let backend = PcuVulkanBackend::new().unwrap();
    genuine_source::<PcuF16Bits>(&backend);
    genuine_source::<PcuBf16Bits>(&backend);
    genuine_source::<PcuF8E4M3FnBits>(&backend);
    genuine_source::<PcuF8E5M2Bits>(&backend);
    genuine_source::<f32>(&backend);
    genuine_source::<f64>(&backend);
}

#[test]
#[ignore = "requires explicitly available Vulkan compute device"]
#[allow(clippy::too_many_lines)] // Exact full numerical tuples and unsupported mutations share the same frozen profile.
fn exact_six_format_cold_offers_and_distinct_portable() {
    use pcu_facade::{
        PcuImplementationOffers, PcuImplementationRequest, PcuExecutorId, PcuCostBoundary,
        PcuNumericalMode, PcuCompoundArithmeticPolicy, PcuPrecisionPolicy, PcuReproducibility,
    };
    let (backend, _) = selected();
    let mut profiles = 0;
    for (format, scalar) in [
        pcu_facade::PcuScalarType::F16,
        pcu_facade::PcuScalarType::BF16,
        pcu_facade::PcuScalarType::F8E4M3FN,
        pcu_facade::PcuScalarType::F8E5M2,
        pcu_facade::PcuScalarType::F32,
        pcu_facade::PcuScalarType::F64,
    ]
    .into_iter()
    .enumerate()
    {
        for (op_code, op) in OPS.into_iter().enumerate() {
            for policy in POLICIES {
                for (range_code, range) in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp]
                    .into_iter()
                    .enumerate()
                {
                    let mut g = graph::Graph::new(scalar, 65, op, policy, range);
                    g.grid = range_code == 1;
                    g.broadcast = op_code == 1;
                    g.with(|kernel| {
                        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                            for compound in [
                                PcuCompoundArithmeticPolicy::Checked,
                                PcuCompoundArithmeticPolicy::BackendDefined,
                            ] {
                                for precision in [
                                    PcuPrecisionPolicy::Preserve,
                                    PcuPrecisionPolicy::BackendOptimized,
                                ] {
                                    let mut kernel = *kernel;
                                    kernel.numerical_requirements.numerical_mode = mode;
                                    kernel
                                        .numerical_requirements
                                        .numerical_options
                                        .compound_arithmetic = compound;
                                    kernel.numerical_requirements.numerical_options.precision =
                                        precision;
                                    let request = PcuImplementationRequest {
                                        device: backend.device_identity().unwrap(),
                                        executor: PcuExecutorId(0),
                                        boundary: PcuCostBoundary::Host,
                                        operation: &kernel,
                                        requirements: kernel.numerical_requirements,
                                    };
                                    assert_eq!(
                                        backend.implementation_offers(&request, &mut []).unwrap(),
                                        1
                                    );
                                    let mut offers = [None];
                                    backend
                                        .implementation_offers(&request, &mut offers)
                                        .unwrap();
                                    let offer = offers[0].unwrap();
                                    assert_eq!(
                                        offer.implementation.local_id,
                                        96 + u32::try_from(format * 4 + range_code * 2 + op_code)
                                            .unwrap()
                                    );
                                    assert_eq!(offer.implementation.revision, 1);
                                    offer.validate_request(&request).unwrap();
                                    profiles += 1;
                                    let mut bad = kernel;
                                    bad.numerical_requirements.numerical_options.reproducibility =
                                        PcuReproducibility::PortableV1;
                                    let unsupported = PcuImplementationRequest {
                                        operation: &bad,
                                        requirements: bad.numerical_requirements,
                                        ..request
                                    };
                                    let mut portable_offers = [None];
                                    assert_eq!(
                                        backend
                                            .implementation_offers(
                                                &unsupported,
                                                &mut portable_offers
                                            )
                                            .unwrap(),
                                        1
                                    );
                                    let portable_offer = portable_offers[0].unwrap();
                                    assert_eq!(
                                        portable_offer.implementation.local_id,
                                        16640
                                            + u32::try_from(format * 4 + range_code * 2 + op_code)
                                                .unwrap()
                                    );
                                    assert_eq!(portable_offer.implementation.revision, 1);
                                    assert_eq!(
                                        portable_offer.requirements,
                                        unsupported.requirements
                                    );
                                    portable_offer.validate_request(&unsupported).unwrap();
                                    assert_eq!(
                                        backend
                                            .implementation_offers(
                                                &PcuImplementationRequest {
                                                    operation: &bad,
                                                    ..request
                                                },
                                                &mut []
                                            )
                                            .unwrap(),
                                        0
                                    );
                                    assert!(backend.prepare_host_kernel(&bad).is_ok());
                                    let mismatch = PcuImplementationRequest {
                                        requirements: pcu_facade::PcuImplementationRequirements {
                                            range_policy: if range == PcuRangePolicy::Clamp {
                                                PcuRangePolicy::Reject
                                            } else {
                                                PcuRangePolicy::Clamp
                                            },
                                            ..request.requirements
                                        },
                                        ..request
                                    };
                                    assert_eq!(
                                        backend.implementation_offers(&mismatch, &mut []).unwrap(),
                                        0
                                    );
                                }
                            }
                        }
                    });
                }
            }
        }
    }
    assert_eq!(profiles, 576);
}

#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn portable_six_format_complete_encodings_and_private_native_oracle() {
    let (_, identity) = selected();
    let count = encoding::<PcuF16Bits, true>(&identity)
        + encoding::<PcuBf16Bits, true>(&identity)
        + encoding::<PcuF8E4M3FnBits, true>(&identity)
        + encoding::<PcuF8E5M2Bits, true>(&identity)
        + encoding::<f32, true>(&identity)
        + encoding::<f64, true>(&identity);
    assert_eq!(count, 4_724_736);
}
#[test]
#[ignore = "requires an explicitly available Vulkan compute device"]
fn portable_six_format_publication_layouts_and_fatal_priority() {
    let backend = PcuVulkanBackend::new().unwrap();
    publication::<PcuF16Bits, true>(&backend);
    publication::<PcuBf16Bits, true>(&backend);
    publication::<PcuF8E4M3FnBits, true>(&backend);
    publication::<PcuF8E5M2Bits, true>(&backend);
    publication::<f32, true>(&backend);
    publication::<f64, true>(&backend);
}
