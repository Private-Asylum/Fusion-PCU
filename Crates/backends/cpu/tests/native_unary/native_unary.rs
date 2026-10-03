//! Exact native-format unary payloads, cold offers, host publication and genuine source parity.
#[path = "../low_unary/graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "source/source.rs"]
mod source;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuHostBackend,PcuCpuCheckedUnary,PcuCpuHostError,PcuCpuPreparedHost};
#[rustfmt::skip]
use pcu_facade::{PcuBindingRef,PcuDispatchFloatUnaryOp,PcuExecutionFault,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuRangePolicy,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
use oracle::Native;
const OPS: [PcuDispatchFloatUnaryOp; 2] =
    [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
];
fn call<T: Native>(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuCpuHostError>,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuCpuHostError> {
    prepared.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
    ])
}
fn encoded<T: Native>() {
    let mask = T::SIGN | (T::SIGN - 1);
    let min_normal = 1 << T::FRACTION;
    let edges = [
        0,
        T::SIGN,
        1,
        T::SIGN | 1,
        min_normal - 1,
        T::SIGN | (min_normal - 1),
        min_normal,
        T::SIGN | min_normal,
        T::MAX,
        T::SIGN | T::MAX,
        T::MAX + 1,
        T::SIGN | (T::MAX + 1),
        T::SIGN - 1,
        mask,
    ];
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let mut prepared = graph::Graph::new(T::TYPE, 1, op, policy, range)
                    .with(|kernel| PcuCpuHostBackend::scalar().prepare_host_kernel(kernel))
                    .unwrap();
                let mut state = 0x718f_ddaa_144e_919b_u64;
                for encoding in edges.into_iter().chain((0..65_536).map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    state & mask
                })) {
                    let input = [T::from_bits(encoding)];
                    let mut output = [T::from_bits(17); 3];
                    let mut expected = output;
                    let result = oracle::native::<T, 1>(&input, &mut expected, op, policy, range);
                    assert_eq!(
                        call(&mut prepared, &input, &mut output)
                            .map_err(|error| error.fault().unwrap()),
                        result
                    );
                    assert_eq!(
                        oracle::bits(&output),
                        oracle::bits(&expected),
                        "{encoding:016x} {op:?}/{policy:?}/{range:?}"
                    );
                }
            }
        }
    }
}
#[test]
fn binary32_raw_encoding_edges_and_random() {
    encoded::<f32>();
}
#[test]
fn binary64_raw_encoding_edges_and_random() {
    encoded::<f64>();
}
fn layouts<T: Native>() {
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    for broadcast in [false, true] {
                        let mut graph = graph::Graph::new(T::TYPE, 7, op, policy, range);
                        graph.grid = grid;
                        graph.broadcast = broadcast;
                        let mut prepared = graph
                            .with(|kernel| PcuCpuHostBackend::scalar().prepare_host_kernel(kernel))
                            .unwrap();
                        let input = if broadcast {
                            [T::from_bits(1); 7]
                        } else {
                            [
                                T::from_bits(0),
                                T::from_bits(T::SIGN),
                                T::from_bits(1),
                                T::from_bits(T::SIGN | 1),
                                T::from_bits(T::MAX),
                                T::from_bits(T::SIGN | T::MAX),
                                T::from_bits(1 << T::FRACTION),
                            ]
                        };
                        let mut output = [T::from_bits(17); 9];
                        let mut expected = output;
                        let result =
                            oracle::native::<T, 7>(&input, &mut expected, op, policy, range);
                        assert_eq!(
                            call(
                                &mut prepared,
                                &input[..if broadcast { 1 } else { 7 }],
                                &mut output
                            )
                            .map_err(|error| error.fault().unwrap()),
                            result
                        );
                        assert_eq!(oracle::bits(&output), oracle::bits(&expected));
                        for lane in if broadcast { &[0][..] } else { &[0, 4, 6][..] } {
                            let mut bad = input;
                            bad[*lane] = T::from_bits(T::SIGN - 1);
                            let before = oracle::bits(&output);
                            let result = call(
                                &mut prepared,
                                &bad[..if broadcast { 1 } else { 7 }],
                                &mut output,
                            )
                            .unwrap_err()
                            .fault()
                            .unwrap();
                            let first = if range == PcuRangePolicy::Reject
                                && policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                                && *lane > 2
                            {
                                PcuExecutionFault {
                                    recovered: false,
                                    kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                                    invocation_id: 2,
                                }
                            } else {
                                PcuExecutionFault {
                                    recovered: false,
                                    kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                                    invocation_id: *lane as u64,
                                }
                            };
                            assert_eq!(result, first);
                            assert_eq!(oracle::bits(&output), before);
                            assert_eq!(
                                call(
                                    &mut prepared,
                                    &input[..if broadcast { 1 } else { 7 }],
                                    &mut output
                                )
                                .map_err(|error| error.fault().unwrap()),
                                oracle::native::<T, 7>(&input, &mut expected, op, policy, range)
                            );
                            assert_eq!(oracle::bits(&output), oracle::bits(&expected));
                        }
                        let before = oracle::bits(&output);
                        assert!(
                            call(
                                &mut prepared,
                                &input[..if broadcast { 0 } else { 6 }],
                                &mut output
                            )
                            .is_err()
                        );
                        assert_eq!(oracle::bits(&output), before);
                    }
                }
            }
        }
    }
}
#[test]
fn binary32_layouts_transaction_fatal_priority_retry_tails() {
    layouts::<f32>();
}
#[test]
fn binary64_layouts_transaction_fatal_priority_retry_tails() {
    layouts::<f64>();
}
#[test]
#[allow(clippy::too_many_lines)] // Exact old/new ownership and offer IDs remain adjacent to numerical negatives.
fn native_ids_preserve_simd_neg_and_close_all_scalar_unary_forms() {
    #[rustfmt::skip]
 use pcu_facade::{PcuDeviceIdentity,PcuObjectRef,PcuObjectKind,PcuProviderId,PcuExecutorId,PcuCostBoundary,PcuImplementationOffers,PcuImplementationRequest,PcuNumericalMode,PcuPrecisionPolicy,PcuCompoundArithmeticPolicy,PcuReproducibility,PcuScalarType};
    use fusion_pcu_cpu::PcuCpuHostOffers;
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), device, PcuExecutorId(0));
    for (scalar, base, old_id) in [(PcuScalarType::F32, 304, 0), (PcuScalarType::F64, 308, 28)] {
        for (offset, op) in OPS.into_iter().enumerate() {
            for policy in POLICIES {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    for broadcast in [false, true] {
                        let mut graph = graph::Graph::new(scalar, 7, op, policy, range);
                        graph.broadcast = broadcast;
                        graph.with(|kernel| {
                            for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                                for permission in [false, true] {
                                    let mut kernel = *kernel;
                                    kernel.numerical_requirements.numerical_mode = mode;
                                    if permission {
                                        kernel.numerical_requirements.numerical_options.precision =
                                            PcuPrecisionPolicy::BackendOptimized;
                                        kernel
                                            .numerical_requirements
                                            .numerical_options
                                            .compound_arithmetic =
                                            PcuCompoundArithmeticPolicy::BackendDefined;
                                    }
                                    let request = PcuImplementationRequest {
                                        device,
                                        executor: PcuExecutorId(0),
                                        boundary: PcuCostBoundary::Host,
                                        operation: &kernel,
                                        requirements: kernel.numerical_requirements,
                                    };
                                    let mut output = [None];
                                    assert_eq!(
                                        offers.implementation_offers(&request, &mut output),
                                        Ok(1)
                                    );
                                    let offer = output[0].unwrap();
                                    let optimized = op == PcuDispatchFloatUnaryOp::Neg
                                        && range == PcuRangePolicy::Reject
                                        && !broadcast;
                                    assert_eq!(
                                        (
                                            offer.implementation.local_id,
                                            offer.implementation.revision
                                        ),
                                        (
                                            if optimized {
                                                old_id
                                            } else {
                                                base + u32::try_from(offset).unwrap()
                                                    + if range == PcuRangePolicy::Clamp {
                                                        2
                                                    } else {
                                                        0
                                                    }
                                            },
                                            1
                                        )
                                    );
                                    offer.validate_request(&request).unwrap();
                                    let prepared = PcuCpuHostBackend::scalar()
                                        .prepare_host_kernel(&kernel)
                                        .unwrap();
                                    assert_eq!(
                                        matches!(prepared, PcuCpuPreparedHost::Neg(_)),
                                        optimized
                                    );
                                    let explicit = match scalar {
                                        PcuScalarType::F32 => PcuCpuCheckedUnary::<f32>::new()
                                            .prepare_host_kernel(&kernel),
                                        _ => PcuCpuCheckedUnary::<f64>::new()
                                            .prepare_host_kernel(&kernel),
                                    };
                                    assert_eq!(explicit.is_err(), optimized);
                                    kernel
                                        .numerical_requirements
                                        .numerical_options
                                        .reproducibility = PcuReproducibility::PortableV1;
                                    assert!(
                                        PcuCpuHostBackend::scalar()
                                            .prepare_host_kernel(&kernel)
                                            .is_ok()
                                    );
                                }
                            }
                            for change_range in [false, true] {
                                let mut kernel = *kernel;
                                if change_range {
                                    kernel.numerical_requirements.range_policy =
                                        if range == PcuRangePolicy::Clamp {
                                            PcuRangePolicy::Reject
                                        } else {
                                            PcuRangePolicy::Clamp
                                        };
                                } else {
                                    kernel.numerical_requirements.float_underflow =
                                        if policy == PcuFloatUnderflowPolicy::IeeeAfterRounding {
                                            PcuFloatUnderflowPolicy::AllowGradualUnderflow
                                        } else {
                                            PcuFloatUnderflowPolicy::IeeeAfterRounding
                                        };
                                }
                                assert!(
                                    PcuCpuHostBackend::scalar()
                                        .prepare_host_kernel(&kernel)
                                        .is_err()
                                );
                            }
                        });
                    }
                }
            }
        }
    }
}
#[allow(clippy::too_many_lines)] // Twelve genuine scalar unary policies share one exact caller publication matrix.
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn ordinary<T: Native>() {
    use pcu_facade::global;
    let input = [
        T::from_bits(T::SIGN),
        T::from_bits(T::MAX),
        T::from_bits(1),
        T::from_bits(T::SIGN | 1),
        T::from_bits(0),
        T::from_bits(T::SIGN | T::MAX),
        T::from_bits(1 << T::FRACTION),
    ];
    let mut output = [T::from_bits(17); 9];
    macro_rules! entry {
        ($entry:ident,$prepare:ident,$op:ident,$policy:ident,$range:ident) => {{
            let mut prepared = source::$prepare::<T, 7, _>(&PcuCpuHostBackend::scalar()).unwrap();
            for lane in [None, Some(0), Some(5), Some(6)] {
                let mut input = input;
                if let Some(lane) = lane {
                    input[lane] = T::from_bits(T::SIGN - 1);
                }
                let before = output;
                let mut expected = output;
                let result = oracle::native::<T, 7>(
                    &input,
                    &mut expected,
                    PcuDispatchFloatUnaryOp::$op,
                    PcuFloatUnderflowPolicy::$policy,
                    PcuRangePolicy::$range,
                );
                assert_eq!(
                    prepared(&input, &mut output).map_err(|error| error.fault().unwrap()),
                    result
                );
                assert_eq!(oracle::bits(&output), oracle::bits(&expected));
                output = before;
                assert_eq!(
                    source::$entry::<T, 7>(&input, &mut output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected source error {other:?}"),
                    }),
                    result
                );
                assert_eq!(oracle::bits(&output), oracle::bits(&expected));
            }
        }};
    }
    entry!(
        neg_reject_ieee,
        neg_reject_ieee_prepare,
        Neg,
        IeeeAfterRounding,
        Reject
    );
    entry!(
        neg_reject_gradual,
        neg_reject_gradual_prepare,
        Neg,
        AllowGradualUnderflow,
        Reject
    );
    entry!(
        neg_reject_strict,
        neg_reject_strict_prepare,
        Neg,
        RejectSubnormalResult,
        Reject
    );
    entry!(
        neg_clamp_ieee,
        neg_clamp_ieee_prepare,
        Neg,
        IeeeAfterRounding,
        Clamp
    );
    entry!(
        neg_clamp_gradual,
        neg_clamp_gradual_prepare,
        Neg,
        AllowGradualUnderflow,
        Clamp
    );
    entry!(
        neg_clamp_strict,
        neg_clamp_strict_prepare,
        Neg,
        RejectSubnormalResult,
        Clamp
    );
    entry!(
        relu_reject_ieee,
        relu_reject_ieee_prepare,
        Relu,
        IeeeAfterRounding,
        Reject
    );
    entry!(
        relu_reject_gradual,
        relu_reject_gradual_prepare,
        Relu,
        AllowGradualUnderflow,
        Reject
    );
    entry!(
        relu_reject_strict,
        relu_reject_strict_prepare,
        Relu,
        RejectSubnormalResult,
        Reject
    );
    entry!(
        relu_clamp_ieee,
        relu_clamp_ieee_prepare,
        Relu,
        IeeeAfterRounding,
        Clamp
    );
    entry!(
        relu_clamp_gradual,
        relu_clamp_gradual_prepare,
        Relu,
        AllowGradualUnderflow,
        Clamp
    );
    entry!(
        relu_clamp_strict,
        relu_clamp_strict_prepare,
        Relu,
        RejectSubnormalResult,
        Clamp
    );
    let mut expected = output;
    let result = oracle::native::<T, 7>(
        &input,
        &mut expected,
        PcuDispatchFloatUnaryOp::Neg,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuRangePolicy::Clamp,
    );
    assert!(
        matches!(source::grid_neg::<T,7>(&input,&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if Some(fault)==result.err())
    );
    assert_eq!(oracle::bits(&output), oracle::bits(&expected));
    assert!(
        matches!(source::broadcast_relu::<T,7>(&T::from_bits(1),&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered)
    );
    assert_eq!(oracle::bits(&output[..7]), vec![1; 7]);
}
#[test]
fn genuine_binary32_binary64_source_transaction_and_payloads() {
    use pcu_facade::global;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    ordinary::<f32>();
    ordinary::<f64>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

#[allow(clippy::too_many_lines)] // Twelve genuine broadcast policies retain matched source/oracle publication checks.
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn broadcasts<T: Native>() {
    use pcu_facade::global;
    let mut output = [T::from_bits(17); 9];
    macro_rules! entry {
        ($entry:ident,$prepare:ident,$op:ident,$policy:ident,$range:ident) => {{
            let mut prepared = source::$prepare::<T, 7, _>(&PcuCpuHostBackend::detect()).unwrap();
            for bits in [
                0,
                T::SIGN,
                1,
                T::SIGN | 1,
                1 << T::FRACTION,
                T::MAX,
                T::SIGN - 1,
            ] {
                let value = T::from_bits(bits);
                let mut expected = output;
                let result = oracle::native_broadcast::<T, 7>(
                    &[value],
                    &mut expected,
                    PcuDispatchFloatUnaryOp::$op,
                    PcuFloatUnderflowPolicy::$policy,
                    PcuRangePolicy::$range,
                );
                assert_eq!(
                    prepared(&value, &mut output).map_err(|error| error.fault().unwrap()),
                    result
                );
                assert_eq!(oracle::bits(&output), oracle::bits(&expected));
                assert_eq!(
                    source::$entry::<T, 7>(&value, &mut output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected broadcast source error {other:?}"),
                    }),
                    result
                );
                assert_eq!(oracle::bits(&output), oracle::bits(&expected));
            }
        }};
    }
    entry!(
        neg_broadcast_reject_ieee,
        neg_broadcast_reject_ieee_prepare,
        Neg,
        IeeeAfterRounding,
        Reject
    );
    entry!(
        neg_broadcast_reject_gradual,
        neg_broadcast_reject_gradual_prepare,
        Neg,
        AllowGradualUnderflow,
        Reject
    );
    entry!(
        neg_broadcast_reject_strict,
        neg_broadcast_reject_strict_prepare,
        Neg,
        RejectSubnormalResult,
        Reject
    );
    entry!(
        neg_broadcast_clamp_ieee,
        neg_broadcast_clamp_ieee_prepare,
        Neg,
        IeeeAfterRounding,
        Clamp
    );
    entry!(
        neg_broadcast_clamp_gradual,
        neg_broadcast_clamp_gradual_prepare,
        Neg,
        AllowGradualUnderflow,
        Clamp
    );
    entry!(
        neg_broadcast_clamp_strict,
        neg_broadcast_clamp_strict_prepare,
        Neg,
        RejectSubnormalResult,
        Clamp
    );
    entry!(
        relu_broadcast_reject_ieee,
        relu_broadcast_reject_ieee_prepare,
        Relu,
        IeeeAfterRounding,
        Reject
    );
    entry!(
        relu_broadcast_reject_gradual,
        relu_broadcast_reject_gradual_prepare,
        Relu,
        AllowGradualUnderflow,
        Reject
    );
    entry!(
        relu_broadcast_reject_strict,
        relu_broadcast_reject_strict_prepare,
        Relu,
        RejectSubnormalResult,
        Reject
    );
    entry!(
        relu_broadcast_clamp_ieee,
        relu_broadcast_clamp_ieee_prepare,
        Relu,
        IeeeAfterRounding,
        Clamp
    );
    entry!(
        relu_broadcast_clamp_gradual,
        relu_broadcast_clamp_gradual_prepare,
        Relu,
        AllowGradualUnderflow,
        Clamp
    );
    entry!(
        relu_broadcast_clamp_strict,
        relu_broadcast_clamp_strict_prepare,
        Relu,
        RejectSubnormalResult,
        Clamp
    );
}
#[test]
fn genuine_all_broadcast_policies_ownership_and_bits() {
    use pcu_facade::global;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    broadcasts::<f32>();
    broadcasts::<f64>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
