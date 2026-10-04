//! Full encoding proof and transactional prepared low-format unary execution.
#[path = "graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuCheckedUnary,PcuCpuHostBackend,PcuCpuHostError};
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuBindingRef,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuDispatchFloatUnaryOp,PcuExecutionFault,PcuExecutionFaultKind,PcuFloatUnderflowPolicy,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuRangePolicy};
use oracle::Low;
const OPS: [PcuDispatchFloatUnaryOp; 2] =
    [PcuDispatchFloatUnaryOp::Neg, PcuDispatchFloatUnaryOp::Relu];
const POLICIES: [PcuFloatUnderflowPolicy; 3] = [
    PcuFloatUnderflowPolicy::IeeeAfterRounding,
    PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    PcuFloatUnderflowPolicy::RejectSubnormalResult,
];
fn call<T: Low>(
    prepared: &mut impl PcuPreparedHostKernel<Error = PcuCpuHostError>,
    input: &[T],
    output: &mut [T],
) -> Result<(), PcuCpuHostError> {
    prepared.call(&mut [
        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
    ])
}
fn exhaustive<T: Low>() {
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                let mut prepared = graph::Graph::new(T::TYPE, 1, op, policy, range)
                    .with(|kernel| PcuCpuCheckedUnary::<T>::new().prepare_host_kernel(kernel))
                    .unwrap();
                for encoding in 0..u32::from(T::SIGN) * 2 {
                    let bits = u16::try_from(encoding).unwrap();
                    let input = [T::from_bits(bits)];
                    let mut output = [T::from_bits(17); 3];
                    let mut expected = output;
                    let result = oracle::native::<T, 1>(&input, &mut expected, op, policy, range);
                    assert_eq!(
                        call(&mut prepared, &input, &mut output)
                            .map_err(|error| error.fault().unwrap()),
                        result,
                        "bits{bits:04x} {op:?}/{policy:?}/{range:?}"
                    );
                    assert_eq!(output, expected);
                }
            }
        }
    }
}
#[test]
fn f16_complete_encoding_unary_matrix() {
    exhaustive::<PcuF16Bits>();
}
#[test]
fn bf16_complete_encoding_unary_matrix() {
    exhaustive::<PcuBf16Bits>();
}
#[test]
fn e4m3fn_complete_encoding_unary_matrix() {
    exhaustive::<PcuF8E4M3FnBits>();
}
#[test]
fn e5m2_complete_encoding_unary_matrix() {
    exhaustive::<PcuF8E5M2Bits>();
}
fn layouts<T: Low>() {
    for op in OPS {
        for policy in POLICIES {
            for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                for grid in [false, true] {
                    let mut graph = graph::Graph::new(T::TYPE, 7, op, policy, range);
                    graph.grid = grid;
                    let mut prepared = graph
                        .with(|kernel| PcuCpuHostBackend::scalar().prepare_host_kernel(kernel))
                        .unwrap();
                    let input = [
                        T::from_bits(0),
                        T::from_bits(T::SIGN),
                        T::from_bits(1),
                        T::from_bits(T::SIGN | 1),
                        T::from_bits(T::MAX),
                        T::from_bits(T::SIGN | T::MAX),
                        T::from_bits(1 << T::FRACTION),
                    ];
                    let mut output = [T::from_bits(17); 9];
                    let mut expected = output;
                    let result = oracle::native::<T, 7>(&input, &mut expected, op, policy, range);
                    assert_eq!(
                        call(&mut prepared, &input, &mut output)
                            .map_err(|error| error.fault().unwrap()),
                        result
                    );
                    assert_eq!(output, expected);
                    for lane in [0, 4, 6] {
                        let mut bad = input;
                        bad[lane] = T::from_bits(T::SIGN - 1);
                        let before = output;
                        let result = call(&mut prepared, &bad, &mut output)
                            .unwrap_err()
                            .fault()
                            .unwrap();
                        let expected_fault = if range == PcuRangePolicy::Reject
                            && policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                            && lane > 2
                        {
                            PcuExecutionFault {
                                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                                recovered: false,
                                invocation_id: 2,
                            }
                        } else {
                            PcuExecutionFault {
                                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                                recovered: false,
                                invocation_id: lane as u64,
                            }
                        };
                        assert_eq!(result, expected_fault);
                        assert_eq!(output, before);
                        assert_eq!(
                            call(&mut prepared, &input, &mut output)
                                .map_err(|error| error.fault().unwrap()),
                            oracle::native::<T, 7>(&input, &mut expected, op, policy, range)
                        );
                        assert_eq!(output, expected);
                    }
                    let before = output;
                    assert!(call(&mut prepared, &input[..6], &mut output).is_err());
                    assert_eq!(output, before);
                    assert!(
                        prepared
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), &[0_u32; 7]),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output)
                            ])
                            .is_err()
                    );
                    assert_eq!(output, before);
                    graph.broadcast = true;
                    let mut broadcast = graph
                        .with(|kernel| PcuCpuHostBackend::scalar().prepare_host_kernel(kernel))
                        .unwrap();
                    let expected_result =
                        oracle::native::<T, 7>(&[input[2]; 7], &mut expected, op, policy, range);
                    assert_eq!(
                        call(&mut broadcast, &input[2..3], &mut output)
                            .map_err(|error| error.fault().unwrap()),
                        expected_result
                    );
                    assert_eq!(output, expected);
                }
            }
        }
    }
}
#[test]
fn f16_layouts_rollback_retry_tails() {
    layouts::<PcuF16Bits>();
}
#[test]
fn bf16_layouts_rollback_retry_tails() {
    layouts::<PcuBf16Bits>();
}
#[test]
fn e4m3fn_layouts_rollback_retry_tails() {
    layouts::<PcuF8E4M3FnBits>();
}
#[test]
fn e5m2_layouts_rollback_retry_tails() {
    layouts::<PcuF8E5M2Bits>();
}
#[test]
#[allow(clippy::too_many_lines)] // Full exact cold IDs/policies/permissions and negative requests remain adjacent.
fn exact_cold_gates_and_offer_ids() {
    #[rustfmt::skip]
 use pcu_facade::{PcuDeviceIdentity,PcuObjectRef,PcuObjectKind,PcuProviderId,PcuExecutorId,PcuCostBoundary,PcuImplementationOffers,PcuImplementationRequest,PcuNumericalMode,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuReproducibility,PcuDispatchDataOp,PcuDispatchOp};
    use fusion_pcu_cpu::PcuCpuHostOffers;
    let device = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
        provider: PcuProviderId(3),
        generation: 7,
        kind: PcuObjectKind::Device,
        id: 0,
    })
    .unwrap();
    let offers = PcuCpuHostOffers::new(PcuCpuHostBackend::scalar(), device, PcuExecutorId(0));
    for (scalar, base) in [
        (PcuF16Bits::TYPE, 288),
        (PcuBf16Bits::TYPE, 292),
        (PcuF8E4M3FnBits::TYPE, 296),
        (PcuF8E5M2Bits::TYPE, 300),
    ] {
        for (offset, op) in OPS.into_iter().enumerate() {
            for policy in POLICIES {
                for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
                    graph::Graph::new(scalar, 7, op, policy, range).with(|kernel| {
                        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
                            for permitted in [false, true] {
                                let mut kernel = *kernel;
                                kernel.numerical_requirements.numerical_mode = mode;
                                if permitted {
                                    kernel
                                        .numerical_requirements
                                        .numerical_options
                                        .compound_arithmetic =
                                        PcuCompoundArithmeticPolicy::BackendDefined;
                                    kernel.numerical_requirements.numerical_options.precision =
                                        PcuPrecisionPolicy::BackendOptimized;
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
                                assert_eq!(
                                    (offer.implementation.local_id, offer.implementation.revision),
                                    (
                                        base + u32::try_from(offset).unwrap()
                                            + if range == PcuRangePolicy::Clamp { 2 } else { 0 },
                                        1
                                    )
                                );
                                offer.validate_request(&request).unwrap();
                                assert_eq!(offer.workspace_bytes, Some(0));
                                let mismatch = PcuImplementationRequest {
                                    requirements: pcu_facade::PcuImplementationRequirements {
                                        numerical_mode: if mode == PcuNumericalMode::Strict {
                                            PcuNumericalMode::Boundary
                                        } else {
                                            PcuNumericalMode::Strict
                                        },
                                        ..request.requirements
                                    },
                                    ..request
                                };
                                assert_eq!(offers.implementation_offers(&mismatch, &mut []), Ok(0));
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
                        let mut range_kernel = *kernel;
                        range_kernel.numerical_requirements.range_policy =
                            if range == PcuRangePolicy::Clamp {
                                PcuRangePolicy::Reject
                            } else {
                                PcuRangePolicy::Clamp
                            };
                        // A differing operation-local policy uses the composed profile;
                        // the canonical unary profile still requires matching headers.
                        assert!(matches!(
                            PcuCpuHostBackend::scalar().prepare_host_kernel(&range_kernel),
                            Ok(fusion_pcu_cpu::PcuCpuPreparedHost::Composed(_))
                        ));
                        let mut kernel = *kernel;
                        kernel.numerical_requirements.float_underflow =
                            if policy == PcuFloatUnderflowPolicy::IeeeAfterRounding {
                                PcuFloatUnderflowPolicy::AllowGradualUnderflow
                            } else {
                                PcuFloatUnderflowPolicy::IeeeAfterRounding
                            };
                        assert!(matches!(
                            PcuCpuHostBackend::scalar().prepare_host_kernel(&kernel),
                            Ok(fusion_pcu_cpu::PcuCpuPreparedHost::Composed(_))
                        ));
                        let mut ops = kernel.ops.to_vec();
                        if let PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
                            value,
                            ..
                        }) = &mut ops[1]
                        {
                            *value = pcu_facade::PcuDispatchValueId(99);
                        }
                        kernel.ops = &ops;
                        assert!(
                            PcuCpuHostBackend::scalar()
                                .prepare_host_kernel(&kernel)
                                .is_err()
                        );
                    });
                }
            }
        }
    }
    for ty in [
        pcu_facade::PcuScalarType::F128,
        pcu_facade::PcuScalarType::F256,
    ] {
        graph::Graph::new(
            ty,
            7,
            PcuDispatchFloatUnaryOp::Relu,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuRangePolicy::Reject,
        )
        .with(|kernel| {
            assert!(
                PcuCpuHostBackend::scalar()
                    .prepare_host_kernel(kernel)
                    .is_err()
            );
        });
    }
}

#[cfg(feature = "source-unary")]
#[path = "source/source.rs"]
mod source;
#[cfg(feature = "source-unary")]
#[allow(clippy::too_many_lines)] // Genuine prepared and ordinary source share one exact policy/fault/retry matrix.
#[allow(clippy::cognitive_complexity)] // Complete format/policy/publication matrix preserves one owner lifecycle and exact case ordering.
fn ordinary<T: Low>() {
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
            for bad_lane in [None, Some(0), Some(5), Some(6)] {
                let mut input = input;
                if let Some(lane) = bad_lane {
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
                assert_eq!(output, expected);
                output = before;
                let ordinary =
                    source::$entry::<T, 7>(&input, &mut output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected ordinary error {other:?}"),
                    });
                assert_eq!(ordinary, result);
                assert_eq!(output, expected);
            }
            let before = output;
            assert!(source::$entry::<T, 7>(&input[..6], &mut output).is_err());
            assert_eq!(output, before);
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
    assert_eq!(output, expected);
    assert!(
        matches!(source::broadcast_relu::<T,7>(&T::from_bits(1),&mut output),Err(global::PcuExecutionError::ArithmeticFault(fault)) if fault.recovered&&fault.invocation_id==0)
    );
    assert_eq!(output[..7], [T::from_bits(1); 7]);
}
#[cfg(feature = "source-unary")]
#[test]
fn genuine_ordinary_and_prepared_source_all_formats_policies_and_ranges() {
    use pcu_facade::global;
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    ordinary::<PcuF16Bits>();
    ordinary::<PcuBf16Bits>();
    ordinary::<PcuF8E4M3FnBits>();
    ordinary::<PcuF8E5M2Bits>();
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
