//! Exact policy admission, source ABI and first-lane fault decoding.
use super::*;
use fusion_pcu::PcuNumericalOptions;
fn graph(
    scalar: PcuScalarType,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(mode);
    graph.set_numerical_options(options);
    let x = graph.input([17], scalar).unwrap();
    let dy = graph.input([17], scalar).unwrap();
    let output = graph.relu_backward(x, dy).unwrap();
    (graph, output)
}
#[test]
fn checked_boundary_and_strict_are_exact_selection_and_validate_both_inputs() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            let (mut graph, output) = graph(scalar, mode, PcuNumericalOptions::default());
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                graph
                    .set_value_float_underflow_policy(output, policy)
                    .unwrap();
                let node = graph.node(output).unwrap();
                assert!(matches!(
                    assess(&graph, node),
                    TensorOperationSupport::Supported {
                        route: TensorExecutionRoute::Synthesized,
                        ..
                    }
                ));
                let profile = Profile::from_node(node).unwrap();
                assert!(profile.checked());
                assert_eq!(
                    profile.requirements()[0].min_required_bytes,
                    if scalar == PcuScalarType::F32 {
                        68
                    } else {
                        136
                    }
                );
                let source = profile.source();
                assert!(source.contains("|| (dy &"));
                assert!(source.contains("positive ? dy :"));
                assert_eq!(
                    source.contains("selected &"),
                    policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                );
            }
        }
    }
}
#[test]
fn native_boundary_ieee_is_status_free_and_stronger_requests_are_checked() {
    let options = PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..Default::default()
    };
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for policy in [
                PcuFloatUnderflowPolicy::IeeeAfterRounding,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            ] {
                let (mut graph, output) = graph(scalar, mode, options);
                graph
                    .set_value_float_underflow_policy(output, policy)
                    .unwrap();
                let node = graph.node(output).unwrap();
                let profile = Profile::from_node(node).unwrap();
                let checked = mode == PcuNumericalMode::Strict
                    || policy != PcuFloatUnderflowPolicy::IeeeAfterRounding;
                assert_eq!(profile.checked(), checked);
                assert_eq!(profile.source().contains("atomicMin"), checked);
                assert_eq!(profile.source().contains("* fault"), checked);
                assert_eq!(profile.source().contains("|| (dy &"), checked);
                assert_eq!(profile.numerical_requirements.numerical_mode, mode);
                assert_eq!(profile.numerical_requirements.numerical_options, options);
                assert_eq!(profile.numerical_requirements.float_underflow, policy);
                assert!(matches!(
                    assess(&graph, node),
                    TensorOperationSupport::Supported { .. }
                ));
                if checked {
                    let error = profile
                        .fault(
                            output,
                            PcuExecutionFault {
                                invocation_id: 2,
                                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                                recovered: false,
                            },
                        )
                        .unwrap();
                    assert!(matches!(
                        error,
                        TensorError::ArithmeticFault {
                            element_index: 2,
                            ..
                        }
                    ));
                }
                graph
                    .set_value_numerical_options(
                        output,
                        PcuNumericalOptions {
                            reproducibility: PcuReproducibility::PortableV1,
                            ..options
                        },
                    )
                    .unwrap();
                assert!(Profile::from_node(graph.node(output).unwrap()).is_err());
            }
        }
    }
}
#[test]
fn descriptor_provenance_extent_and_fault_domain_are_checked() {
    let (graph, output) = graph(
        PcuScalarType::F32,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
    );
    let profile = Profile::from_node(graph.node(output).unwrap()).unwrap();
    let mut descriptor = graph.node(output).unwrap();
    descriptor.shape = &[16];
    assert!(matches!(
        assess(&graph, descriptor),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation
        }
    ));
    for index in [0, 16] {
        let error = profile
            .fault(
                output,
                PcuExecutionFault {
                    invocation_id: index,
                    kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                    recovered: false,
                },
            )
            .unwrap();
        assert!(
            matches!(error, TensorError::ArithmeticFault { element_index, .. } if element_index as u64 == index)
        );
    }
    for fault in [
        PcuExecutionFault {
            invocation_id: 17,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand,
            recovered: false,
        },
        PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: false,
        },
        PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: true,
        },
    ] {
        assert!(profile.fault(output, fault).is_err());
    }
}

#[test]
fn low_formats_serve_all_permissions_with_exact_frozen_checked_profiles() {
    use fusion_pcu::PcuPrecisionPolicy;
    let mut profiles = Vec::new();
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
    ] {
        for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
            for compound in [
                PcuCompoundArithmeticPolicy::Checked,
                PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    PcuPrecisionPolicy::Preserve,
                    PcuPrecisionPolicy::BackendOptimized,
                ] {
                    for policy in [
                        PcuFloatUnderflowPolicy::IeeeAfterRounding,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                    ] {
                        let options = PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            ..Default::default()
                        };
                        let (mut graph, output) = graph(scalar, mode, options);
                        graph
                            .set_value_float_underflow_policy(output, policy)
                            .unwrap();
                        let node = graph.node(output).unwrap();
                        let profile = Profile::from_node(node).unwrap();
                        assert!(profile.checked());
                        assert_eq!(profile.numerical_requirements.numerical_mode, mode);
                        assert_eq!(profile.numerical_requirements.numerical_options, options);
                        assert_eq!(profile.numerical_requirements.float_underflow, policy);
                        assert_eq!(
                            profile.numerical_requirements.range_policy,
                            PcuRangePolicy::Reject
                        );
                        assert_eq!(
                            profile.requirements()[0].min_required_bytes,
                            u64::from(scalar.bit_width() / 8) * 17
                        );
                        assert!(matches!(
                            assess(&graph, node),
                            TensorOperationSupport::Supported {
                                route: TensorExecutionRoute::Synthesized,
                                ..
                            }
                        ));
                        assert!(!profiles.contains(&profile));
                        profiles.push(profile);
                        // Captured per-origin metadata survives later graph-default changes.
                        graph.set_numerical_options(PcuNumericalOptions::default());
                        graph.set_numerical_mode(PcuNumericalMode::Boundary);
                        assert_eq!(
                            Profile::from_node(graph.node(output).unwrap()).unwrap(),
                            profile
                        );
                        let source = profile.source();
                        assert!(source.contains("Frozen PCU numerical requirements"));
                        assert!(source.contains("atomicMin"));
                        assert!(source.contains("|| (dy &"));
                        assert!(source.contains("positive ? dy :"));
                        if scalar == PcuScalarType::F8E4M3FN {
                            assert!(source.contains("(x & 0x7fu) > 0x7eu"));
                            assert!(!source.contains("(x & 0x78u) == 0x78u"));
                        }
                        assert_eq!(
                            source.contains("selected &"),
                            policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        );
                    }
                }
            }
        }
    }
    assert_eq!(profiles.len(), 96);
}
