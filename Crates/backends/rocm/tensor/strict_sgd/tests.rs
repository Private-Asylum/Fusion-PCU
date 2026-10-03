//! Cold contract/provenance and compound status decoding without a GPU requirement.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuNumericalOptions,
    PcuReproducibility,
};

fn graph(scalar: PcuScalarType, rate: f32) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let weights = graph.input([17], scalar).unwrap();
    let gradient = graph.input([17], scalar).unwrap();
    let output = graph.sgd_update(weights, gradient, rate).unwrap();
    (graph, output)
}

#[test]
fn strict_profile_freezes_typed_storage_order_policy_and_exact_rate() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            let (mut graph, output) = graph(scalar, 0.1);
            graph
                .set_value_float_underflow_policy(output, policy)
                .unwrap();
            let spec = StrictSgdSpec::from_node(&graph, graph.node(output).unwrap()).unwrap();
            assert_eq!(spec.shape().invocation_count().get(), 17);
            assert_eq!(spec.fault_extent(), 2 * 17);
            assert_eq!(spec.policy, policy);
            let width = if scalar == PcuScalarType::F32 { 4 } else { 8 };
            assert_eq!(
                spec.requirements().map(|r| r.min_required_bytes),
                [17 * width; 3]
            );
            let source = spec.source();
            let rate_bits = if scalar == PcuScalarType::F32 {
                u64::from(0.1_f32.to_bits())
            } else {
                f64::from(0.1_f32).to_bits()
            };
            assert!(source.contains(&format!("({rate_bits}ull)")));
            assert!(source.contains("gradient[id], 2u,"));
            assert!(source.contains("weights[id], product.bits, 1u,"));
            assert!(source.contains("if (id >= 17ull) return;"));
            assert!(source.contains("atomicMin(fault"));
            assert!(!source.contains("fma("));
        }
    }
}

#[test]
fn signed_rate_zero_and_descriptor_policy_or_provenance_are_not_interchangeable() {
    let (mut graph, output) = graph(PcuScalarType::F32, 0.0);
    let mut descriptor = graph.node(output).unwrap();
    let OpDescriptor::SgdUpdate {
        weights, gradient, ..
    } = descriptor.op
    else {
        unreachable!()
    };
    descriptor.op = OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate: -0.0,
    };
    assert!(StrictSgdSpec::from_node(&graph, descriptor).is_none());
    let (foreign, other_output) = self::graph(PcuScalarType::F32, 0.0);
    assert!(StrictSgdSpec::from_node(&graph, foreign.node(other_output).unwrap()).is_none());
    {
        let options = PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        };
        graph.set_value_numerical_options(output, options).unwrap();
        assert!(StrictSgdSpec::from_node(&graph, graph.node(output).unwrap()).is_none());
    }
    graph
        .set_value_numerical_options(output, PcuNumericalOptions::default())
        .unwrap();
    graph
        .set_value_numerical_mode(output, PcuNumericalMode::Boundary)
        .unwrap();
    assert!(StrictSgdSpec::from_node(&graph, graph.node(output).unwrap()).is_none());
}

#[test]
fn compound_fault_steps_decode_in_logical_element_order_and_reject_unproved_status() {
    let (graph, output) = graph(PcuScalarType::F64, 0.5);
    let spec = StrictSgdSpec::from_node(&graph, graph.node(output).unwrap()).unwrap();
    for (event, element, step) in [
        (0, 0, TensorArithmeticStep::Multiply),
        (1, 0, TensorArithmeticStep::Subtract),
        (33, 16, TensorArithmeticStep::Subtract),
    ] {
        assert_eq!(
            spec.fault_error(
                output,
                PcuExecutionFault {
                    invocation_id: event,
                    kind: PcuExecutionFaultKind::ArithmeticOverflow,
                    recovered: false
                }
            )
            .unwrap(),
            TensorError::CompoundArithmeticFault {
                value: output,
                element_index: element,
                reduction_index: 0,
                step,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
            }
        );
    }
    for fault in [
        PcuExecutionFault {
            invocation_id: 34,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: false,
        },
        PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            recovered: true,
        },
        PcuExecutionFault {
            invocation_id: 0,
            kind: PcuExecutionFaultKind::DivideByZero,
            recovered: false,
        },
    ] {
        assert!(spec.fault_error(output, fault).is_err());
    }
}

#[test]
fn empty_launch_is_rejected_cold_but_a_scalar_has_one_invocation() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    for (shape, supported) in [
        (Vec::new(), true),
        (vec![0], false),
        (vec![usize::MAX, 2], false),
    ] {
        let Ok(weights) = graph.input(shape, PcuScalarType::F32) else {
            continue;
        };
        let output = graph.sgd_update(weights, weights, 1.0).unwrap();
        assert_eq!(
            StrictSgdSpec::from_node(&graph, graph.node(output).unwrap()).is_some(),
            supported
        );
    }
}

#[test]
fn standard_error_chain_preserves_structured_compound_fault() {
    let (graph, value) = graph(PcuScalarType::F32, 0.5);
    let spec = StrictSgdSpec::from_node(&graph, graph.node(value).unwrap()).unwrap();
    let expected = spec
        .fault_error(
            value,
            PcuExecutionFault {
                invocation_id: 3,
                kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                recovered: false,
            },
        )
        .unwrap();
    let error = RocmTensorExecutionError::Graph(expected.clone());
    let source = std::error::Error::source(&error).unwrap();
    assert_eq!(source.downcast_ref::<TensorError>(), Some(&expected));
    assert!(std::error::Error::source(&RocmTensorExecutionError::InvalidPlan(value)).is_none());
}

#[test]
fn stronger_strict_permissions_keep_distinct_frozen_cache_tuples() {
    for scalar in [
        fusion_pcu::PcuScalarType::F32,
        fusion_pcu::PcuScalarType::F64,
    ] {
        let mut profiles = Vec::new();
        for underflow in [
            fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            let mut same_arithmetic = None;
            for compound in [
                fusion_pcu::PcuCompoundArithmeticPolicy::Checked,
                fusion_pcu::PcuCompoundArithmeticPolicy::BackendDefined,
            ] {
                for precision in [
                    fusion_pcu::PcuPrecisionPolicy::Preserve,
                    fusion_pcu::PcuPrecisionPolicy::BackendOptimized,
                ] {
                    let mut graph = fusion_pcu::dialect::tensor::Graph::default();
                    graph.set_numerical_mode(fusion_pcu::PcuNumericalMode::Strict);

                    let options = fusion_pcu::PcuNumericalOptions {
                        compound_arithmetic: compound,
                        precision,
                        ..Default::default()
                    };
                    graph.set_numerical_options(options);
                    let left = graph.input([2, 2], scalar).unwrap();
                    let right = graph.input([2, 2], scalar).unwrap();
                    let output = graph.sgd_update(left, right, 0.5).unwrap();
                    graph
                        .set_value_float_underflow_policy(output, underflow)
                        .unwrap();
                    let profile =
                        StrictSgdSpec::from_node(&graph, graph.node(output).unwrap()).unwrap();
                    let requirements = fusion_pcu::PcuImplementationRequirements {
                        numerical_mode: fusion_pcu::PcuNumericalMode::Strict,
                        numerical_options: options,
                        float_underflow: underflow,
                        range_policy: fusion_pcu::PcuRangePolicy::Reject,
                    };
                    assert_eq!(profile.numerical_requirements, requirements);
                    assert!(
                        !profiles.contains(&profile),
                        "requested permissions must retain distinct cache identities"
                    );
                    let source = profile.source();
                    if let Some(previous) = &same_arithmetic {
                        assert_eq!(previous, &source);
                    } else {
                        same_arithmetic = Some(source);
                    }
                    // Later mutable Graph defaults cannot replace the captured invocation tuple.
                    graph.set_numerical_mode(fusion_pcu::PcuNumericalMode::Boundary);
                    graph.set_numerical_options(fusion_pcu::PcuNumericalOptions::default());
                    assert_eq!(profile.numerical_requirements, requirements);
                    profiles.push(profile);
                }
            }
        }
        assert_eq!(profiles.len(), 12);
    }
}
