//! Admission and complete packed ordered-step fault attribution.
use super::*;
use fusion_pcu::PcuNumericalOptions;
fn graph(scalar: PcuScalarType, count: usize) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let x = graph.input([count], scalar).unwrap();
    let y = graph.input([count], scalar).unwrap();
    let output = graph.mean_squared_error(x, y).unwrap();
    (graph, output)
}
#[test]
fn ordered_loss_freezes_both_extents_and_final_scalar_output() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        let (graph, output) = graph(scalar, 17);
        let profile = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
        assert_eq!(profile.count(), 1);
        assert_eq!(profile.fault_extent(), 3 * 17 + 1);
        let width = if scalar == PcuScalarType::F32 { 4 } else { 8 };
        assert_eq!(
            profile
                .requirements()
                .iter()
                .map(|r| r.min_required_bytes)
                .collect::<Vec<_>>(),
            [17 * width, 17 * width, width]
        );
        for (event, index, step) in [
            (0, 0, TensorArithmeticStep::Subtract),
            (1, 0, TensorArithmeticStep::Multiply),
            (2, 0, TensorArithmeticStep::Add),
            (50, 16, TensorArithmeticStep::Add),
            (51, 17, TensorArithmeticStep::Divide),
        ] {
            assert_eq!(
                profile
                    .fault(
                        output,
                        PcuExecutionFault {
                            invocation_id: event,
                            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                            recovered: false
                        }
                    )
                    .unwrap(),
                TensorError::CompoundArithmeticFault {
                    value: output,
                    element_index: 0,
                    reduction_index: index,
                    step,
                    kind: PcuExecutionFaultKind::ArithmeticUnderflow
                }
            );
        }
        assert!(
            profile
                .fault(
                    output,
                    PcuExecutionFault {
                        invocation_id: 52,
                        kind: PcuExecutionFaultKind::ArithmeticUnderflow,
                        recovered: false
                    }
                )
                .is_err()
        );
    }
}
#[test]
fn independent_unproved_profiles_and_rewritten_descriptors_reject_cold() {
    let (mut graph, output) = graph(PcuScalarType::F32, 17);
    let mut descriptor = graph.node(output).unwrap();
    descriptor.shape = &[1];
    assert!(Profile::from_node(&graph, descriptor).is_err());
    {
        let options = PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        };
        graph.set_value_numerical_options(output, options).unwrap();
        assert!(Profile::from_node(&graph, graph.node(output).unwrap()).is_err());
    }
    graph
        .set_value_numerical_options(output, PcuNumericalOptions::default())
        .unwrap();
    graph
        .set_value_numerical_mode(output, PcuNumericalMode::Boundary)
        .unwrap();
    assert!(Profile::from_node(&graph, graph.node(output).unwrap()).is_err());
}

#[test]
fn cold_strict_loss_uses_no_native_squared_dispatch_or_blas_requirement() {
    struct Assessor;
    impl fusion_pcu::dialect::tensor::TensorOperationAssessor for Assessor {
        fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
            super::super::assess_tensor_node(graph, node)
        }
    }
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        let (graph, output) = graph(scalar, 17);
        let prepared = super::super::prepare_graph(&graph, output, &Assessor).unwrap();
        assert!(prepared.fixed_dispatches.iter().all(Option::is_none));
        assert!(!super::super::nodes_require_blas(&prepared.nodes));
        assert_eq!(
            super::super::mse_scratch_element_count(&graph, &prepared.nodes).unwrap(),
            0
        );
    }
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
                    let output = graph.mean_squared_error(left, right).unwrap();
                    graph
                        .set_value_float_underflow_policy(output, underflow)
                        .unwrap();
                    graph
                        .set_value_float_underflow_policy(output, underflow)
                        .unwrap();
                    let profile = Profile::from_node(&graph, graph.node(output).unwrap()).unwrap();
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
