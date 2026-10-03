//! Cold policy/provenance bounds and hardware fault/retry checks for ordered SGD.
#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    OpDescriptor,
    TensorArithmeticStep,
    TensorUnsupportedReason,
    ValueId,
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
fn strict_sgd_source_freezes_rate_bits_width_policy_and_order() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for rate in [0.0_f32, -0.0, -0.25, f32::MAX] {
            let (graph, output) = graph(scalar, rate);
            let profile = super::assess(&graph, graph.node(output).unwrap()).unwrap();
            let source = profile.source();
            assert!(source.contains("id * 2ull"));
            assert!(source.contains("product.bits, 1u, 0u"));
            assert!(!source.contains("__fmaf") && !source.contains("__fma_rn"));
            let expected = if scalar == PcuScalarType::F32 {
                u64::from(rate.to_bits())
            } else {
                f64::from(rate).to_bits()
            };
            assert!(source.contains(&format!("({expected}ull)")));
            assert_eq!(profile.count(), 17);
            assert_eq!(profile.fault_extent(), 2 * 17);
            let requirements = profile.requirements();
            assert_eq!(requirements.len(), 3);
            assert!(
                requirements.iter().all(|r| r.min_required_bytes
                    == 17 * if scalar == PcuScalarType::F32 { 4 } else { 8 })
            );
        }
    }
}

#[test]
fn strict_sgd_rejects_unproved_options_absent_mode_and_forged_signed_rate() {
    let (mut graph, output) = graph(PcuScalarType::F32, 0.0);
    let original = graph.node(output).unwrap();
    for mode in [None, Some(PcuNumericalMode::Boundary)] {
        let mut node = original;
        node.numerical_mode = mode;
        assert!(matches!(
            super::assess(&graph, node),
            Err(TensorUnsupportedReason::NumericalPolicy { .. })
        ));
    }
    {
        let options = PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        };
        let mut node = original;
        node.numerical_options = options;
        assert!(matches!(
            super::assess(&graph, node),
            Err(TensorUnsupportedReason::NumericalPolicy { .. })
        ));
    }
    let OpDescriptor::SgdUpdate {
        weights, gradient, ..
    } = original.op
    else {
        panic!("SGD descriptor");
    };
    let mut forged = original;
    forged.op = OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate: -0.0,
    };
    assert_eq!(
        super::assess(&graph, forged),
        Err(TensorUnsupportedReason::Operation)
    );
    let scalar = graph.mul(weights, gradient).unwrap();
    let mut forged = graph.node(scalar).unwrap();
    forged.op = OpDescriptor::SgdUpdate {
        weights,
        gradient,
        learning_rate: 0.0,
    };
    assert!(super::assess(&graph, forged).is_err());
}

#[test]
fn strict_sgd_rejects_empty_shape_wide_count_and_nonfloat_profile() {
    for (shape, scalar) in [
        ([0], PcuScalarType::F32),
        ([usize::try_from(u32::MAX).unwrap() + 1], PcuScalarType::F32),
    ] {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let w = graph.input(shape, scalar).unwrap();
        let g = graph.input(shape, scalar).unwrap();
        let output = graph.sgd_update(w, g, 0.5).unwrap();
        assert!(super::assess(&graph, graph.node(output).unwrap()).is_err());
    }
    let (g, output) = graph(PcuScalarType::F32, 0.5);
    let mut node = g.node(output).unwrap();
    node.scalar_type = PcuScalarType::F16;
    assert_eq!(
        super::assess(&g, node),
        Err(TensorUnsupportedReason::ElementType)
    );
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        let (mut graph, output) = graph(PcuScalarType::F32, 0.5);
        // Policies are frozen per-node, so build a new SGD node after changing the default.
        let OpDescriptor::SgdUpdate {
            weights, gradient, ..
        } = graph.node(output).unwrap().op
        else {
            panic!("SGD descriptor");
        };
        let updated = graph.sgd_update(weights, gradient, 0.5).unwrap();
        graph
            .set_value_float_underflow_policy(updated, policy)
            .unwrap();
        let profile = super::assess(&graph, graph.node(updated).unwrap()).unwrap();
        assert_eq!(profile.policy, policy);
    }
}

#[test]
fn strict_sgd_fault_mapping_has_multiply_subtract_order_and_rejects_corrupt_events() {
    let (graph, output) = graph(PcuScalarType::F64, 0.5);
    let profile = super::assess(&graph, graph.node(output).unwrap()).unwrap();
    for (event, step) in [
        (0, TensorArithmeticStep::Multiply),
        (1, TensorArithmeticStep::Subtract),
        (33, TensorArithmeticStep::Subtract),
    ] {
        assert!(
            matches!(profile.fault(output, PcuExecutionFault { invocation_id: event, kind: PcuExecutionFaultKind::ArithmeticUnderflow, recovered: false }).unwrap(), fusion_pcu::dialect::tensor::TensorError::CompoundArithmeticFault { element_index, reduction_index: 0, step: actual_step, .. } if element_index == usize::try_from(event / 2).unwrap() && actual_step == step)
        );
    }
    for (event, recovered) in [(34, false), (0, true)] {
        assert!(
            profile
                .fault(
                    output,
                    PcuExecutionFault {
                        invocation_id: event,
                        kind: PcuExecutionFaultKind::ArithmeticOverflow,
                        recovered
                    }
                )
                .is_err()
        );
    }
}

#[test]
fn tensor_execution_error_exposes_structured_compound_source_and_leaf_absence() {
    use std::error::Error;
    let (graph, output) = graph(PcuScalarType::F32, 0.5);
    let profile = super::assess(&graph, graph.node(output).unwrap()).unwrap();
    let fault = profile
        .fault(
            output,
            PcuExecutionFault {
                invocation_id: 1,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                recovered: false,
            },
        )
        .unwrap();
    let error = super::CudaTensorExecutionError::Graph(fault.clone());
    assert_eq!(
        error
            .source()
            .unwrap()
            .downcast_ref::<fusion_pcu::dialect::tensor::TensorError>(),
        Some(&fault)
    );
    assert!(
        super::CudaTensorExecutionError::FailedCompletion
            .source()
            .is_none()
    );
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
                    graph
                        .set_value_float_underflow_policy(output, underflow)
                        .unwrap();
                    let profile = super::assess(&graph, graph.node(output).unwrap()).unwrap();
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
