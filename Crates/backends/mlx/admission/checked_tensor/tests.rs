//! Cold graph provenance, unused-effect closure, scalar/shape and complete request admission.
use super::*;
#[rustfmt::skip]
use fusion_pcu::{PcuNumericalMode,PcuNumericalOptions,PcuCompoundArithmeticPolicy,PcuPrecisionPolicy,PcuFloatUnderflowPolicy};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{Graph,TensorArithmeticRewritePolicy,TensorArithmeticCapability,TensorPointwiseGroupingPolicy};
fn selected(graph: Graph, output: ValueId) -> TensorOwnedSelectedProgram {
    graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap()
}
#[test]
fn authentic_six_format_identity_relu_and_discarded_effect_freeze_full_tuple() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
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
                        PcuFloatUnderflowPolicy::RejectSubnormalResult,
                        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
                    ] {
                        for profile in 0..3 {
                            let requirements = PcuImplementationRequirements {
                                numerical_mode: mode,
                                numerical_options: PcuNumericalOptions {
                                    compound_arithmetic: compound,
                                    precision,
                                    reproducibility: PcuReproducibility::Unspecified,
                                },
                                float_underflow: policy,
                                range_policy: PcuRangePolicy::Reject,
                            };
                            let mut graph = Graph::default();
                            graph.set_numerical_mode(mode);
                            graph.set_numerical_options(requirements.numerical_options);
                            let input = graph.input([3, 7], scalar).unwrap();
                            let effect = if profile == 0 {
                                None
                            } else {
                                let effect = graph.relu(input).unwrap();
                                graph
                                    .set_value_float_underflow_policy(effect, policy)
                                    .unwrap();
                                Some(effect)
                            };
                            let output = if profile == 1 { effect.unwrap() } else { input };
                            let program = selected(graph, output);
                            let plan = MlxCheckedTensorPlan::assess_program(&program, requirements)
                                .unwrap();
                            assert_eq!(plan.input(), input);
                            assert_eq!(plan.output(), output);
                            assert_eq!(plan.relu_effect(), effect);
                            assert_eq!(plan.shape(), &[3, 7]);
                            assert_eq!(plan.element_count(), 21);
                            assert_eq!(plan.scalar_type(), scalar);
                            assert_eq!(plan.requirements(), requirements);
                            let shape = plan.shape_owner();
                            assert!(Rc::ptr_eq(&shape, &plan.shape_owner()));
                            let mut wrong = requirements;
                            wrong.range_policy = PcuRangePolicy::Clamp;
                            assert!(MlxCheckedTensorPlan::assess_program(&program, wrong).is_err());
                            wrong = requirements;
                            wrong.numerical_options.reproducibility =
                                PcuReproducibility::PortableV1;
                            assert!(MlxCheckedTensorPlan::assess_program(&program, wrong).is_err());
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn extra_effects_nodes_shapes_and_unproved_types_reject_cold() {
    let requirements = PcuImplementationRequirements::default();
    for scalar in [PcuScalarType::Bool, PcuScalarType::I4, PcuScalarType::U4] {
        let mut graph = Graph::default();
        let input = graph.input([3], scalar).unwrap();
        assert!(
            graph
                .into_selected_program(
                    &[input],
                    TensorArithmeticRewritePolicy::Disabled,
                    TensorArithmeticCapability::Strict,
                    TensorPointwiseGroupingPolicy::Disabled,
                )
                .is_err()
        );
    }
    for extent in [0, usize::try_from(i32::MAX).unwrap() + 1] {
        let mut graph = Graph::default();
        let input = graph.input([extent], PcuScalarType::F16).unwrap();
        assert!(
            MlxCheckedTensorPlan::assess_program(&selected(graph, input), requirements).is_err()
        );
    }
    let mut graph = Graph::default();
    let input = graph.input([3], PcuScalarType::F16).unwrap();
    let effect = graph.relu(input).unwrap();
    let _second = graph.relu(effect).unwrap();
    let program = selected(graph, input);
    assert_eq!(program.node_order().len(), 3);
    assert_eq!(
        MlxCheckedTensorPlan::assess_program(&program, requirements),
        Err(TensorUnsupportedReason::Operation)
    );
    let mut graph = Graph::default();
    let input = graph.input([3], PcuScalarType::F16).unwrap();
    let effect = graph.relu(input).unwrap();
    graph
        .set_value_float_underflow_policy(effect, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap();
    assert!(matches!(
        MlxCheckedTensorPlan::assess_program(&selected(graph, effect), requirements),
        Err(TensorUnsupportedReason::UnderflowPolicy(_))
    ));
    let mut graph = Graph::default();
    let input = graph.input([], PcuScalarType::F16).unwrap();
    let plan = MlxCheckedTensorPlan::assess_program(&selected(graph, input), requirements).unwrap();
    assert_eq!(plan.element_count(), 1);
    assert!(plan.shape().is_empty());
}

#[test]
fn all_twenty_two_identity_plans_preserve_type_extent_and_independent_arithmetic_limits() {
    let requirements = PcuImplementationRequirements::default();
    for scalar in PcuScalarType::ALL
        .into_iter()
        .filter(|scalar| scalar.bit_width() >= 8)
    {
        let mut graph = Graph::default();
        let input = graph.input([7, 3], scalar).unwrap();
        let program = selected(graph, input);
        let plan = MlxCheckedTensorPlan::assess_program(&program, requirements).unwrap();
        assert_eq!(plan.scalar_type(), scalar);
        assert_eq!(plan.element_count(), 21);
        assert_eq!(plan.shape(), &[7, 3]);
        assert_eq!(plan.relu_effect(), None);
        if matches!(scalar, PcuScalarType::F128 | PcuScalarType::F256) {
            let mut graph = Graph::default();
            let input = graph.input([3], scalar).unwrap();
            if let Ok(output) = graph.relu(input) {
                assert!(
                    MlxCheckedTensorPlan::assess_program(&selected(graph, output), requirements)
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn source_leaf_storage_does_not_override_arithmetic_node_options() {
    let mut graph = Graph::default();
    let input = graph.input([3], PcuScalarType::BF16).unwrap();
    let options = PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        reproducibility: PcuReproducibility::Unspecified,
    };
    graph.set_numerical_options(options);
    let effect = graph.relu(input).unwrap();
    let requirements = PcuImplementationRequirements {
        numerical_options: options,
        ..PcuImplementationRequirements::default()
    };
    let program = selected(graph, effect);
    assert!(MlxCheckedTensorPlan::assess_program(&program, requirements).is_ok());
    assert!(
        MlxCheckedTensorPlan::assess_program(&program, PcuImplementationRequirements::default())
            .is_err()
    );
    let mut identity = Graph::default();
    let input = identity.input([3], PcuScalarType::BF16).unwrap();
    let plan =
        MlxCheckedTensorPlan::assess_program(&selected(identity, input), requirements).unwrap();
    assert_eq!(plan.requirements(), requirements);
}
