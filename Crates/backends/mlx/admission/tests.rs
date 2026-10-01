#[rustfmt::skip]
use super::{
    Graph,
    MlxMatmulPlan,
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuNumericalRequirement,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuScalarType,
    TensorUnsupportedReason,
};

fn admitted_options() -> PcuNumericalOptions {
    PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        reproducibility: PcuReproducibility::Unspecified,
    }
}

#[test]
fn selected_program_requires_the_exact_single_matrix_closure() {
    #[rustfmt::skip]
    use fusion_pcu::dialect::tensor::{
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorPointwiseGroupingPolicy,
    };
    for chain in [false, true] {
        let mut graph = Graph::default();
        graph.set_numerical_options(admitted_options());
        let a = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let b = graph.input([2, 2], PcuScalarType::F32).unwrap();
        let product = graph.matmul(a, b).unwrap();
        let output = if chain {
            graph.matmul(product, b).unwrap()
        } else {
            product
        };
        let program = graph
            .into_selected_program(
                &[output],
                TensorArithmeticRewritePolicy::Disabled,
                TensorArithmeticCapability::Strict,
                TensorPointwiseGroupingPolicy::Disabled,
            )
            .unwrap();
        let assessment = MlxMatmulPlan::assess_program(&program);
        if chain {
            assert_eq!(assessment, Err(TensorUnsupportedReason::Operation));
        } else {
            let plan = assessment.unwrap();
            assert_eq!(plan.inputs(), [a, b]);
            assert_eq!(plan.output(), product);
            assert_eq!(program.node_order().len(), 3);
        }
    }
}

#[test]
fn independent_policy_matrix_and_graph_provenance_are_cold() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound in [
            PcuCompoundArithmeticPolicy::Checked,
            PcuCompoundArithmeticPolicy::BackendDefined,
        ] {
            for precision in [
                PcuPrecisionPolicy::Preserve,
                PcuPrecisionPolicy::BackendOptimized,
            ] {
                for reproducibility in [
                    PcuReproducibility::Unspecified,
                    PcuReproducibility::PortableV1,
                ] {
                    let mut graph = Graph::default();
                    graph.set_numerical_mode(mode);
                    let options = PcuNumericalOptions {
                        compound_arithmetic: compound,
                        precision,
                        reproducibility,
                    };
                    graph.set_numerical_options(options);
                    let a = graph.input([2, 3], PcuScalarType::F32).unwrap();
                    let b = graph.input([3, 4], PcuScalarType::F32).unwrap();
                    let c = graph.matmul(a, b).unwrap();
                    let node = graph.node(c).unwrap();
                    let result = MlxMatmulPlan::assess(&graph, node);
                    if mode == PcuNumericalMode::Boundary && options == admitted_options() {
                        let plan = result.unwrap();
                        assert_eq!(plan.inputs(), [a, b]);
                        assert_eq!(plan.output(), c);
                        assert_eq!(plan.shape(), [2, 4]);
                        assert_eq!(plan.numerical_options(), options);
                    } else {
                        assert!(matches!(
                            result,
                            Err(TensorUnsupportedReason::NumericalPolicy { .. })
                        ));
                    }
                    let mut forged = node;
                    forged.numerical_options = admitted_options();
                    forged.numerical_mode = Some(PcuNumericalMode::Boundary);
                    if forged != node {
                        assert_eq!(
                            MlxMatmulPlan::assess(&graph, forged),
                            Err(TensorUnsupportedReason::Operation)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn stronger_underflow_precision_and_scalar_contracts_do_not_disappear() {
    let mut graph = Graph::default();
    graph.set_numerical_options(admitted_options());
    let a = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let b = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let c = graph.matmul(a, b).unwrap();
    for policy in [
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        graph.set_value_float_underflow_policy(c, policy).unwrap();
        assert_eq!(
            MlxMatmulPlan::assess(&graph, graph.node(c).unwrap()),
            Err(TensorUnsupportedReason::UnderflowPolicy(policy))
        );
    }
    graph
        .set_value_float_underflow_policy(c, PcuFloatUnderflowPolicy::IeeeAfterRounding)
        .unwrap();
    graph
        .set_value_numerical_options(
            c,
            PcuNumericalOptions {
                precision: PcuPrecisionPolicy::Preserve,
                ..admitted_options()
            },
        )
        .unwrap();
    assert_eq!(
        MlxMatmulPlan::assess(&graph, graph.node(c).unwrap()),
        Err(TensorUnsupportedReason::NumericalPolicy {
            requirement: PcuNumericalRequirement::Precision,
            options: PcuNumericalOptions {
                precision: PcuPrecisionPolicy::Preserve,
                ..admitted_options()
            }
        })
    );
    let mut other = Graph::default();
    other.set_numerical_options(admitted_options());
    let a = other.input([2, 2], PcuScalarType::F64).unwrap();
    let b = other.input([2, 2], PcuScalarType::F64).unwrap();
    let c = other.matmul(a, b).unwrap();
    assert_eq!(
        MlxMatmulPlan::assess(&other, other.node(c).unwrap()),
        Err(TensorUnsupportedReason::ElementType)
    );
}

#[test]
fn zero_extents_transpose_and_foreign_descriptors_reject() {
    let mut graph = Graph::default();
    graph.set_numerical_options(admitted_options());
    let a = graph.input([2, 0], PcuScalarType::F32).unwrap();
    let b = graph.input([0, 4], PcuScalarType::F32).unwrap();
    let c = graph.matmul(a, b).unwrap();
    assert_eq!(
        MlxMatmulPlan::assess(&graph, graph.node(c).unwrap()),
        Err(TensorUnsupportedReason::Shape)
    );
    let mut other = Graph::default();
    other.set_numerical_options(admitted_options());
    let a = other.input([3, 2], PcuScalarType::F32).unwrap();
    let b = other.input([3, 4], PcuScalarType::F32).unwrap();
    let c = other.matmul_transposed(a, b, true, false).unwrap();
    assert_eq!(
        MlxMatmulPlan::assess(&other, other.node(c).unwrap()),
        Err(TensorUnsupportedReason::Layout)
    );
    assert_eq!(
        MlxMatmulPlan::assess(&graph, other.node(c).unwrap()),
        Err(TensorUnsupportedReason::Operation)
    );
}
