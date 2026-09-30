//! Cold numerical admission matrix; these tests require no GPU or loaded `ROCm` libraries.

use super::assess_tensor_node;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuNumericalRequirement,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorExecutionRoute,
    TensorOperationSupport,
    TensorUnsupportedReason,
    ValueId,
};

fn matmul_graph(
    scalar: PcuScalarType,
    mode: PcuNumericalMode,
    options: PcuNumericalOptions,
) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    let left = graph.input([2, 3], scalar).unwrap();
    let right = graph.input([3, 4], scalar).unwrap();
    graph.set_numerical_mode(mode);
    graph.set_numerical_options(options);
    let result = graph.matmul(left, right).unwrap();
    (graph, result)
}

#[test]
fn matrix_admits_only_implemented_independent_policy_combinations() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
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
                        let options = PcuNumericalOptions {
                            compound_arithmetic: compound,
                            precision,
                            reproducibility,
                        };
                        let (graph, result) = matmul_graph(scalar, mode, options);
                        let support = assess_tensor_node(&graph, graph.node(result).unwrap());
                        if reproducibility == PcuReproducibility::PortableV1 {
                            assert_eq!(
                                support,
                                TensorOperationSupport::Unsupported {
                                    reason: TensorUnsupportedReason::NumericalPolicy {
                                        requirement: PcuNumericalRequirement::Reproducibility,
                                        options,
                                    },
                                }
                            );
                        } else if mode == PcuNumericalMode::Boundary
                            && compound == PcuCompoundArithmeticPolicy::BackendDefined
                        {
                            assert_eq!(
                                support,
                                TensorOperationSupport::Supported {
                                    route: TensorExecutionRoute::Library,
                                    workspace_bytes: None,
                                }
                            );
                        } else if mode == PcuNumericalMode::Strict
                            && compound == PcuCompoundArithmeticPolicy::Checked
                        {
                            assert_eq!(
                                support,
                                TensorOperationSupport::Supported {
                                    route: TensorExecutionRoute::Synthesized,
                                    workspace_bytes: Some(0),
                                }
                            );
                        } else {
                            assert!(matches!(
                                support,
                                TensorOperationSupport::Unsupported { .. }
                            ));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn forged_native_permission_cannot_relax_graph_contract() {
    let (graph, result) = matmul_graph(
        PcuScalarType::F32,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions::default(),
    );
    let mut forged = graph.node(result).unwrap();
    forged.numerical_options.compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
    assert_eq!(
        assess_tensor_node(&graph, forged),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation,
        }
    );
}

#[test]
fn native_matmul_rejects_unproved_tight_underflow() {
    let (mut graph, result) = matmul_graph(
        PcuScalarType::F64,
        PcuNumericalMode::Boundary,
        PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            ..PcuNumericalOptions::default()
        },
    );
    graph
        .set_value_float_underflow_policy(result, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap();
    assert_eq!(
        assess_tensor_node(&graph, graph.node(result).unwrap()),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::UnderflowPolicy(
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
        }
    );
}

#[test]
fn native_matmul_validates_actual_scalar_byte_extent() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        let mut graph = Graph::default();
        graph.set_numerical_options(PcuNumericalOptions {
            compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
            ..PcuNumericalOptions::default()
        });
        let dimension = usize::try_from(i32::MAX).unwrap();
        let left = graph.input([dimension, dimension], scalar).unwrap();
        let right = graph.input([dimension, 1], scalar).unwrap();
        let result = graph.matmul(left, right).unwrap();
        let supported = dimension
            .checked_mul(dimension)
            .and_then(|count| count.checked_mul(if scalar == PcuScalarType::F32 { 4 } else { 8 }))
            .is_some();
        assert_eq!(
            matches!(
                assess_tensor_node(&graph, graph.node(result).unwrap()),
                TensorOperationSupport::Supported { .. }
            ),
            supported
        );
    }
}
