use super::*;
#[rustfmt::skip]
use fusion_pcu::{
    PcuNumericalOptions,
    PcuPrecisionPolicy,
};

fn graph(mode: PcuNumericalMode, options: PcuNumericalOptions) -> (Graph, super::super::ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(mode);
    graph.set_numerical_options(options);
    let prediction = graph.input([3, 5], PcuScalarType::F32).unwrap();
    let target = graph.input([3, 5], PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    (graph, loss)
}

#[test]
fn loss_policy_matrix_admits_only_explicit_boundary_native() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for compound_arithmetic in [
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
                        compound_arithmetic,
                        precision,
                        reproducibility,
                    };
                    let (graph, loss) = graph(mode, options);
                    let admitted = mode == PcuNumericalMode::Boundary
                        && compound_arithmetic == PcuCompoundArithmeticPolicy::BackendDefined
                        && reproducibility == PcuReproducibility::Unspecified;
                    assert_eq!(
                        matches!(
                            assess(&graph, graph.node(loss).unwrap()),
                            TensorOperationSupport::Supported { .. }
                        ),
                        admitted,
                    );
                    if admitted {
                        assert_eq!(
                            assess(&graph, graph.node(loss).unwrap()),
                            TensorOperationSupport::Supported {
                                route: TensorExecutionRoute::Library,
                                workspace_bytes: Some(60),
                            }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn loss_native_permission_cannot_be_forged_or_override_tight_underflow() {
    let (mut graph, loss) = graph(PcuNumericalMode::Boundary, PcuNumericalOptions::default());
    let mut forged = graph.node(loss).unwrap();
    forged.numerical_options.compound_arithmetic = PcuCompoundArithmeticPolicy::BackendDefined;
    assert_eq!(
        assess(&graph, forged),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation,
        }
    );
    graph
        .set_value_numerical_options(loss, forged.numerical_options)
        .unwrap();
    graph
        .set_value_float_underflow_policy(loss, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap();
    assert_eq!(
        assess(&graph, graph.node(loss).unwrap()),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::UnderflowPolicy(
                PcuFloatUnderflowPolicy::RejectSubnormalResult
            ),
        }
    );
}

#[test]
fn loss_rejects_unrepresentable_native_reduction_count() {
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..PcuNumericalOptions::default()
    });
    let count = usize::try_from(i32::MAX).unwrap() + 1;
    let prediction = graph.input([count], PcuScalarType::F32).unwrap();
    let target = graph.input([count], PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    assert_eq!(
        assess(&graph, graph.node(loss).unwrap()),
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Shape,
        }
    );
}
