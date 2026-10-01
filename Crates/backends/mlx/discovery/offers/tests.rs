#[rustfmt::skip]
use super::{
    validate_graph_descriptor,
    Graph,
    MlxError,
    MlxMatmulRequest,
};
use fusion_pcu::PcuScalarType;

#[test]
fn forged_descriptor_is_an_error_before_capability_filtering() {
    let mut graph = Graph::default();
    let a = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let b = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let c = graph.matmul(a, b).unwrap();
    let node = graph.node(c).unwrap();
    validate_graph_descriptor(&MlxMatmulRequest {
        graph: &graph,
        node,
    })
    .unwrap();
    let mut forged = node;
    forged.scalar_type = PcuScalarType::F64;
    assert!(matches!(
        validate_graph_descriptor(&MlxMatmulRequest {
            graph: &graph,
            node: forged
        }),
        Err(MlxError::InvalidRequest(_))
    ));
}

#[test]
fn authentic_unsupported_operation_is_not_a_provenance_error() {
    let mut graph = Graph::default();
    let a = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let node = graph.node(a).unwrap();
    validate_graph_descriptor(&MlxMatmulRequest {
        graph: &graph,
        node,
    })
    .unwrap();
    assert!(crate::MlxMatmulPlan::assess(&graph, node).is_err());
}
