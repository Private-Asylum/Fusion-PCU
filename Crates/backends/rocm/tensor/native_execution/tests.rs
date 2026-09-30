//! Numerical/status and output-copy boundaries must stay outside this completion selector.

use super::batch_native_matmuls;
#[rustfmt::skip]
use fusion_pcu::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuReproducibility,
    PcuScalarType,
};
use fusion_pcu::dialect::tensor::Graph;

fn native_graph(scalar: PcuScalarType) -> (Graph, [fusion_pcu::dialect::tensor::ValueId; 4]) {
    let mut graph = Graph::default();
    graph.set_numerical_options(PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..PcuNumericalOptions::default()
    });
    let left = graph.input([2, 2], scalar).unwrap();
    let right = graph.input([2, 2], scalar).unwrap();
    let first = graph.matmul(left, right).unwrap();
    let second = graph.matmul(first, right).unwrap();
    (graph, [left, right, first, second])
}

#[test]
fn native_matmuls_select_one_terminal_event_for_each_precision() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        for precision in [
            PcuPrecisionPolicy::Preserve,
            PcuPrecisionPolicy::BackendOptimized,
        ] {
            let (mut graph, [_, _, first, second]) = native_graph(scalar);
            graph
                .set_value_numerical_options(
                    first,
                    PcuNumericalOptions {
                        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
                        precision,
                        ..PcuNumericalOptions::default()
                    },
                )
                .unwrap();
            let nodes: Vec<_> = graph.nodes().collect();
            assert!(batch_native_matmuls(&nodes, &[second]));
            assert!(batch_native_matmuls(&nodes, &[first, second]));
            assert!(batch_native_matmuls(&nodes[..3], &[first]));
        }
    }
}

#[test]
fn checked_strict_portable_tight_and_device_status_boundaries_do_not_batch() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        let (mut graph, [_, _, first, second]) = native_graph(scalar);
        let options = graph.node(first).unwrap().numerical_options;
        graph
            .set_value_numerical_options(first, PcuNumericalOptions::default())
            .unwrap();
        assert!(!batch_native_matmuls(
            &graph.nodes().collect::<Vec<_>>(),
            &[second]
        ));
        graph.set_value_numerical_options(first, options).unwrap();
        graph
            .set_value_numerical_mode(first, PcuNumericalMode::Strict)
            .unwrap();
        assert!(!batch_native_matmuls(
            &graph.nodes().collect::<Vec<_>>(),
            &[second]
        ));
        graph
            .set_value_numerical_mode(first, PcuNumericalMode::Boundary)
            .unwrap();
        graph
            .set_value_numerical_options(
                first,
                PcuNumericalOptions {
                    reproducibility: PcuReproducibility::PortableV1,
                    ..options
                },
            )
            .unwrap();
        assert!(!batch_native_matmuls(
            &graph.nodes().collect::<Vec<_>>(),
            &[second]
        ));
        graph.set_value_numerical_options(first, options).unwrap();
        graph
            .set_value_float_underflow_policy(first, PcuFloatUnderflowPolicy::RejectSubnormalResult)
            .unwrap();
        assert!(!batch_native_matmuls(
            &graph.nodes().collect::<Vec<_>>(),
            &[second]
        ));
        graph
            .set_value_float_underflow_policy(first, PcuFloatUnderflowPolicy::IeeeAfterRounding)
            .unwrap();
        let relu = graph.relu(second).unwrap();
        assert!(!batch_native_matmuls(
            &graph.nodes().collect::<Vec<_>>(),
            &[relu]
        ));
    }
}

#[test]
fn input_only_missing_and_host_copy_outputs_do_not_batch() {
    let (graph, [left, _, _, second]) = native_graph(PcuScalarType::F32);
    let nodes: Vec<_> = graph.nodes().collect();
    assert!(!batch_native_matmuls(&nodes[..2], &[left]));
    assert!(!batch_native_matmuls(&nodes, &[second, left]));
    let foreign = Graph::default().input([1], PcuScalarType::F32).unwrap();
    assert!(!batch_native_matmuls(&nodes, &[foreign]));
    assert!(!batch_native_matmuls(&nodes, &[]));
    assert!(!batch_native_matmuls(&[], &[]));
}
