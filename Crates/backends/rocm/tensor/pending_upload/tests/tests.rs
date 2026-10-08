#[rustfmt::skip]
use super::{
    eligible,
    RocmHostedTensorInput,
};
#[rustfmt::skip]
use super::super::{
    Graph,
    RocmOwnedPreparedTensorGraph,
    RocmTensorAssessor,
    RocmTensorExecutionError,
    Tensor,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
    prepare_owned_graph_data,
    tests::{
        PureRocmAssessor,
        rocm_test_session,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuMemoryPoolId,
};
use fusion_pcu::dialect::tensor::ValueId;
use std::sync::Arc;

fn prepare(
    assessor: &RocmTensorAssessor<'_>,
    n: usize,
    math: bool,
) -> (ValueId, RocmOwnedPreparedTensorGraph) {
    let mut graph = Graph::default();
    let input = graph.input_typed::<u32>([n]).unwrap();
    let output = if math {
        let literal = graph.constant_typed(Tensor::new([n], vec![1_u32; n]).unwrap());
        let factor = graph.uniform_typed([n], 2_u32).unwrap();
        let sum = graph.add_typed(input, literal).unwrap();
        graph.mul_typed(sum, factor).unwrap()
    } else {
        input
    };
    let program = graph
        .into_selected_program(
            &[output.erase()],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    (
        input.erase(),
        assessor.prepare_owned_program(program).unwrap(),
    )
}

fn pure_prepared(graph: Graph, outputs: &[ValueId]) -> RocmOwnedPreparedTensorGraph {
    let program = graph
        .into_selected_program(
            outputs,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let data = prepare_owned_graph_data(&program, &PureRocmAssessor).unwrap();
    RocmOwnedPreparedTensorGraph::from_parts(Arc::new(program), data).unwrap()
}

#[test]
fn staging_proof_tracks_concrete_dispatches_and_refuses_missing_or_mismatched_entries() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<u32>([65]).unwrap();
    let factor = graph.uniform_typed([65], 2_u32).unwrap();
    let sum = graph.add_typed(input, factor).unwrap();
    let output = graph.mul_typed(sum, factor).unwrap();
    let mut prepared = pure_prepared(graph, &[output.erase()]);
    assert!(eligible(&prepared));
    assert!(prepared.supports_host_staging());
    assert!(prepared.staging.stream.is_none());
    let index = prepared.data.index_by_value[&sum.erase()];
    let dispatch = prepared.data.fixed_dispatches[index].take().unwrap();
    assert!(!eligible(&prepared));
    prepared.data.fixed_dispatches[index] = Some(dispatch);
    prepared.data.fixed_dispatches[index]
        .as_mut()
        .unwrap()
        .value = output.erase();
    assert!(!eligible(&prepared));
}

#[test]
fn staging_proof_refuses_source_only_multi_output_and_unproved_native_consumers() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>([2, 2]).unwrap();
    assert!(!pure_prepared(graph, &[input.erase()]).supports_host_staging());
    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>([2, 2]).unwrap();
    let sum = graph.add_typed(input, input).unwrap();
    assert!(!pure_prepared(graph, &[input.erase(), sum.erase()]).supports_host_staging());
    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>([2, 2]).unwrap();
    let output = graph.matmul(input.erase(), input.erase()).unwrap();
    assert!(!pure_prepared(graph, &[output]).supports_host_staging());
}

#[path = "aggregate/aggregate.rs"]
mod aggregate;

#[test]
fn staging_declines_spilled_upload_owners_without_limiting_ordinary_graph_arity() {
    let mut graph = Graph::default();
    let inputs = (0..=crate::RetainedHostUploads::INLINE_CAPACITY)
        .map(|_| graph.input_typed::<u32>([65]).unwrap())
        .collect::<Vec<_>>();
    let mut output = inputs[0];
    for &input in &inputs[1..] {
        output = graph.add_typed(output, input).unwrap();
    }
    let prepared = pure_prepared(graph, &[output.erase()]);
    assert_eq!(prepared.data.input_values.len(), 9);
    assert!(!prepared.supports_host_staging());
    assert_eq!(prepared.data.fixed_dispatches.iter().flatten().count(), 8);
}
