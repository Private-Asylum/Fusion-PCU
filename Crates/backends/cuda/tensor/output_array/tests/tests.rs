//! Fixed-array preflight and terminal publication witnesses.
#[rustfmt::skip]
use super::{
    CudaOwnedPreparedTensorGraph,
    CudaTensorAssessor,
    CudaTensorExecutionError,
    PcuDeviceTensor,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    SmallVec,
    output_array,
    validate_output_array_plan,
};
use fusion_pcu::PcuScalarType;
#[rustfmt::skip]
use super::super::{
    Graph,
    NodeDescriptor,
    TensorOperationAssessor,
    TensorOperationSupport,
    TensorArithmeticRewritePolicy,
    TensorArithmeticCapability,
    TensorPointwiseGroupingPolicy,
    Arc,
    assess_tensor_node,
    prepare_owned_graph_data,
};

struct Owner<'a>(&'a core::cell::Cell<usize>);
impl Drop for Owner<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}

struct PureAssessor;
impl TensorOperationAssessor for PureAssessor {
    fn assess_node(&self, graph: &Graph, node: NodeDescriptor<'_>) -> TensorOperationSupport {
        assess_tensor_node(graph, node)
    }
}

fn plan() -> CudaOwnedPreparedTensorGraph {
    let mut graph = Graph::default();
    let input = graph.input([4], PcuScalarType::F32).unwrap();
    let small = graph.input([2], PcuScalarType::F32).unwrap();
    let outputs = [
        graph.relu(input).unwrap(),
        graph.add(input, input).unwrap(),
        graph.relu(small).unwrap(),
    ];
    let program = graph
        .into_selected_program(
            &outputs,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    let data = prepare_owned_graph_data(&program, &PureAssessor).unwrap();
    CudaOwnedPreparedTensorGraph::from_parts(Arc::new(program), data).unwrap()
}

#[test]
fn exact_count_type_duplicate_and_empty_guards() {
    let mut prepared = plan();
    validate_output_array_plan::<f32, 3>(&prepared).unwrap();
    assert!(matches!(
        validate_output_array_plan::<f32, 2>(&prepared),
        Err(CudaTensorExecutionError::OutputCountMismatch {
            expected: 2,
            actual: 3
        })
    ));
    assert!(matches!(
        validate_output_array_plan::<f64, 3>(&prepared),
        Err(CudaTensorExecutionError::UnsupportedScalarType(_))
    ));
    prepared.data.outputs[1] = prepared.data.outputs[0];
    assert!(matches!(
        validate_output_array_plan::<f32, 3>(&prepared),
        Err(CudaTensorExecutionError::DuplicateOutput(_))
    ));
    prepared.data.outputs.clear();
    assert!(matches!(
        validate_output_array_plan::<f32, 0>(&prepared),
        Err(CudaTensorExecutionError::EmptyOutputs)
    ));
}

#[test]
fn conversion_preserves_order_and_drops_mismatched_owners() {
    let outputs: SmallVec<[u32; 1]> = [7, 3, 11].into_iter().collect();
    assert_eq!(output_array::<_, 3>(outputs).unwrap(), [7, 3, 11]);
    let drops = core::cell::Cell::new(0);
    let outputs: SmallVec<[Owner<'_>; 1]> = [Owner(&drops), Owner(&drops)].into_iter().collect();
    assert!(output_array::<_, 3>(outputs).is_err());
    assert_eq!(drops.get(), 2);
}

#[test]
#[ignore = "requires an actual cuda device"]
#[allow(clippy::too_many_lines)] // Keep the complete owner/fault/retry lifecycle in one witness.
fn native_array_order_shapes_late_fault_old_owners_and_retry() {
    let (_discovery, session) = super::super::tests::cuda_test_session();
    let pool = PcuMemoryPoolId(0x4152_0003);
    let assessor = CudaTensorAssessor::new(&session).unwrap();
    let prepared = plan();
    let ids = prepared.tensor_program().input_values();
    let owner = PcuDeviceTensor::new(
        [4],
        session
            .upload_buffer(pool, &[-1.0_f32, 2.0, 3.0, 4.0])
            .unwrap(),
    )
    .unwrap();
    let small =
        PcuDeviceTensor::new([2], session.upload_buffer(pool, &[5.0_f32, -6.0]).unwrap()).unwrap();
    let input = assessor.borrow_device_input_ref(&owner, pool).unwrap();
    let small_input = assessor.borrow_device_input_ref(&small, pool).unwrap();
    let mut memory = session.memory_provider(pool);
    #[cfg(feature = "allocation-census")]
    crate::reset_cuda_api_census();
    let mut old = assessor
        .execute_owned_program_output_array_from_inputs::<f32, _, 3>(
            &prepared,
            &[(ids[0], &input), (ids[1], &small_input)],
            pool,
            &mut memory,
        )
        .unwrap();
    #[cfg(feature = "allocation-census")]
    assert_eq!(crate::cuda_api_census().kernel_launches, 3);
    assert_eq!(old[0].shape(), &[4]);
    assert_eq!(old[1].shape(), &[4]);
    assert_eq!(old[2].shape(), &[2]);
    let bad = PcuDeviceTensor::new(
        [4],
        session
            .upload_buffer(pool, &[1.0_f32, f32::MAX, 3.0, 4.0])
            .unwrap(),
    )
    .unwrap();
    let bad_input = assessor.borrow_device_input_ref(&bad, pool).unwrap();
    assert!(
        matches!(assessor.execute_owned_program_output_array_from_inputs::<f32, _, 3>(&prepared,
        &[(ids[0], &bad_input), (ids[1], &small_input)], pool, &mut memory),
        Err(CudaTensorExecutionError::ExecutionFault(fault)) if fault.invocation_id == 1
        && fault.kind == fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow && !fault.recovered)
    );
    let retry = assessor
        .execute_owned_program_output_array_from_inputs::<f32, _, 3>(
            &prepared,
            &[(ids[0], &input), (ids[1], &small_input)],
            pool,
            &mut memory,
        )
        .unwrap();
    for outputs in [&old, &retry] {
        for (tensor, expected) in outputs.iter().zip([
            &[0.0_f32, 2.0, 3.0, 4.0][..],
            &[-2.0, 4.0, 6.0, 8.0],
            &[5.0, 0.0],
        ]) {
            let mut actual = vec![0.0_f32; expected.len()];
            session
                .download_buffer(pool, tensor.buffer(), &mut actual)
                .unwrap();
            assert_eq!(
                actual
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
    }
    let replacement: Vec<u8> = [9.0_f32; 4]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    memory
        .transfer_to(old[0].resource_mut(), 0, &replacement)
        .unwrap();
    let mut first = [0.0_f32; 4];
    session
        .download_buffer(pool, old[0].buffer(), &mut first)
        .unwrap();
    assert_eq!(first.map(f32::to_bits), [9.0_f32; 4].map(f32::to_bits));
    session
        .download_buffer(pool, old[1].buffer(), &mut first)
        .unwrap();
    assert_eq!(
        first.map(f32::to_bits),
        [-2.0_f32, 4.0, 6.0, 8.0].map(f32::to_bits)
    );
    session
        .download_buffer(pool, retry[0].buffer(), &mut first)
        .unwrap();
    assert_eq!(
        first.map(f32::to_bits),
        [0.0_f32, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
    session
        .download_buffer(pool, owner.buffer(), &mut first)
        .unwrap();
    assert_eq!(
        first.map(f32::to_bits),
        [-1.0_f32, 2.0, 3.0, 4.0].map(f32::to_bits)
    );
}
