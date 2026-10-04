use super::*;
#[rustfmt::skip]
use crate::{
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuCompoundArithmeticPolicy,
    PcuRangePolicy,
    PcuReproducibility,
    dialect::tensor::{
        OpDescriptor,
        Tensor,
        TensorValue,
        TensorArithmeticStep,
    },
};
use alloc::vec;

fn selected(graph: Graph, outputs: &[ValueId]) -> TensorOwnedSelectedProgram {
    graph
        .into_selected_program(
            outputs,
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap()
}

#[test]
fn heterogeneous_leaf_requests_keep_local_policies_and_enclosing_range() {
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input_typed::<f32>([2, 2]).unwrap();
    let pointwise = graph.add_typed(input, input).unwrap();
    let compound = graph.matmul_typed(pointwise, input).unwrap();
    let local_options = PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::Checked,
        precision: PcuPrecisionPolicy::Preserve,
        reproducibility: PcuReproducibility::PortableV1,
    };
    graph
        .set_value_numerical_mode(compound.erase(), PcuNumericalMode::Strict)
        .unwrap();
    graph
        .set_value_float_underflow_policy(
            pointwise.erase(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    graph
        .set_value_float_underflow_policy(
            compound.erase(),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        )
        .unwrap();
    graph
        .set_value_numerical_options(compound.erase(), local_options)
        .unwrap();
    let program = selected(graph, &[compound.erase()]);
    let enclosing = PcuImplementationRequirements {
        numerical_mode: PcuNumericalMode::Boundary,
        numerical_options: PcuNumericalOptions {
            precision: PcuPrecisionPolicy::BackendOptimized,
            ..PcuNumericalOptions::default()
        },
        float_underflow: PcuFloatUnderflowPolicy::IeeeAfterRounding,
        range_policy: PcuRangePolicy::Clamp,
    };
    let pointwise_request = program
        .operation_fragment(pointwise.erase())
        .unwrap()
        .implementation_requirements(enclosing);
    assert_eq!(pointwise_request.numerical_mode, PcuNumericalMode::Boundary);
    assert_eq!(
        pointwise_request.float_underflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult
    );
    assert_eq!(
        pointwise_request.numerical_options,
        PcuNumericalOptions::default()
    );
    assert_eq!(pointwise_request.range_policy, PcuRangePolicy::Clamp);
    let compound_fragment = program.operation_fragment(compound.erase()).unwrap();
    let compound_request = compound_fragment.implementation_requirements(enclosing);
    assert_eq!(compound_request.numerical_mode, PcuNumericalMode::Strict);
    assert_eq!(
        compound_request.float_underflow,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow
    );
    assert_eq!(compound_request.numerical_options, local_options);
    assert_eq!(compound_request.range_policy, PcuRangePolicy::Clamp);
    // A policy on an earlier producer is not the policy of this operation.
    let earlier_input = compound_fragment
        .program()
        .graph()
        .node(compound_fragment.inputs()[0].child_value)
        .unwrap();
    assert!(earlier_input.float_underflow_policy.is_none());
    assert_eq!(
        earlier_input.numerical_options,
        PcuNumericalOptions::default()
    );
}

#[test]
fn repeated_roles_metadata_and_foreign_identity_remain_exact() {
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input_typed::<f64>([2, 2]).unwrap();
    let product = graph
        .matmul_transposed_typed(input, input, true, false)
        .unwrap();
    let options = PcuNumericalOptions {
        precision: PcuPrecisionPolicy::BackendOptimized,
        ..PcuNumericalOptions::default()
    };
    graph
        .set_value_numerical_options(product.erase(), options)
        .unwrap();
    graph
        .set_value_numerical_mode(product.erase(), PcuNumericalMode::Strict)
        .unwrap();
    graph
        .set_value_float_underflow_policy(
            product.erase(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    let program = selected(graph, &[product.erase()]);
    let fragment = program.operation_fragment(product.erase()).unwrap();
    assert_eq!(fragment.inputs().len(), 1);
    assert_eq!(fragment.operand_input_indexes(), [0, 0]);
    assert_eq!(fragment.inputs()[0].parent_value, input.erase());
    assert_eq!(fragment.parent_value(), product.erase());
    assert_eq!(fragment.parent_operation_index(), 1);
    let node = fragment
        .program()
        .graph()
        .node(fragment.child_output())
        .unwrap();
    let original = program.graph().node(product.erase()).unwrap();
    assert_eq!(node.shape, original.shape);
    assert_eq!(node.scalar_type, original.scalar_type);
    assert_eq!(node.numerical_options, original.numerical_options);
    assert_eq!(node.numerical_mode, original.numerical_mode);
    assert_eq!(node.float_underflow_policy, original.float_underflow_policy);
    assert!(
        matches!(node.op, OpDescriptor::MatMul {left,right,transpose_left:true,transpose_right:false} if left == right && left == fragment.inputs()[0].child_value)
    );
    assert_eq!(fragment.parent_value_for(node.value), Some(product.erase()));
    assert_eq!(
        fragment.parent_value_for(fragment.inputs()[0].child_value),
        Some(input.erase())
    );
    assert_eq!(fragment.parent_value_for(product.erase()), None);
    assert!(matches!(
        program.operation_fragment(input.erase()),
        Err(TensorFragmentError::NotComputed(_))
    ));
    let foreign = Graph::try_new()
        .unwrap()
        .input_typed::<f64>([2, 2])
        .unwrap();
    assert!(matches!(
        program.operation_fragment(foreign.erase()),
        Err(TensorFragmentError::NotSelected(_))
    ));
}

#[test]
fn fragments_execute_a_training_chain_with_parent_storage_and_loss_shape() {
    let mut graph = Graph::try_new().unwrap();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let input = graph.input_typed::<f32>([2, 2]).unwrap();
    let weights = graph.input_typed::<f32>([2, 2]).unwrap();
    let projected = graph.matmul_typed(input, weights).unwrap();
    let activated = graph.relu_typed(projected).unwrap();
    let loss = graph.mean_squared_error_typed(activated, weights).unwrap();
    let derivative = graph
        .relu_backward(activated.erase(), weights.erase())
        .unwrap();
    let updated = graph
        .sgd_update(projected.erase(), derivative, 0.5)
        .unwrap();
    let outputs = [loss.erase(), updated];
    let program = selected(graph, &outputs);
    let values = [
        (
            input.erase(),
            TensorValue::from(Tensor::new([2, 2], vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap()),
        ),
        (
            weights.erase(),
            TensorValue::from(Tensor::new([2, 2], vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap()),
        ),
    ];
    let reference = program.graph().evaluate_checked(&values).unwrap();
    let fragments = program.operation_fragments().unwrap();
    assert_eq!(fragments.len(), 5);
    assert_eq!(
        fragments
            .iter()
            .map(TensorOperationFragment::parent_value)
            .collect::<Vec<_>>(),
        vec![
            projected.erase(),
            activated.erase(),
            loss.erase(),
            derivative,
            updated
        ]
    );
    let materialized = execute_fragments(&fragments, &values);
    for output in outputs {
        let result = materialized
            .iter()
            .find(|(value, _)| *value == output)
            .unwrap();
        assert_eq!(&result.1, reference.value(output).unwrap());
    }
    let loss_fragment = &fragments[2];
    assert!(
        loss_fragment
            .program()
            .graph()
            .node(loss_fragment.child_output())
            .unwrap()
            .shape
            .is_empty()
    );
    assert_eq!(
        loss_fragment
            .program()
            .graph()
            .node(loss_fragment.inputs()[0].child_value)
            .unwrap()
            .shape,
        [2, 2]
    );
    // Child output pins do not alter the parent's inclusive last-use or scratch legality.
    assert_eq!(program.output_values(), outputs);
    let scratch = program
        .scratch_storage_plan(&[projected.erase(), activated.erase(), derivative])
        .unwrap();
    assert!(!scratch.slots().is_empty());
}

#[test]
fn selected_input_keeps_discarded_checked_effect_and_original_fault_identity() {
    let mut graph = Graph::try_new().unwrap();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let input = graph.input_typed::<f32>([1]).unwrap();
    let loss = graph.mean_squared_error_typed(input, input).unwrap();
    let program = selected(graph, &[input.erase()]);
    let fragments = program.operation_fragments().unwrap();
    assert_eq!(fragments.len(), 1);
    let fragment = &fragments[0];
    assert_eq!(fragment.parent_value(), loss.erase());
    assert_eq!(fragment.operand_input_indexes(), [0, 0]);
    let error = fragment
        .program()
        .graph()
        .evaluate_checked(&[(
            fragment.inputs()[0].child_value,
            TensorValue::from(Tensor::new([1], vec![f32::INFINITY]).unwrap()),
        )])
        .unwrap_err();
    let TensorError::CompoundArithmeticFault {
        value,
        step: TensorArithmeticStep::Subtract,
        element_index: 0,
        reduction_index: 0,
        ..
    } = error
    else {
        panic!("lost ordered first-fault domain: {error:?}");
    };
    assert_eq!(fragment.parent_value_for(value), Some(loss.erase()));
    assert_eq!(program.output_values(), &[input.erase()]);
}

#[test]
fn constants_remain_parent_owned_producers_and_fused_schedules_are_rejected() {
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input_typed::<u32>([2]).unwrap();
    let constant = graph.constant_typed(Tensor::new([2], vec![7_u32, 8]).unwrap());
    let uniform = graph.uniform_typed([2], 3_u32).unwrap();
    let added = graph.add_typed(input, constant).unwrap();
    let output = graph.mul_typed(added, uniform).unwrap();
    let program = selected(graph, &[output.erase()]);
    let fragments = program.operation_fragments().unwrap();
    assert_eq!(fragments.len(), 2);
    assert_eq!(fragments[0].inputs()[1].parent_value, constant.erase());
    assert_eq!(fragments[1].inputs()[1].parent_value, uniform.erase());
    assert!(
        fragments
            .iter()
            .flat_map(|fragment| fragment.program().graph().nodes())
            .all(|node| !matches!(
                node.op,
                OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. }
            ))
    );
    assert!(matches!(
        program.graph().node(constant.erase()).unwrap().op,
        OpDescriptor::Constant(_)
    ));
    assert!(matches!(
        program.graph().node(uniform.erase()).unwrap().op,
        OpDescriptor::Uniform { .. }
    ));
    let mut graph = Graph::try_new().unwrap();
    let input = graph.input_typed::<f64>([2]).unwrap().erase();
    let sum = graph.add(input, input).unwrap();
    let output = graph.relu(sum).unwrap();
    // Internal unchecked planner fixture only: production checked Add retains its boundary.
    graph.nodes[sum.index].float_underflow_policy = None;
    let grouped = graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::SingleUseAddRelu,
        )
        .unwrap();
    assert!(matches!(
        grouped.operation_fragments(),
        Err(TensorFragmentError::SelectedFusion { .. })
    ));
    assert!(matches!(
        grouped.operation_fragment(output),
        Err(TensorFragmentError::SelectedFusion { .. })
    ));
    assert!(matches!(
        grouped.operation_fragment(sum),
        Err(TensorFragmentError::NotSelected(_))
    ));
}

#[test]
fn selected_contraction_cannot_silently_revert_to_original_subtraction() {
    let mut graph = Graph::try_new().unwrap();
    let weights = graph.input_typed::<f32>([1]).unwrap();
    let gradient = graph.input_typed::<f32>([1]).unwrap();
    let rate = graph.uniform_typed([1], 0.5_f32).unwrap();
    let scaled = graph.mul_typed(rate, gradient).unwrap();
    let updated = graph.sub_typed(weights, scaled).unwrap();
    let program = graph
        .into_selected_program(
            &[updated.erase()],
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
            TensorPointwiseGroupingPolicy::Disabled,
        )
        .unwrap();
    assert_eq!(program.rewrites().len(), 1);
    assert!(matches!(
        program.operation_fragment(updated.erase()),
        Err(TensorFragmentError::SelectedRewrite { .. })
    ));
    assert!(matches!(
        program.operation_fragments(),
        Err(TensorFragmentError::SelectedRewrite { .. })
    ));
}

fn execute_fragments(
    fragments: &[TensorOperationFragment],
    values: &[(ValueId, TensorValue)],
) -> Vec<(ValueId, TensorValue)> {
    let mut materialized = values.to_vec();
    for fragment in fragments {
        let inputs: Vec<_> = fragment
            .inputs()
            .iter()
            .map(|binding| {
                let value = &materialized
                    .iter()
                    .find(|(id, _)| *id == binding.parent_value)
                    .unwrap()
                    .1;
                let node = fragment
                    .program()
                    .graph()
                    .node(binding.child_value)
                    .unwrap();
                assert!(matches!(node.op, OpDescriptor::Input));
                assert_eq!(node.numerical_mode, None);
                (binding.child_value, value.clone())
            })
            .collect();
        let result = fragment
            .program()
            .graph()
            .evaluate_checked(&inputs)
            .unwrap();
        materialized.push((
            fragment.parent_value(),
            result.value(fragment.child_output()).unwrap().clone(),
        ));
    }
    materialized
}
