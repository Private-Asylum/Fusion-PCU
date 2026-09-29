use super::*;
#[rustfmt::skip]
use crate::{
    PcuBf16Bits,
    PcuF16Bits,
    core::PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    PcuMemoryAccess,
    PcuMemoryOverlap,
    PcuMemoryRange,
    PcuMemoryResource,
    PcuMemoryPoolId,
    PcuMemoryResourceOrigin,
};

#[test]
fn owned_selected_program_matches_borrowed_plan_without_rebuilding_reference_semantics() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let bias = graph
        .constant_typed::<f32>(Tensor::new([2], vec![-2.0, 1.0]).unwrap())
        .erase();
    let sum = graph.add(input, bias).unwrap();
    let activated = graph.relu(sum).unwrap();
    let output = graph.mul(activated, activated).unwrap();
    let inputs = [(
        input,
        TensorValue::from(Tensor::<f32>::new([2], vec![1.0, 3.0]).unwrap()),
    )];
    let reference = graph.evaluate(&inputs).unwrap();

    let plan = graph.execution_plan_for_outputs(&[output]).unwrap();
    let source_constraints = plan.storage_constraints().unwrap();
    let lowering = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    );
    let operation_constraints = lowering.operation_storage_constraints().unwrap();
    let scratch = lowering.scratch_storage_plan(&[activated, output]).unwrap();
    let expected_operation_count = lowering.operations().len();
    drop(lowering);
    drop(plan);

    let program = graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::SingleUseAddRelu,
        )
        .unwrap();
    assert_eq!(program.output_values(), &[output]);
    assert_eq!(program.input_values(), &[input]);
    assert_eq!(program.operations().len(), expected_operation_count);
    assert!(program.operations().iter().any(|operation| matches!(
        operation,
        TensorOwnedSelectedOperation::FusedAddRelu {
            add_output,
            relu_output,
            left,
            right,
        } if *add_output == sum && *relu_output == activated && *left == input && *right == bias
    )));
    assert_eq!(program.source_storage_constraints(), source_constraints);
    assert_eq!(
        program.operation_storage_constraints(),
        operation_constraints
    );
    assert_eq!(
        program.scratch_storage_plan(&[activated, output]).unwrap(),
        scratch
    );
    assert_eq!(
        program
            .execute_reference(&inputs)
            .unwrap()
            .value_typed::<f32>(output)
            .unwrap(),
        reference.value_typed::<f32>(output).unwrap()
    );
    assert_eq!(program.graph().node(output).unwrap().value, output);
}

fn selected_program(graph: Graph, outputs: &[ValueId]) -> TensorOwnedSelectedProgram {
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
fn consumed_relu_reuse_proof_requires_direct_unpinned_single_use_input() {
    let mut graph = Graph::default();
    let input = graph.input([2, 3], PcuScalarType::F64).unwrap();
    let output = graph.relu(input).unwrap();
    let program = selected_program(graph, &[output]);
    let proof = program.prove_consumed_relu_reuse(input, output).unwrap();

    assert_eq!(proof.graph_id(), input.graph_id);
    assert_eq!(proof.input(), input);
    assert_eq!(proof.output(), output);
    assert_eq!(proof.scalar_type(), PcuScalarType::F64);
    assert_eq!(proof.bytes(), 48);
    assert_eq!(proof.alignment_bytes(), 8);
}

#[test]
fn consumed_relu_reuse_proof_rejects_fanout_pins_wrong_ops_and_foreign_values() {
    let mut fanout = Graph::default();
    let input = fanout.input([2], PcuScalarType::F32).unwrap();
    let relu = fanout.relu(input).unwrap();
    let other = fanout.add(input, input).unwrap();
    let program = selected_program(fanout, &[relu, other]);
    assert!(matches!(
        program.prove_consumed_relu_reuse(input, relu),
        Err(TensorStorageReuseError::InputUseCount { value, actual: 3 }) if value == input
    ));
    assert!(matches!(
        program.prove_consumed_relu_reuse(input, input),
        Err(TensorStorageReuseError::OutputIsNotSelectedOutput(value)) if value == input
    ));

    let mut pinned = Graph::default();
    let input = pinned.input([2], PcuScalarType::F32).unwrap();
    let relu = pinned.relu(input).unwrap();
    let program = selected_program(pinned, &[input, relu]);
    assert!(matches!(
        program.prove_consumed_relu_reuse(input, relu),
        Err(TensorStorageReuseError::InputIsSelectedOutput(value)) if value == input
    ));

    let mut nonterminal = Graph::default();
    let input = nonterminal.input([2], PcuScalarType::F32).unwrap();
    let relu = nonterminal.relu(input).unwrap();
    let other = nonterminal.input([2], PcuScalarType::F32).unwrap();
    let downstream = nonterminal.add(relu, other).unwrap();
    let program = selected_program(nonterminal, &[relu, downstream]);
    assert!(matches!(
        program.prove_consumed_relu_reuse(input, relu),
        Err(TensorStorageReuseError::OutputHasSelectedConsumers { value, actual: 1 })
            if value == relu
    ));

    let mut wrong_op = Graph::default();
    let input = wrong_op.input([2], PcuScalarType::F32).unwrap();
    let other = wrong_op.input([2], PcuScalarType::F32).unwrap();
    let output = wrong_op.add(input, other).unwrap();
    let program = selected_program(wrong_op, &[output]);
    assert!(matches!(
        program.prove_consumed_relu_reuse(input, output),
        Err(TensorStorageReuseError::NotDirectRelu { input: found, output: found_output })
            if found == input && found_output == output
    ));

    let mut foreign_graph = Graph::default();
    let foreign_input = foreign_graph.input([2], PcuScalarType::F32).unwrap();
    let foreign_output = foreign_graph.relu(foreign_input).unwrap();
    let foreign_program = selected_program(foreign_graph, &[foreign_output]);
    assert!(matches!(
        foreign_program.prove_consumed_relu_reuse(input, foreign_output),
        Err(TensorStorageReuseError::ValueNotInProgram(value)) if value == input
    ));

    let mut computed_input = Graph::default();
    let source = computed_input.input([2], PcuScalarType::F32).unwrap();
    let intermediate = computed_input.relu(source).unwrap();
    let output = computed_input.relu(intermediate).unwrap();
    let program = selected_program(computed_input, &[output]);
    assert!(matches!(
        program.prove_consumed_relu_reuse(intermediate, output),
        Err(TensorStorageReuseError::InputIsNotGraphInput(value)) if value == intermediate
    ));
}

#[test]
fn consumed_relu_reuse_ignores_graph_branches_pruned_from_selected_program() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let selected_output = graph.relu(input).unwrap();
    let unused_branch = graph.add(input, input).unwrap();
    let program = selected_program(graph, &[selected_output]);

    assert!(!program.selected_nodes().contains(&unused_branch));
    assert_eq!(
        program
            .prove_consumed_relu_reuse(input, selected_output)
            .unwrap()
            .input(),
        input
    );
}

#[test]
fn consumed_binary_donor_proof_preserves_operation_and_operand_order() {
    for (operation, donor_operand) in [
        (TensorBinaryOperation::Add, TensorBinaryOperand::Left),
        (TensorBinaryOperation::Add, TensorBinaryOperand::Right),
        (TensorBinaryOperation::Sub, TensorBinaryOperand::Left),
        (TensorBinaryOperation::Sub, TensorBinaryOperand::Right),
        (TensorBinaryOperation::Mul, TensorBinaryOperand::Left),
        (TensorBinaryOperation::Mul, TensorBinaryOperand::Right),
    ] {
        let mut graph = Graph::default();
        let left = graph.input([2, 3], PcuScalarType::F64).unwrap();
        let right = graph.input([2, 3], PcuScalarType::F64).unwrap();
        let (donor, other) = if donor_operand == TensorBinaryOperand::Left {
            (left, right)
        } else {
            (right, left)
        };
        let output = match operation {
            TensorBinaryOperation::Add => graph.add(left, right),
            TensorBinaryOperation::Sub => graph.sub(left, right),
            TensorBinaryOperation::Mul => graph.mul(left, right),
        }
        .unwrap();
        let program = selected_program(graph, &[output]);
        let proof = program
            .prove_consumed_binary_donor(donor, other, output)
            .unwrap();

        assert_eq!(proof.graph_id(), donor.graph_id);
        assert_eq!(proof.donor(), donor);
        assert_eq!(proof.other(), other);
        assert_eq!(proof.output(), output);
        assert_eq!(proof.operation(), operation);
        assert_eq!(proof.donor_operand(), donor_operand);
        assert_eq!(proof.scalar_type(), PcuScalarType::F64);
        assert_eq!(proof.bytes(), 48);
        assert_eq!(proof.alignment_bytes(), 8);
    }
}

#[test]
fn consumed_binary_donor_proof_rejects_repeated_use_pins_and_nonterminal_work() {
    let mut repeated = Graph::default();
    let input = repeated.input([4], PcuScalarType::F32).unwrap();
    let other = repeated.input([4], PcuScalarType::F32).unwrap();
    let output = repeated.add(input, other).unwrap();
    let program = selected_program(repeated, &[output]);
    assert!(matches!(
        program.prove_consumed_binary_donor(input, input, output),
        Err(TensorStorageReuseError::DonorAndOtherAreSameValue(value)) if value == input
    ));

    let mut fanout = Graph::default();
    let donor = fanout.input([4], PcuScalarType::F32).unwrap();
    let other = fanout.input([4], PcuScalarType::F32).unwrap();
    let output = fanout.sub(donor, other).unwrap();
    let extra = fanout.relu(donor).unwrap();
    let program = selected_program(fanout, &[output, extra]);
    assert!(matches!(
        program.prove_consumed_binary_donor(donor, other, output),
        Err(TensorStorageReuseError::NotTerminalBinary { .. })
    ));

    let mut pinned = Graph::default();
    let donor = pinned.input([4], PcuScalarType::F32).unwrap();
    let other = pinned.input([4], PcuScalarType::F32).unwrap();
    let output = pinned.mul(donor, other).unwrap();
    let program = selected_program(pinned, &[donor, output]);
    assert!(matches!(
        program.prove_consumed_binary_donor(donor, other, output),
        Err(TensorStorageReuseError::NotTerminalBinary { .. })
    ));

    let mut nonterminal = Graph::default();
    let donor = nonterminal.input([4], PcuScalarType::F32).unwrap();
    let other = nonterminal.input([4], PcuScalarType::F32).unwrap();
    let intermediate = nonterminal.add(donor, other).unwrap();
    let output = nonterminal.relu(intermediate).unwrap();
    let program = selected_program(nonterminal, &[output]);
    assert!(matches!(
        program.prove_consumed_binary_donor(donor, other, intermediate),
        Err(TensorStorageReuseError::NotTerminalBinary { .. })
    ));
}

#[test]
fn consumed_binary_donor_proof_rejects_unsupported_dense_layout_and_foreign_values() {
    let mut unsupported = Graph::default();
    let donor = unsupported.input([2], PcuScalarType::F32).unwrap();
    let other = unsupported.input([2], PcuScalarType::F32).unwrap();
    let output = unsupported.add(donor, other).unwrap();
    let mut program = selected_program(unsupported, &[output]);
    program.graph.nodes[donor.index].scalar_type = PcuScalarType::I4;
    program.graph.nodes[other.index].scalar_type = PcuScalarType::I4;
    program.graph.nodes[output.index].scalar_type = PcuScalarType::I4;
    assert!(matches!(
        program.prove_consumed_binary_donor(donor, other, output),
        Err(TensorStorageReuseError::UnsupportedScalarType { value, scalar_type: PcuScalarType::I4 }) if value == donor
    ));

    let mut graph = Graph::default();
    let donor = graph.input([2], PcuScalarType::F32).unwrap();
    let other = graph.input([2], PcuScalarType::F32).unwrap();
    let output = graph.mul(donor, other).unwrap();
    let program = selected_program(graph, &[output]);
    let mut foreign = Graph::default();
    let foreign_value = foreign.input([2], PcuScalarType::F32).unwrap();
    assert!(matches!(
        program.prove_consumed_binary_donor(foreign_value, other, output),
        Err(TensorStorageReuseError::ValueNotInProgram(value)) if value == foreign_value
    ));
}

#[test]
fn typed_graph_handles_preserve_heterogeneous_scalar_identity_and_checked_erasure() {
    let mut graph = Graph::default();
    let left = graph.input_typed::<f64>([2, 3]).unwrap();
    let right = graph.constant_typed::<f64>(
        Tensor::new([2, 3], vec![f64::from_bits(0x3ff0_0000_0000_0001); 6]).unwrap(),
    );
    let sum = graph.add_typed(left, right).unwrap();
    let output = graph.relu_typed(sum).unwrap();

    assert_eq!(output.scalar_type(), PcuScalarType::F64);
    assert_eq!(graph.shape(output.erase()).unwrap(), &[2, 3]);
    assert_eq!(
        graph.node(output.erase()).unwrap().scalar_type,
        PcuScalarType::F64
    );
    assert_eq!(graph.typed_view::<f64>(output.erase()).unwrap(), output);
    assert!(matches!(
        graph.typed_view::<f32>(output.erase()),
        Err(TensorError::ScalarTypeMismatch {
            value,
            expected: PcuScalarType::F32,
            actual: PcuScalarType::F64,
        }) if value == output.erase()
    ));

    let plan = graph.execution_plan_for_outputs(&[output.erase()]).unwrap();
    assert_eq!(plan.value_liveness()[0].scalar_type, PcuScalarType::F64);
    assert_eq!(plan.value_liveness()[0].output_bytes, Some(48));
    let inputs = [(
        left.erase(),
        TensorValue::from(Tensor::<f64>::new([2, 3], vec![1.0; 6]).unwrap()),
    )];
    let execution = graph.evaluate(&inputs).unwrap();
    assert_eq!(
        execution
            .value_typed::<f64>(output.erase())
            .unwrap()
            .shape(),
        &[2, 3]
    );
}

#[test]
fn dynamic_graph_operations_reject_dtype_mismatch_and_foreign_typed_views() {
    let mut graph = Graph::default();
    let f32_value = graph.input([2], PcuScalarType::F32).unwrap();
    let f64_value = graph.input([2], PcuScalarType::F64).unwrap();
    assert!(matches!(
        graph.add(f32_value, f64_value),
        Err(TensorError::ScalarTypeMismatch {
            value,
            expected: PcuScalarType::F32,
            actual: PcuScalarType::F64,
        }) if value == f64_value
    ));

    let other_graph = Graph::default();
    assert!(matches!(
        other_graph.typed_view::<f32>(f32_value),
        Err(TensorError::UnknownValue(value)) if value == f32_value
    ));
}

#[test]
fn opaque_half_arithmetic_is_rejected_without_blocking_typed_storage() {
    let mut graph = Graph::default();
    let left = graph.input_typed::<PcuF16Bits>([1]).unwrap();
    let right = graph.input_typed::<PcuF16Bits>([1]).unwrap();
    assert!(matches!(
        graph.add_typed(left, right),
        Err(TensorError::UnsupportedScalarType {
            value,
            scalar_type: PcuScalarType::F16,
        }) if value == left.erase()
    ));
}

#[test]
fn reference_assessor_and_evaluator_preserve_f64_identity_values() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F64).unwrap();
    graph.nodes[input.index].scalar_type = PcuScalarType::F64;

    let assessment = graph.assess_with(&TensorReferenceAssessor);
    assert!(matches!(
        assessment.nodes[0].support,
        TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Reference,
            ..
        }
    ));
    let source = Tensor::<f64>::new([2], vec![1.5, -2.25]).unwrap();
    let execution = graph
        .evaluate(&[(input, TensorValue::from(source.clone()))])
        .unwrap();
    assert_eq!(
        execution.value_typed::<f64>(input).unwrap().data(),
        source.data()
    );
}

#[test]
fn typed_host_tensor_storage_preserves_scalar_bits_and_shapes() {
    let f64_bits = [0x7ff8_0000_0000_0042, 0x8000_0000_0000_0000];
    let f64_tensor = Tensor::<f64>::new([2], f64_bits.map(f64::from_bits).to_vec()).unwrap();
    assert_eq!(f64_tensor.scalar_type(), PcuScalarType::F64);
    assert_eq!(f64_tensor.shape(), &[2]);
    assert_eq!(f64_tensor.len(), 2);
    assert!(!f64_tensor.is_empty());
    assert_eq!(
        f64_tensor
            .data()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        f64_bits
    );

    let integer_tensor = Tensor::<i64>::new([2], vec![i64::MIN, i64::MAX]).unwrap();
    assert_eq!(integer_tensor.data(), &[i64::MIN, i64::MAX]);
    let cloned_integer_tensor = integer_tensor.clone();
    assert_eq!(cloned_integer_tensor.data(), integer_tensor.data());
    assert_eq!(integer_tensor.into_data(), vec![i64::MIN, i64::MAX]);

    let f16 = Tensor::<PcuF16Bits>::scalar(PcuF16Bits::from_bits(0x7d21));
    assert_eq!(f16.scalar_type(), PcuScalarType::F16);
    assert_eq!(f16.data()[0].to_bits(), 0x7d21);
    let bf16 = Tensor::<PcuBf16Bits>::new([1], vec![PcuBf16Bits::from_bits(0xff81)]).unwrap();
    assert_eq!(bf16.scalar_type(), PcuScalarType::BF16);
    assert_eq!(bf16.data()[0].to_bits(), 0xff81);

    let empty = Tensor::<u32>::new([0, 4], Vec::new()).unwrap();
    assert!(empty.is_empty());
    assert_eq!(empty.known_uniform_value(), None);
    assert_eq!(empty.into_data(), Vec::<u32>::new());
}

#[test]
fn selected_fused_program_preserves_scalar_layout_facts() {
    let mut graph = Graph::default();
    let left = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let right = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let sum = graph.add(left, right).unwrap();
    let output = graph.relu(sum).unwrap();
    for node in &mut graph.nodes {
        node.scalar_type = PcuScalarType::F64;
    }

    let plan = graph.execution_plan_for_outputs(&[output]).unwrap();
    let requirements = plan.value_storage_requirements().unwrap();
    assert!(requirements.iter().all(|requirement| {
        requirement.scalar_type == PcuScalarType::F64
            && requirement.output_bytes == 48
            && requirement.alignment_bytes == 8
    }));
    assert!(
        plan.storage_constraints()
            .unwrap()
            .iter()
            .all(|constraint| { constraint.left_bytes == 48 && constraint.right_bytes == 48 })
    );

    let selected = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    );
    assert!(matches!(
        selected.operations().last(),
        Some(TensorSelectedOperation::FusedAddRelu { .. })
    ));
    let selected_liveness = selected.operation_liveness().unwrap();
    assert!(selected_liveness.iter().all(|life| {
        life.scalar_type == PcuScalarType::F64
            && life.output_bytes == Some(48)
            && life.alignment_bytes == Some(8)
    }));
    assert!(
        selected
            .operation_storage_constraints()
            .unwrap()
            .iter()
            .all(|constraint| constraint.left_bytes == 48 && constraint.right_bytes == 48)
    );

    let selected = graph
        .into_selected_program(
            &[output],
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::SingleUseAddRelu,
        )
        .unwrap();
    assert!(selected.operation_liveness().iter().all(|life| {
        life.scalar_type == PcuScalarType::F64
            && life.output_bytes == Some(48)
            && life.alignment_bytes == Some(8)
    }));
}

#[test]
#[allow(clippy::cognitive_complexity, clippy::too_many_lines)] // One test covers schedule selection and its numeric contract.
fn sgd_rewrite_is_opt_in_inspectable_and_does_not_change_reference_rounding() {
    let mut graph = Graph::default();
    let weights = graph.input([1], PcuScalarType::F32).unwrap();
    let gradient = graph.input([1], PcuScalarType::F32).unwrap();
    let rate_value = 1.0 + 2.0_f32.powi(-23);
    let gradient_value = 1.0 - 2.0_f32.powi(-23);
    let rate = graph
        .constant_typed::<f32>(Tensor::new([1], vec![rate_value]).unwrap())
        .erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let updated = graph.sub(weights, scaled).unwrap();
    let plan = graph.execution_plan_for_outputs(&[updated]).unwrap();

    assert!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )
        .is_empty()
    );
    assert!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::Strict,
        )
        .is_empty()
    );
    let candidates = plan.sgd_rewrite_candidates(
        TensorArithmeticRewritePolicy::AllowContractedArithmetic,
        TensorArithmeticCapability::ContractedMultiplyAdd,
    );
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].output, updated);
    assert_eq!(candidates[0].multiply, scaled);
    assert_eq!(candidates[0].weights, weights);
    assert_eq!(candidates[0].gradient, gradient);
    assert_eq!(candidates[0].learning_rate.to_bits(), rate_value.to_bits());

    let strict_schedule = plan.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::ContractedMultiplyAdd,
    );
    assert_eq!(strict_schedule.nodes().len(), plan.node_order().len());
    assert!(strict_schedule.rewrites().is_empty());
    assert_eq!(strict_schedule.output_values(), &[updated]);
    assert_eq!(
        strict_schedule.storage_constraints().unwrap(),
        plan.storage_constraints().unwrap()
    );

    let contracted_schedule = plan.select_lowering(
        TensorArithmeticRewritePolicy::AllowContractedArithmetic,
        TensorArithmeticCapability::ContractedMultiplyAdd,
    );
    assert_eq!(
        contracted_schedule.nodes().len(),
        plan.node_order().len() - 2
    );
    assert_eq!(contracted_schedule.output_values(), &[updated]);
    assert_eq!(contracted_schedule.suppressed_values(), &[scaled, rate]);
    assert_eq!(contracted_schedule.rewrites().len(), 1);
    assert_eq!(contracted_schedule.index_of(scaled), None);
    assert_eq!(contracted_schedule.use_counts(), &[1, 1, 1]);
    let pinned_rate = graph.execution_plan_for_outputs(&[updated, rate]).unwrap();
    let pinned_schedule = pinned_rate.select_lowering(
        TensorArithmeticRewritePolicy::AllowContractedArithmetic,
        TensorArithmeticCapability::ContractedMultiplyAdd,
    );
    assert_eq!(pinned_schedule.index_of(rate), Some(2));
    assert_eq!(pinned_schedule.suppressed_values(), &[scaled]);
    let mut other_graph = Graph::default();
    let foreign_value = other_graph.input([1], PcuScalarType::F32).unwrap();
    assert_eq!(contracted_schedule.index_of(foreign_value), None);
    assert!(matches!(
        contracted_schedule.nodes().last().unwrap().op,
        OpDescriptor::SgdUpdate { weights: w, gradient: g, .. } if w == weights && g == gradient
    ));
    // Selection is a separate immutable schedule; the original graph keeps both operations.
    assert!(matches!(
        graph.nodes().nth(3).unwrap().op,
        OpDescriptor::Mul { .. }
    ));
    assert!(matches!(
        graph.nodes().nth(4).unwrap().op,
        OpDescriptor::Sub { .. }
    ));

    // The exact product is 1 - 2^-46, which rounds to 1.0 in f32. The reference's
    // separately rounded subtraction therefore returns +0.0, while one fused operation
    // retains the residual +2^-46. A backend consuming the candidate must opt into this
    // documented numerical difference; it cannot claim bitwise equivalence.
    let execution = graph
        .evaluate(&[
            (
                weights,
                TensorValue::from(Tensor::<f32>::new([1], vec![1.0]).unwrap()),
            ),
            (
                gradient,
                TensorValue::from(Tensor::<f32>::new([1], vec![gradient_value]).unwrap()),
            ),
        ])
        .unwrap();
    let separate = execution.value_typed::<f32>(updated).unwrap().data()[0];
    let contracted = (-rate_value).mul_add(gradient_value, 1.0);
    assert_eq!(separate.to_bits(), 0.0_f32.to_bits());
    assert_eq!(contracted.to_bits(), 0x2880_0000); // +2^-46
    assert_ne!(separate.to_bits(), contracted.to_bits());
    assert_eq!(
        graph.nodes().nth(4).unwrap().op,
        OpDescriptor::Sub {
            left: weights,
            right: scaled,
        }
    );
}

#[test]
fn selected_storage_constraints_account_for_extended_gradient_lifetime() {
    let mut graph = Graph::default();
    let weights = graph.input([1], PcuScalarType::F32).unwrap();
    let gradient = graph.input([1], PcuScalarType::F32).unwrap();
    let rate = graph
        .constant_typed::<f32>(Tensor::new([1], vec![0.25]).unwrap())
        .erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let intervening = graph.relu(weights).unwrap();
    let updated = graph.sub(intervening, scaled).unwrap();
    let source = graph.execution_plan_for_outputs(&[updated]).unwrap();
    // Read-only input storage remains caller-owned beyond its final graph read, so
    // it must not alias writable storage even when graph intervals do not overlap.
    assert!(
        source
            .storage_constraints()
            .unwrap()
            .iter()
            .any(|constraint| { constraint.left == gradient && constraint.right == intervening })
    );
    let selected = source.select_lowering(
        TensorArithmeticRewritePolicy::AllowContractedArithmetic,
        TensorArithmeticCapability::ContractedMultiplyAdd,
    );
    assert_eq!(selected.suppressed_values(), &[scaled, rate]);
    assert!(
        selected
            .storage_constraints()
            .unwrap()
            .iter()
            .any(|constraint| { constraint.left == gradient && constraint.right == intervening })
    );
}

#[test]
fn sgd_rewrite_rejects_shared_multiply_and_nonuniform_rates() {
    let mut graph = Graph::default();
    let weights = graph.input([2], PcuScalarType::F32).unwrap();
    let gradient = graph.input([2], PcuScalarType::F32).unwrap();
    let rate = graph
        .constant_typed::<f32>(Tensor::new([2], vec![0.25, 0.5]).unwrap())
        .erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let updated = graph.sub(weights, scaled).unwrap();
    let also_used = graph.add(scaled, weights).unwrap();
    let plan = graph
        .execution_plan_for_outputs(&[updated, also_used])
        .unwrap();
    assert!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )
        .is_empty()
    );

    let mut graph = Graph::default();
    let weights = graph.input([2], PcuScalarType::F32).unwrap();
    let gradient = graph.input([2], PcuScalarType::F32).unwrap();
    let rate = graph
        .constant_typed::<f32>(Tensor::new([2], vec![0.25, 0.5]).unwrap())
        .erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let updated = graph.sub(weights, scaled).unwrap();
    let plan = graph.execution_plan_for_outputs(&[updated]).unwrap();
    assert!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )
        .is_empty()
    );

    let mut graph = Graph::default();
    let weights = graph.input([2], PcuScalarType::F32).unwrap();
    let gradient = graph.input([2], PcuScalarType::F32).unwrap();
    let rate = graph
        .constant_typed::<f32>(Tensor::new([2], vec![0.25, 0.25]).unwrap())
        .erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let updated = graph.sub(weights, scaled).unwrap();
    let plan = graph.execution_plan_for_outputs(&[updated]).unwrap();
    assert_eq!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )[0]
        .learning_rate
        .to_bits(),
        0.25_f32.to_bits()
    );

    let mut graph = Graph::default();
    let weights = graph.input([1], PcuScalarType::F32).unwrap();
    let gradient = graph.input([1], PcuScalarType::F32).unwrap();
    let rate = graph
        .constant_typed::<f32>(Tensor::new([1], vec![f32::NAN]).unwrap())
        .erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let updated = graph.sub(weights, scaled).unwrap();
    let plan = graph.execution_plan_for_outputs(&[updated]).unwrap();
    assert!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )
        .is_empty()
    );
}

#[test]
fn splat_constants_expose_uniformity_without_changing_tensor_value_equality() {
    let splat = Tensor::splat([4], 0.25).unwrap();
    let materialized = Tensor::new([4], vec![0.25; 4]).unwrap();
    assert_eq!(splat, materialized);
    assert_eq!(splat.data(), materialized.data());
    assert_eq!(splat.known_uniform_value(), Some(0.25));
    assert_eq!(materialized.known_uniform_value(), None);
    assert_eq!(
        Tensor::splat([0], 0.25).unwrap().known_uniform_value(),
        None
    );

    let mut graph = Graph::default();
    let weights = graph.input([4], PcuScalarType::F32).unwrap();
    let gradient = graph.input([4], PcuScalarType::F32).unwrap();
    let rate = graph.constant_typed::<f32>(splat).erase();
    let scaled = graph.mul(rate, gradient).unwrap();
    let updated = graph.sub(weights, scaled).unwrap();
    let plan = graph.execution_plan_for_outputs(&[updated]).unwrap();
    assert_eq!(
        plan.sgd_rewrite_candidates(
            TensorArithmeticRewritePolicy::AllowContractedArithmetic,
            TensorArithmeticCapability::ContractedMultiplyAdd,
        )[0]
        .learning_rate
        .to_bits(),
        0.25_f32.to_bits()
    );
}

#[test]
fn graph_uniform_is_compact_but_keeps_dense_logical_storage_facts() {
    let mut graph = Graph::default();
    let input = graph.input([4], PcuScalarType::F32).unwrap();
    let uniform = graph
        .uniform_value([4], TensorScalarValue::F32(2.5))
        .unwrap();
    let sum = graph.add(input, uniform).unwrap();

    let uniform_node = graph.nodes().nth(1).unwrap();
    assert_eq!(uniform_node.value, uniform);
    assert_eq!(uniform_node.shape, &[4]);
    assert_eq!(
        uniform_node.op,
        OpDescriptor::Uniform {
            value: TensorScalarValue::F32(2.5)
        }
    );

    let requirements = graph.requirements();
    assert_eq!(requirements.values[1].output_bytes, Some(16));
    assert_eq!(requirements.total_value_bytes, Some(48));

    let plan = graph.execution_plan_for_outputs(&[sum]).unwrap();
    assert_eq!(plan.value_liveness()[1].output_bytes, Some(16));
    assert_eq!(plan.peak_live_bytes(), Some(48));
    let constraints = plan.storage_constraints().unwrap();
    assert!(!constraints.iter().any(|constraint| {
        (constraint.left == input && constraint.right == uniform)
            || (constraint.left == uniform && constraint.right == input)
    }));
    assert!(constraints.iter().any(|constraint| {
        constraint.left == uniform && constraint.right == sum && constraint.left_bytes == 16
    }));

    let assessment = graph.assess_with(&ReferenceAssessor);
    assert_eq!(assessment.nodes[1].output_bytes, Some(16));
}

#[test]
fn graph_uniform_cpu_evaluation_materializes_dense_values_and_checks_shape() {
    let mut graph = Graph::default();
    let uniform = graph
        .uniform_value([2, 2], TensorScalarValue::F32(-3.0))
        .unwrap();
    let empty = graph
        .uniform_value([2, 0], TensorScalarValue::F32(7.0))
        .unwrap();
    let scalar = graph
        .uniform_value([], TensorScalarValue::F32(4.0))
        .unwrap();
    assert_eq!(
        graph.uniform_value([usize::MAX, 2], TensorScalarValue::F32(0.0)),
        Err(TensorError::ShapeOverflow)
    );

    let execution = graph.evaluate(&[] as &[(ValueId, TensorValue)]).unwrap();
    assert_eq!(
        execution.value_typed::<f32>(uniform).unwrap().shape(),
        &[2, 2]
    );
    assert_eq!(
        execution.value_typed::<f32>(uniform).unwrap().data(),
        &[-3.0; 4]
    );
    assert_eq!(
        execution.value_typed::<f32>(empty).unwrap().shape(),
        &[2, 0]
    );
    assert!(
        execution
            .value_typed::<f32>(empty)
            .unwrap()
            .data()
            .is_empty()
    );
    assert_eq!(execution.value_typed::<f32>(scalar).unwrap().shape(), &[]);
    assert_eq!(execution.value_typed::<f32>(scalar).unwrap().data(), &[4.0]);
}

struct TestMemory(PcuMemoryOverlap, u64);

impl PcuMemoryResource for TestMemory {
    fn pool(&self) -> PcuMemoryPoolId {
        PcuMemoryPoolId(0)
    }
    fn size_bytes(&self) -> u64 {
        self.1
    }
    fn alignment_bytes(&self) -> u64 {
        1
    }
    fn access(&self) -> PcuMemoryAccess {
        PcuMemoryAccess::ReadWrite
    }
    fn is_device_local(&self) -> Option<bool> {
        None
    }
    fn origin(&self) -> PcuMemoryResourceOrigin {
        PcuMemoryResourceOrigin::ProviderManaged
    }
    fn overlap(&self, other: &Self, _a: PcuMemoryRange, _b: PcuMemoryRange) -> PcuMemoryOverlap {
        match (self.0, other.0) {
            (PcuMemoryOverlap::Unknown, _) | (_, PcuMemoryOverlap::Unknown) => {
                PcuMemoryOverlap::Unknown
            }
            (PcuMemoryOverlap::Overlapping, _) | (_, PcuMemoryOverlap::Overlapping) => {
                PcuMemoryOverlap::Overlapping
            }
            _ => PcuMemoryOverlap::Disjoint,
        }
    }
}

struct AccessMemory(PcuMemoryAccess, u64);

impl PcuMemoryResource for AccessMemory {
    fn pool(&self) -> PcuMemoryPoolId {
        PcuMemoryPoolId(0)
    }
    fn size_bytes(&self) -> u64 {
        64
    }
    fn alignment_bytes(&self) -> u64 {
        self.1
    }
    fn access(&self) -> PcuMemoryAccess {
        self.0
    }
    fn is_device_local(&self) -> Option<bool> {
        None
    }
    fn origin(&self) -> PcuMemoryResourceOrigin {
        PcuMemoryResourceOrigin::ProviderManaged
    }
}

#[test]
fn execution_plan_storage_requirements_include_capacity_and_access() {
    let mut graph = Graph::default();
    let input = graph.input([4], PcuScalarType::F32).unwrap();
    let output = graph.relu(input).unwrap();
    let plan = graph.execution_plan_for_outputs(&[output]).unwrap();
    let requirements = plan.value_storage_requirements().unwrap();
    assert_eq!(requirements.len(), 2);
    assert_eq!(requirements[0].value, input);
    assert_eq!(requirements[0].access, PcuMemoryAccess::ReadOnly);
    assert_eq!(requirements[1].value, output);
    assert_eq!(requirements[1].access, PcuMemoryAccess::ReadWrite);
    assert_eq!(requirements[0].output_bytes, 16);
    assert_eq!(
        requirements[0].validate(&AccessMemory(PcuMemoryAccess::ReadOnly, 4)),
        Ok(())
    );
    assert_eq!(
        requirements[1].validate(&AccessMemory(PcuMemoryAccess::ReadWrite, 4)),
        Ok(())
    );
    assert_eq!(
        requirements[1].validate(&AccessMemory(PcuMemoryAccess::ReadOnly, 4)),
        Err(TensorStorageValidationError::InsufficientAccess {
            value: output,
            required: PcuMemoryAccess::ReadWrite,
            available: PcuMemoryAccess::ReadOnly,
        })
    );
    assert_eq!(
        requirements[1].validate(&AccessMemory(PcuMemoryAccess::ReadWrite, 2)),
        Err(TensorStorageValidationError::InsufficientAlignment {
            value: output,
            required_bytes: 4,
            available_bytes: 2,
        })
    );
}

#[test]
fn storage_constraints_follow_liveness_and_reject_unknown_overlap() {
    let mut graph = Graph::default();
    let left = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let sum = graph.add(left, right).unwrap();
    let output = graph.relu(sum).unwrap();
    // Pinning the input as a requested output keeps its storage live through the final node.
    let plan = graph.execution_plan_for_outputs(&[left, output]).unwrap();
    let constraints = plan.storage_constraints().unwrap();

    // Read-only inputs may alias each other, while each live computed output needs distinct
    // storage from every value still live at its production/use point.
    assert!(
        !constraints
            .iter()
            .any(|c| c.left == left && c.right == right)
    );
    assert!(constraints.iter().any(|c| c.left == left && c.right == sum));
    assert!(
        constraints
            .iter()
            .any(|c| c.left == sum && c.right == output)
    );
    assert!(
        constraints
            .iter()
            .any(|c| c.left == left && c.right == output)
    );

    let constraint = constraints
        .iter()
        .find(|c| c.left == left && c.right == sum)
        .unwrap();
    let a = TestMemory(PcuMemoryOverlap::Disjoint, 64);
    let b = TestMemory(PcuMemoryOverlap::Unknown, 64);
    assert_eq!(constraint.validate(&[(left, &a), (sum, &a)]), Ok(()));
    assert_eq!(constraint.validate_resources(&a, &a), Ok(()));
    assert_eq!(
        constraint.validate(&[(left, &a), (sum, &b)]),
        Err(TensorStorageValidationError::UnknownOverlap { left, right: sum })
    );
    let overlapping = TestMemory(PcuMemoryOverlap::Overlapping, 64);
    assert!(matches!(
        constraint.validate_resources(&a, &overlapping),
        Err(TensorStorageValidationError::Overlapping { .. })
    ));
    assert!(matches!(
        constraint.validate(&[(left, &a), (sum, &overlapping)]),
        Err(TensorStorageValidationError::Overlapping { .. })
    ));

    let undersized = TestMemory(PcuMemoryOverlap::Disjoint, constraint.right_bytes - 1);
    assert_eq!(
        constraint.validate_resources(&a, &undersized),
        Err(TensorStorageValidationError::ResourceTooSmall {
            value: sum,
            required_bytes: constraint.right_bytes,
            available_bytes: constraint.right_bytes - 1,
        })
    );
}

fn assert_gradient_graph_matches_reference(
    graph: &mut Graph,
    inputs: &[(ValueId, TensorValue)],
    loss: ValueId,
    values: &[ValueId],
) {
    let gradient_values = graph.backward_mse(loss).unwrap();
    let execution = graph.evaluate(inputs).unwrap();
    let reference = execution.gradients(graph, loss).unwrap();
    for value in values {
        let graph_gradient = execution
            .value_typed::<f32>(gradient_values[value.index].unwrap())
            .unwrap();
        let reference_gradient = reference[value.index].as_ref().unwrap();
        assert_eq!(graph_gradient.shape(), reference_gradient.shape());
        for (actual, expected) in graph_gradient.data().iter().zip(reference_gradient.data()) {
            assert!(
                (actual - expected).abs() <= 1.0e-5,
                "{actual} != {expected}"
            );
        }
    }
}

#[test]
fn backward_mse_graph_matches_reference_with_fanout() {
    let mut graph = Graph::default();
    let samples = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let weights = graph.input([3, 2], PcuScalarType::F32).unwrap();
    let prediction = graph.matmul(samples, weights).unwrap();
    let squared = graph.mul(prediction, prediction).unwrap();
    let target = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(squared, target).unwrap();
    let inputs = [
        (
            samples,
            TensorValue::from(
                Tensor::<f32>::new([2, 3], vec![1.0, 2.0, -1.0, 0.5, -2.0, 3.0]).unwrap(),
            ),
        ),
        (
            weights,
            TensorValue::from(
                Tensor::<f32>::new([3, 2], vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.2]).unwrap(),
            ),
        ),
        (
            target,
            TensorValue::from(Tensor::<f32>::new([2, 2], vec![0.0, 1.0, -1.0, 0.5]).unwrap()),
        ),
    ];
    assert_gradient_graph_matches_reference(
        &mut graph,
        &inputs,
        loss,
        &[samples, weights, prediction],
    );
}

#[test]
fn backward_mse_graph_matches_reference_for_transposed_matmul() {
    let mut graph = Graph::default();
    let left = graph.input([3, 2], PcuScalarType::F32).unwrap();
    let right = graph.input([4, 3], PcuScalarType::F32).unwrap();
    let prediction = graph.matmul_transposed(left, right, true, true).unwrap();
    let target = graph.input([2, 4], PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    let inputs = [
        (
            left,
            TensorValue::from(
                Tensor::<f32>::new([3, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5]).unwrap(),
            ),
        ),
        (
            right,
            TensorValue::from(
                Tensor::<f32>::new(
                    [4, 3],
                    vec![
                        0.2, -0.1, 0.3, 0.4, 0.5, -0.2, 0.1, -0.3, 0.6, 0.7, 0.2, 0.8,
                    ],
                )
                .unwrap(),
            ),
        ),
        (
            target,
            TensorValue::from(Tensor::<f32>::new([2, 4], vec![0.0; 8]).unwrap()),
        ),
    ];
    assert_gradient_graph_matches_reference(&mut graph, &inputs, loss, &[left, right]);
}

#[test]
fn relu_backward_checks_shapes_and_matches_strict_derivative() {
    let mut graph = Graph::default();
    let input = graph.input([2], PcuScalarType::F32).unwrap();
    let upstream = graph.input([2], PcuScalarType::F32).unwrap();
    let wrong_shape = graph
        .constant_typed::<f32>(Tensor::new([1, 2], vec![0.0, 0.0]).unwrap())
        .erase();
    assert_eq!(
        graph.relu_backward(input, wrong_shape),
        Err(TensorError::ShapeMismatch {
            left: vec![2],
            right: vec![1, 2]
        })
    );
    let derivative = graph.relu_backward(input, upstream).unwrap();
    let inputs = [
        (
            input,
            TensorValue::from(Tensor::<f32>::new([2], vec![f32::NAN, 0.0]).unwrap()),
        ),
        (
            upstream,
            TensorValue::from(Tensor::<f32>::new([2], vec![3.0, 4.0]).unwrap()),
        ),
    ];
    assert_eq!(
        graph
            .evaluate(&inputs)
            .unwrap()
            .value_typed::<f32>(derivative)
            .unwrap()
            .data(),
        &[0.0, 0.0]
    );
}

#[test]
fn backward_mse_graph_matches_reference_for_two_hidden_relu_layers() {
    let mut graph = Graph::default();
    let samples = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let weights1 = graph.input([3, 4], PcuScalarType::F32).unwrap();
    let hidden1 = graph.matmul(samples, weights1).unwrap();
    let activated1 = graph.relu(hidden1).unwrap();
    let weights2 = graph.input([4, 4], PcuScalarType::F32).unwrap();
    let hidden2 = graph.matmul(activated1, weights2).unwrap();
    let activated2 = graph.relu(hidden2).unwrap();
    let weights3 = graph.input([4, 2], PcuScalarType::F32).unwrap();
    let prediction = graph.matmul(activated2, weights3).unwrap();
    let target = graph.input([2, 2], PcuScalarType::F32).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    let inputs = [
        (
            samples,
            TensorValue::from(
                Tensor::<f32>::new([2, 3], vec![1.0, -2.0, 0.5, -1.0, 0.25, 2.0]).unwrap(),
            ),
        ),
        (
            weights1,
            TensorValue::from(
                Tensor::<f32>::new(
                    [3, 4],
                    vec![
                        0.2, -0.3, 0.5, 0.1, -0.4, 0.6, 0.2, -0.1, 0.3, 0.2, -0.5, 0.4,
                    ],
                )
                .unwrap(),
            ),
        ),
        (
            weights2,
            TensorValue::from(
                Tensor::<f32>::new(
                    [4, 4],
                    vec![
                        0.1, 0.2, -0.3, 0.4, -0.2, 0.5, 0.1, -0.1, 0.3, -0.4, 0.2, 0.6, 0.5, 0.1,
                        -0.2, 0.3,
                    ],
                )
                .unwrap(),
            ),
        ),
        (
            weights3,
            TensorValue::from(
                Tensor::<f32>::new([4, 2], vec![0.2, -0.1, 0.4, 0.3, -0.5, 0.2, 0.1, 0.6]).unwrap(),
            ),
        ),
        (
            target,
            TensorValue::from(Tensor::<f32>::new([2, 2], vec![0.0, 1.0, -0.5, 0.25]).unwrap()),
        ),
    ];
    assert_gradient_graph_matches_reference(
        &mut graph,
        &inputs,
        loss,
        &[
            samples, weights1, hidden1, activated1, weights2, hidden2, activated2, weights3,
            prediction,
        ],
    );
}

struct ReferenceAssessor;

#[test]
fn elementwise_algebra_checks_shapes_and_evaluates_without_broadcasting() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let sub = graph.sub(left, right).unwrap();
    let mul = graph.mul(sub, right).unwrap();
    let shape_mismatch = graph.input(vec![2], PcuScalarType::F32).unwrap();
    assert!(matches!(
        graph.mul(left, shape_mismatch),
        Err(TensorError::ShapeMismatch { .. })
    ));
    let execution = graph
        .evaluate(&[
            (
                left,
                TensorValue::from(
                    Tensor::<f32>::new(vec![2, 2], vec![3.0, 5.0, 7.0, 9.0]).unwrap(),
                ),
            ),
            (
                right,
                TensorValue::from(
                    Tensor::<f32>::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
                ),
            ),
            (
                shape_mismatch,
                TensorValue::from(Tensor::<f32>::new(vec![2], vec![0.0, 0.0]).unwrap()),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution.value_typed::<f32>(sub).unwrap().data(),
        &[2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(
        execution.value_typed::<f32>(mul).unwrap().data(),
        &[2.0, 6.0, 12.0, 20.0]
    );
    assert!(
        matches!(graph.nodes().nth(2).unwrap().op, OpDescriptor::Sub { left: a, right: b } if a == left && b == right)
    );
    assert!(
        matches!(graph.nodes().nth(3).unwrap().op, OpDescriptor::Mul { left: a, right: b } if a == sub && b == right)
    );
    let plan = graph.execution_plan();
    assert_eq!(plan.node_order().len(), graph.nodes().len());
    assert_eq!(plan.output_values(), &[mul, shape_mismatch]);
}

#[test]
fn sgd_update_checks_inputs_and_evaluates_explicit_operation() {
    let mut graph = Graph::default();
    let weights = graph.input([3], PcuScalarType::F32).unwrap();
    let gradient = graph.input([3], PcuScalarType::F32).unwrap();
    let updated = graph.sgd_update(weights, gradient, 0.25).unwrap();
    let wrong_shape = graph.input([2], PcuScalarType::F32).unwrap();
    assert!(matches!(
        graph.sgd_update(weights, wrong_shape, 0.25),
        Err(TensorError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        graph.sgd_update(weights, gradient, f32::NAN),
        Err(TensorError::InvalidLearningRate)
    ));
    assert!(matches!(
        graph.sgd_update(weights, gradient, f32::INFINITY),
        Err(TensorError::InvalidLearningRate)
    ));

    let execution = graph
        .evaluate(&[
            (
                weights,
                TensorValue::from(Tensor::<f32>::new([3], vec![1.0, 2.0, -3.0]).unwrap()),
            ),
            (
                gradient,
                TensorValue::from(Tensor::<f32>::new([3], vec![0.4, -2.0, 8.0]).unwrap()),
            ),
            (
                wrong_shape,
                TensorValue::from(Tensor::<f32>::new([2], vec![0.0, 0.0]).unwrap()),
            ),
        ])
        .unwrap();
    for (actual, expected) in execution
        .value_typed::<f32>(updated)
        .unwrap()
        .data()
        .iter()
        .zip([0.9, 2.5, -5.0])
    {
        assert!((actual - expected).abs() < 1.0e-6);
    }
    assert!(matches!(
        graph.nodes().nth(2).unwrap().op,
        OpDescriptor::SgdUpdate {
            weights: value_weights,
            gradient: value_gradient,
            learning_rate
        } if value_weights == weights
            && value_gradient == gradient
            && (learning_rate - 0.25).abs() < f32::EPSILON
    ));
}

#[test]
fn transposed_matmul_evaluates_and_differentiates_in_original_layout() {
    let mut graph = Graph::default();
    let left = graph.input(vec![3, 2], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![3, 4], PcuScalarType::F32).unwrap();
    let product = graph.matmul_transposed(left, right, true, false).unwrap();
    assert_eq!(graph.shape(product).unwrap(), &[2, 4]);
    assert!(matches!(
        graph.matmul_transposed(left, right, false, false),
        Err(TensorError::MatMulShape { .. })
    ));
    let execution = graph
        .evaluate(&[
            (
                left,
                TensorValue::from(
                    Tensor::<f32>::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ),
            ),
            (
                right,
                TensorValue::from(
                    Tensor::<f32>::new(
                        vec![3, 4],
                        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                    )
                    .unwrap(),
                ),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution.value_typed::<f32>(product).unwrap().data(),
        &[1.0, 3.0, 5.0, 5.0, 2.0, 4.0, 6.0, 6.0]
    );
    let mut scalar_graph = Graph::default();
    let x = scalar_graph.input(vec![3, 2], PcuScalarType::F32).unwrap();
    let y = scalar_graph.input(vec![3, 1], PcuScalarType::F32).unwrap();
    let transposed = scalar_graph.matmul_transposed(x, y, true, false).unwrap();
    let target = scalar_graph
        .constant_typed::<f32>(Tensor::new(vec![2, 1], vec![0.0, 0.0]).unwrap())
        .erase();
    let loss = scalar_graph.mean_squared_error(transposed, target).unwrap();
    let execution = scalar_graph
        .evaluate(&[
            (
                x,
                TensorValue::from(
                    Tensor::<f32>::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
                ),
            ),
            (
                y,
                TensorValue::from(Tensor::<f32>::new(vec![3, 1], vec![1.0, 1.0, 1.0]).unwrap()),
            ),
        ])
        .unwrap();
    let gradients = execution.gradients(&scalar_graph, loss).unwrap();
    assert_eq!(
        gradients[x.index].as_ref().unwrap().data(),
        &[9.0, 12.0, 9.0, 12.0, 9.0, 12.0]
    );
    assert_eq!(
        gradients[y.index].as_ref().unwrap().data(),
        &[33.0, 75.0, 117.0]
    );
}

impl TensorOperationAssessor for ReferenceAssessor {
    fn assess_node(&self, _graph: &Graph, _node: NodeDescriptor<'_>) -> TensorOperationSupport {
        TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Reference,
            workspace_bytes: None,
        }
    }
}

#[test]
fn selected_assessment_and_storage_requirements_preserve_node_order() {
    let mut graph = Graph::default();
    let input = graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let constant = graph
        .constant_typed::<f32>(Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap())
        .erase();
    let sum = graph.add(input, constant).unwrap();
    let output = graph.relu(sum).unwrap();

    let assessment = graph.assess_with(&ReferenceAssessor);
    assert_eq!(assessment.nodes.len(), 4);
    assert_eq!(assessment.nodes[0].node.value, input);
    assert_eq!(assessment.nodes[1].node.value, constant);
    assert_eq!(assessment.nodes[2].node.value, sum);
    assert_eq!(assessment.nodes[3].node.value, output);
    assert!(assessment.nodes.iter().all(|node| {
        matches!(
            node.support,
            TensorOperationSupport::Supported {
                route: TensorExecutionRoute::Reference,
                workspace_bytes: None,
            }
        ) && node.output_bytes == Some(16)
    }));
    assert!(assessment.is_supported());

    let requirements = graph.requirements();
    assert_eq!(requirements.values.len(), 4);
    assert_eq!(requirements.total_value_bytes, Some(64));
    assert!(
        requirements
            .values
            .iter()
            .all(|value| value.output_bytes == Some(16))
    );
}

#[test]
fn execution_plan_reports_stable_order_outputs_and_fanout_liveness() {
    let mut graph = Graph::default();
    let input = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let constant = graph
        .constant_typed::<f32>(Tensor::new(vec![2], vec![1.0, 1.0]).unwrap())
        .erase();
    let first = graph.add(input, constant).unwrap();
    let second = graph.relu(first).unwrap();
    let output = graph.add(first, second).unwrap();

    let plan = graph.execution_plan();
    assert_eq!(plan.node_order(), &[input, constant, first, second, output]);
    assert_eq!(plan.input_values(), &[input]);
    assert_eq!(plan.output_values(), &[output]);
    assert_eq!(plan.value_liveness()[2].first_live_node, 2);
    assert_eq!(plan.value_liveness()[2].last_live_node, 4);
    assert_eq!(plan.peak_live_bytes(), Some(24));

    let inputs = [(
        input,
        TensorValue::from(Tensor::<f32>::new(vec![2], vec![-1.0, 2.0]).unwrap()),
    )];
    let planned = plan.execute_reference(&inputs).unwrap();
    let ordinary = graph.evaluate(&inputs).unwrap();
    assert_eq!(
        planned.value_typed::<f32>(output).unwrap().data(),
        &[0.0, 6.0]
    );
    assert_eq!(
        planned.value_typed::<f32>(output).unwrap(),
        ordinary.value_typed::<f32>(output).unwrap()
    );
}

#[test]
fn selected_output_plan_prunes_unrelated_nodes_and_pins_outputs() {
    let mut graph = Graph::default();
    let input = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let constant = graph
        .constant_typed::<f32>(Tensor::new(vec![2], vec![1.0, 1.0]).unwrap())
        .erase();
    let selected = graph.add(input, constant).unwrap();
    let unrelated = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let _other = graph.relu(unrelated).unwrap();

    let plan = graph.execution_plan_for_outputs(&[selected]).unwrap();
    assert_eq!(plan.node_order(), &[input, constant, selected]);
    assert_eq!(plan.input_values(), &[input]);
    assert_eq!(plan.output_values(), &[selected]);
    assert_eq!(plan.index_of(selected), Some(2));
    assert_eq!(plan.index_of(unrelated), None);
    assert_eq!(plan.use_counts(), &[1, 1, 1]);
    assert_eq!(plan.value_liveness()[0].last_live_node, 2);
    assert_eq!(plan.value_liveness()[1].last_live_node, 2);
    assert_eq!(plan.value_liveness()[2].last_live_node, 2);
    assert_eq!(
        plan.nodes().map(|node| node.value).collect::<Vec<_>>(),
        plan.node_order()
    );

    assert_eq!(
        graph.execution_plan_for_outputs(&[]).unwrap_err(),
        TensorError::EmptyOutputs
    );
    assert_eq!(
        graph
            .execution_plan_for_outputs(&[selected, selected])
            .unwrap_err(),
        TensorError::DuplicateOutput(selected)
    );
    assert_eq!(graph.execution_plan().node_order().len(), 5);
}

#[test]
fn selected_add_relu_group_suppresses_intermediate_only_when_opted_in() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let add = graph.add(left, right).unwrap();
    let relu = graph.relu(add).unwrap();
    let plan = graph.execution_plan_for_outputs(&[relu]).unwrap();

    let default = plan.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );
    assert!(default.pointwise_fusion_groups().is_empty());
    assert_eq!(default.operations().len(), default.nodes().len());
    assert_eq!(default.operation_index_of(add), Some(2));
    assert_eq!(default.operation_use_counts(), default.use_counts());

    let grouped = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    );
    assert_eq!(grouped.pointwise_fusion_groups().len(), 1);
    assert_eq!(grouped.operations().len(), 3);
    assert_eq!(grouped.operation_index_of(add), None);
    assert_eq!(grouped.operation_index_of(relu), Some(2));
    assert_eq!(grouped.operation_use_counts(), &[1, 1, 1]);
    assert!(matches!(
        grouped.operations()[2],
        TensorSelectedOperation::FusedAddRelu {
            add_output,
            relu_output,
            left: lhs,
            right: rhs,
            shape: [2, 2],
        } if add_output == add && relu_output == relu && lhs == left && rhs == right
    ));
    assert_eq!(grouped.pointwise_fusion_groups()[0].shape, vec![2, 2]);
    assert!(
        grouped
            .operation_storage_constraints()
            .unwrap()
            .iter()
            .all(|constraint| constraint.left != add && constraint.right != add)
    );

    // Selection describes a backend schedule; the graph and reference evaluator stay intact.
    let inputs = [
        (
            left,
            TensorValue::from(Tensor::<f32>::new(vec![2, 2], vec![-2.0, 1.0, 0.0, 3.0]).unwrap()),
        ),
        (
            right,
            TensorValue::from(Tensor::<f32>::new(vec![2, 2], vec![1.0, 2.0, -0.0, -4.0]).unwrap()),
        ),
    ];
    assert_eq!(
        graph
            .evaluate(&inputs)
            .unwrap()
            .value_typed::<f32>(relu)
            .unwrap()
            .data(),
        &[0.0, 3.0, 0.0, 0.0]
    );
}

#[test]
fn selected_add_relu_group_respects_fanout_and_output_pins() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let add = graph.add(left, right).unwrap();
    let relu = graph.relu(add).unwrap();
    let fanout = graph.add(add, left).unwrap();
    let fanout_plan = graph.execution_plan_for_outputs(&[relu, fanout]).unwrap();
    let fanout_schedule = fanout_plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    );
    assert!(fanout_schedule.pointwise_fusion_groups().is_empty());
    assert_eq!(fanout_schedule.operation_index_of(add), Some(2));

    let pinned_plan = graph.execution_plan_for_outputs(&[add, relu]).unwrap();
    let pinned_schedule = pinned_plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::SingleUseAddRelu,
    );
    assert!(pinned_schedule.pointwise_fusion_groups().is_empty());
    assert_eq!(pinned_schedule.operation_index_of(add), Some(2));
}

#[test]
fn bounded_mul_identity_preserves_order_repeated_leaves_uniforms_and_downstream_use() {
    let mut graph = Graph::default();
    let value = graph.input(vec![3], PcuScalarType::F32).unwrap();
    let scale = graph
        .uniform_value(vec![3], TensorScalarValue::F32(0.5))
        .unwrap();
    let first = graph.mul(value, value).unwrap();
    let second = graph.mul(first, scale).unwrap();
    let output = graph.add(second, value).unwrap();
    let selected = graph
        .execution_plan_for_outputs(&[output])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
    assert!(selected.bounded_pointwise_fusion_groups().is_empty());
    assert_eq!(selected.bounded_mul_fusion_groups().len(), 1);
    let group = &selected.bounded_mul_fusion_groups()[0];
    assert_eq!(group.output, second);
    assert_eq!(group.leaves, [value, scale]);
    assert_eq!(group.shape, [3]);
    assert_eq!(
        group.steps,
        [
            TensorPointwiseMulStep {
                output: first,
                left: TensorPointwiseOperand::Leaf(0),
                right: TensorPointwiseOperand::Leaf(0)
            },
            TensorPointwiseMulStep {
                output: second,
                left: TensorPointwiseOperand::Step(0),
                right: TensorPointwiseOperand::Leaf(1)
            },
        ]
    );
    assert_eq!(selected.operation_index_of(first), None);
    assert_eq!(selected.operation_index_of(second), Some(2));
    assert!(
        matches!(selected.operations()[2], TensorSelectedOperation::FusedMul { group: ref chosen } if chosen == group)
    );
    assert_eq!(selected.operation_index_of(output), Some(3));
    assert_eq!(selected.operation_use_counts(), &[3, 1, 1, 1]);
}

#[test]
fn bounded_mul_identity_respects_pins_fanout_and_caps() {
    let mut graph = Graph::default();
    let a = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let b = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let c = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let d = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let first = graph.mul(a, b).unwrap();
    let second = graph.mul(first, c).unwrap();
    let fanout = graph.add(first, d).unwrap();
    let fanout_schedule = graph
        .execution_plan_for_outputs(&[second, fanout])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
    assert!(fanout_schedule.bounded_mul_fusion_groups().is_empty());
    let pinned = graph
        .execution_plan_for_outputs(&[first, second])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
    assert!(pinned.bounded_mul_fusion_groups().is_empty());

    let mut too_many_steps = Graph::default();
    let mut value = too_many_steps.input(vec![1], PcuScalarType::F32).unwrap();
    let factor = too_many_steps.input(vec![1], PcuScalarType::F32).unwrap();
    for _ in 0..9 {
        value = too_many_steps.mul(value, factor).unwrap();
    }
    let selected = too_many_steps
        .execution_plan_for_outputs(&[value])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
    assert!(selected.bounded_mul_fusion_groups().is_empty());

    let mut too_many_leaves = Graph::default();
    let leaves = (0..5)
        .map(|_| too_many_leaves.input(vec![1], PcuScalarType::F32).unwrap())
        .collect::<Vec<_>>();
    let mut value = too_many_leaves.mul(leaves[0], leaves[1]).unwrap();
    for leaf in &leaves[2..] {
        value = too_many_leaves.mul(value, *leaf).unwrap();
    }
    let selected = too_many_leaves
        .execution_plan_for_outputs(&[value])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedMulIdentity,
        );
    assert!(selected.bounded_mul_fusion_groups().is_empty());
}

#[test]
fn bounded_add_sub_relu_group_preserves_chain_order_and_unique_leaves() {
    let mut graph = Graph::default();
    let left = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let bias = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let difference = graph.sub(left, right).unwrap();
    let adjusted = graph.sub(bias, difference).unwrap();
    let output = graph.relu(adjusted).unwrap();
    let plan = graph.execution_plan_for_outputs(&[output]).unwrap();
    let selected = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
    );

    assert!(selected.pointwise_fusion_groups().is_empty());
    assert_eq!(selected.bounded_pointwise_fusion_groups().len(), 1);
    let group = &selected.bounded_pointwise_fusion_groups()[0];
    assert_eq!(group.output, output);
    assert_eq!(group.leaves, [left, right, bias]);
    assert_eq!(group.shape, [4]);
    assert_eq!(
        group.steps,
        [
            TensorPointwiseStep {
                output: difference,
                op: TensorPointwiseArithmeticOp::Sub,
                left: TensorPointwiseOperand::Leaf(0),
                right: TensorPointwiseOperand::Leaf(1),
            },
            TensorPointwiseStep {
                output: adjusted,
                op: TensorPointwiseArithmeticOp::Sub,
                left: TensorPointwiseOperand::Leaf(2),
                right: TensorPointwiseOperand::Step(0),
            },
        ]
    );
    assert_eq!(selected.operation_index_of(difference), None);
    assert_eq!(selected.operation_index_of(adjusted), None);
    assert!(matches!(
        selected.operations().last(),
        Some(TensorSelectedOperation::FusedAddSub { group: selected_group })
            if selected_group == group
    ));
    assert_eq!(selected.operation_use_counts(), &[1, 1, 1, 1]);
    assert!(
        selected
            .operation_storage_constraints()
            .unwrap()
            .iter()
            .all(|constraint| constraint.left != difference
                && constraint.right != difference
                && constraint.left != adjusted
                && constraint.right != adjusted)
    );
}

#[test]
fn bounded_add_sub_relu_group_keeps_duplicate_leaf_operand_order() {
    let mut graph = Graph::default();
    let value = graph.input(vec![3], PcuScalarType::F32).unwrap();
    let doubled = graph.add(value, value).unwrap();
    let difference = graph.sub(doubled, value).unwrap();
    let output = graph.relu(difference).unwrap();
    let selected = graph
        .execution_plan_for_outputs(&[output])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    let group = &selected.bounded_pointwise_fusion_groups()[0];
    assert_eq!(group.leaves, [value]);
    assert_eq!(group.steps[0].left, TensorPointwiseOperand::Leaf(0));
    assert_eq!(group.steps[0].right, TensorPointwiseOperand::Leaf(0));
    assert_eq!(group.steps[1].left, TensorPointwiseOperand::Step(0));
    assert_eq!(group.steps[1].right, TensorPointwiseOperand::Leaf(0));
    // Counts describe actual operand reads, including a value used twice in one step.
    assert_eq!(selected.operation_use_counts(), &[3, 1]);
}

#[test]
fn bounded_add_sub_identity_preserves_terminal_output_and_downstream_consumers() {
    let mut graph = Graph::default();
    let left = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let uniform = graph
        .uniform_value(vec![4], TensorScalarValue::F32(0.25))
        .unwrap();
    let bias = graph.input(vec![4], PcuScalarType::F32).unwrap();
    let first = graph.sub(left, uniform).unwrap();
    let output = graph.add(first, bias).unwrap();
    let downstream = graph.sub(output, uniform).unwrap();
    let plan = graph
        .execution_plan_for_outputs(&[output, downstream])
        .unwrap();
    let selected = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
    );

    let group = selected
        .bounded_pointwise_fusion_groups()
        .iter()
        .find(|group| group.output == output)
        .expect("terminal output should have an identity group");
    assert_eq!(group.epilogue, TensorPointwiseEpilogue::Identity);
    assert_eq!(group.leaves, [left, uniform, bias]);
    assert_eq!(group.steps.len(), 2);
    assert_eq!(group.steps[0].output, first);
    assert_eq!(group.steps[1].output, output);
    assert_eq!(selected.operation_index_of(first), None);
    assert!(selected.operation_index_of(output).is_some());
    assert!(matches!(
        selected.operations().iter().find(|operation| selected_operation_output(operation) == output),
        Some(TensorSelectedOperation::FusedAddSub { group: selected_group })
            if selected_group == group
    ));
    assert!(selected.operation_index_of(downstream).is_some());
    assert!(
        selected
            .operation_storage_constraints()
            .unwrap()
            .iter()
            .all(|constraint| constraint.left != first && constraint.right != first)
    );
}

#[test]
fn selected_scratch_plan_reuses_only_after_a_strict_lifetime_gap() {
    let mut graph = Graph::default();
    let input = graph.input([4], PcuScalarType::F32).unwrap();
    let first = graph.relu(input).unwrap();
    let second = graph.relu(first).unwrap();
    let third = graph.relu(second).unwrap();
    let output = graph.relu(third).unwrap();
    let source = graph.execution_plan_for_outputs(&[output]).unwrap();
    let selected = source.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );

    let scratch = selected
        .scratch_storage_plan(&[first, second, third, output])
        .unwrap();
    assert_eq!(scratch.slot_for(first), scratch.slot_for(third));
    assert_ne!(scratch.slot_for(first), scratch.slot_for(second));
    assert_eq!(scratch.slot_for(output), None);
    assert_eq!(scratch.slots().len(), 2);
    assert_eq!(scratch.total_bytes(), 32);
    assert!(
        scratch
            .slots()
            .iter()
            .all(|slot| slot.alignment_bytes == 4 && slot.capacity_bytes == 16)
    );
}

#[test]
fn selected_scratch_plan_keeps_fanout_values_live_across_branches() {
    let mut graph = Graph::default();
    let source_input = graph.input([4], PcuScalarType::F32).unwrap();
    let other_input = graph.input([4], PcuScalarType::F32).unwrap();
    let shared = graph.relu(source_input).unwrap();
    let first_branch = graph.relu(shared).unwrap();
    let later_value = graph.relu(other_input).unwrap();
    let second_branch = graph.add(shared, later_value).unwrap();
    let output = graph.add(first_branch, second_branch).unwrap();
    let source = graph.execution_plan_for_outputs(&[output]).unwrap();
    let selected = source.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );

    let scratch = selected
        .scratch_storage_plan(&[shared, later_value, first_branch, second_branch])
        .unwrap();
    assert_ne!(scratch.slot_for(shared), scratch.slot_for(later_value));
}

#[test]
fn selected_scratch_plan_excludes_every_requested_escape_output() {
    let mut graph = Graph::default();
    let input = graph.input([4], PcuScalarType::F32).unwrap();
    let pinned = graph.relu(input).unwrap();
    let output = graph.relu(pinned).unwrap();
    let source = graph.execution_plan_for_outputs(&[pinned, output]).unwrap();
    let selected = source.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );

    let scratch = selected.scratch_storage_plan(&[pinned, output]).unwrap();
    assert!(scratch.assignments().is_empty());
    assert!(scratch.slots().is_empty());
    assert_eq!(scratch.total_bytes(), 0);
}

#[test]
fn selected_scratch_plan_reuses_and_grows_slots_for_different_extents() {
    let mut graph = Graph::default();
    let small_input = graph.input([2], PcuScalarType::F32).unwrap();
    let large_input = graph.input([8], PcuScalarType::F32).unwrap();
    let small = graph.relu(small_input).unwrap();
    let small_tail = graph.relu(small).unwrap();
    let large = graph.relu(large_input).unwrap();
    let large_tail = graph.relu(large).unwrap();
    let source = graph
        .execution_plan_for_outputs(&[small_tail, large_tail])
        .unwrap();
    let selected = source.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );

    let grown = selected.scratch_storage_plan(&[small, large]).unwrap();
    assert_eq!(grown.slot_for(small), grown.slot_for(large));
    assert_eq!(grown.slots().len(), 1);
    assert_eq!(grown.slots()[0].capacity_bytes, 32);
    assert_eq!(grown.total_bytes(), 32);

    let mut graph = Graph::default();
    let large_input = graph.input([8], PcuScalarType::F32).unwrap();
    let small_input = graph.input([2], PcuScalarType::F32).unwrap();
    let large = graph.relu(large_input).unwrap();
    let large_tail = graph.relu(large).unwrap();
    let small = graph.relu(small_input).unwrap();
    let small_tail = graph.relu(small).unwrap();
    let source = graph
        .execution_plan_for_outputs(&[large_tail, small_tail])
        .unwrap();
    let selected = source.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );

    let fit = selected.scratch_storage_plan(&[large, small]).unwrap();
    assert_eq!(fit.slot_for(large), fit.slot_for(small));
    assert_eq!(fit.slots().len(), 1);
    assert_eq!(fit.slots()[0].capacity_bytes, 32);
    assert_eq!(fit.total_bytes(), 32);
}

#[test]
fn external_read_only_storage_stays_disjoint_after_its_graph_lifetime() {
    let mut graph = Graph::default();
    let early_input = graph.input([4], PcuScalarType::F32).unwrap();
    let transient = graph.relu(early_input).unwrap();
    let early_tail = graph.relu(transient).unwrap();
    // This borrowed input enters the selected schedule after the transient has had its final
    // graph read. Its provider storage still belongs to the caller for the whole execution.
    let late_input = graph.input([4], PcuScalarType::F32).unwrap();
    let late_tail = graph.relu(late_input).unwrap();
    let plan = graph
        .execution_plan_for_outputs(&[early_tail, late_tail])
        .unwrap();
    let transient_life = plan
        .value_liveness()
        .iter()
        .find(|life| life.value == transient)
        .unwrap();
    let late_input_life = plan
        .value_liveness()
        .iter()
        .find(|life| life.value == late_input)
        .unwrap();
    assert!(transient_life.last_live_node < late_input_life.first_live_node);

    let contains_pair = |constraints: &[TensorStorageConstraint]| {
        constraints.iter().any(|constraint| {
            (constraint.left == transient && constraint.right == late_input)
                || (constraint.left == late_input && constraint.right == transient)
        })
    };
    assert!(contains_pair(&plan.storage_constraints().unwrap()));

    let selected = plan.select_lowering(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
    );
    assert!(contains_pair(&selected.storage_constraints().unwrap()));
    assert!(contains_pair(
        &selected.operation_storage_constraints().unwrap()
    ));
}

#[test]
fn bounded_add_sub_identity_keeps_requested_terminal_for_pin_and_fanout() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let first = graph.add(left, right).unwrap();
    let terminal = graph.sub(first, right).unwrap();
    let branch = graph.add(terminal, left).unwrap();
    let selected = graph
        .execution_plan_for_outputs(&[terminal, branch])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );

    let group = selected
        .bounded_pointwise_fusion_groups()
        .iter()
        .find(|group| group.output == terminal)
        .expect("terminal output should remain available to both consumers");
    assert_eq!(group.epilogue, TensorPointwiseEpilogue::Identity);
    assert_eq!(group.steps.len(), 2);
    assert_eq!(selected.operation_index_of(first), None);
    assert!(selected.operation_index_of(terminal).is_some());
    assert!(selected.operation_index_of(branch).is_some());
    assert_eq!(
        selected.operation_use_counts()[selected.operation_index_of(terminal).unwrap()],
        2
    );

    let mut pinned_graph = Graph::default();
    let left = pinned_graph.input(vec![2], PcuScalarType::F32).unwrap();
    let right = pinned_graph.input(vec![2], PcuScalarType::F32).unwrap();
    let pinned_intermediate = pinned_graph.add(left, right).unwrap();
    let terminal = pinned_graph.sub(pinned_intermediate, right).unwrap();
    let selected = pinned_graph
        .execution_plan_for_outputs(&[pinned_intermediate, terminal])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );
    assert!(selected.bounded_pointwise_fusion_groups().is_empty());
    assert!(selected.operation_index_of(pinned_intermediate).is_some());
    assert!(selected.operation_index_of(terminal).is_some());
}

#[test]
fn bounded_add_sub_identity_respects_step_and_leaf_limits() {
    let mut graph = Graph::default();
    let first = graph.input(vec![1], PcuScalarType::F32).unwrap();
    let second = graph
        .uniform_value(vec![1], TensorScalarValue::F32(0.5))
        .unwrap();
    let third = graph.input(vec![1], PcuScalarType::F32).unwrap();
    let fourth = graph.input(vec![1], PcuScalarType::F32).unwrap();
    let mut accumulator = graph.add(first, second).unwrap();
    accumulator = graph.sub(accumulator, third).unwrap();
    accumulator = graph.add(accumulator, fourth).unwrap();
    let accepted = graph
        .execution_plan_for_outputs(&[accumulator])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );
    let group = &accepted.bounded_pointwise_fusion_groups()[0];
    assert_eq!(group.steps.len(), 3);
    assert_eq!(group.leaves.len(), 4);
    assert_eq!(group.output, accumulator);

    let mut too_many_steps = Graph::default();
    let lhs = too_many_steps.input(vec![1], PcuScalarType::F32).unwrap();
    let rhs = too_many_steps.input(vec![1], PcuScalarType::F32).unwrap();
    let mut accumulator = too_many_steps.add(lhs, rhs).unwrap();
    for _ in 0..8 {
        accumulator = too_many_steps.sub(accumulator, rhs).unwrap();
    }
    let rejected = too_many_steps
        .execution_plan_for_outputs(&[accumulator])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());

    let mut too_many_leaves = Graph::default();
    let leaves = (0..5)
        .map(|_| too_many_leaves.input(vec![1], PcuScalarType::F32).unwrap())
        .collect::<Vec<_>>();
    let mut accumulator = too_many_leaves.add(leaves[0], leaves[1]).unwrap();
    for leaf in leaves.iter().skip(2) {
        accumulator = too_many_leaves.add(accumulator, *leaf).unwrap();
    }
    let rejected = too_many_leaves
        .execution_plan_for_outputs(&[accumulator])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());
}

#[test]
fn bounded_add_sub_relu_respects_step_and_leaf_limits() {
    let mut graph = Graph::default();
    let left = graph.input(vec![1], PcuScalarType::F32).unwrap();
    let right = graph.input(vec![1], PcuScalarType::F32).unwrap();
    let mut accumulator = graph.add(left, right).unwrap();
    for _ in 0..7 {
        accumulator = graph.sub(accumulator, right).unwrap();
    }
    let within_limit = graph.relu(accumulator).unwrap();
    let accepted = graph
        .execution_plan_for_outputs(&[within_limit])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    assert_eq!(accepted.bounded_pointwise_fusion_groups()[0].steps.len(), 8);

    let mut over_limit_graph = Graph::default();
    let first = over_limit_graph.input(vec![1], PcuScalarType::F32).unwrap();
    let second = over_limit_graph.input(vec![1], PcuScalarType::F32).unwrap();
    let mut accumulator = over_limit_graph.add(first, second).unwrap();
    for _ in 0..8 {
        accumulator = over_limit_graph.sub(accumulator, second).unwrap();
    }
    let output = over_limit_graph.relu(accumulator).unwrap();
    let rejected = over_limit_graph
        .execution_plan_for_outputs(&[output])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());

    let mut too_many_leaves_graph = Graph::default();
    let values = (0..5)
        .map(|_| {
            too_many_leaves_graph
                .input(vec![1], PcuScalarType::F32)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let mut accumulator = too_many_leaves_graph.add(values[0], values[1]).unwrap();
    for (index, value) in values.into_iter().skip(2).enumerate() {
        accumulator = if index.is_multiple_of(2) {
            too_many_leaves_graph.add(accumulator, value).unwrap()
        } else {
            too_many_leaves_graph.sub(accumulator, value).unwrap()
        };
    }
    let output = too_many_leaves_graph.relu(accumulator).unwrap();
    let rejected = too_many_leaves_graph
        .execution_plan_for_outputs(&[output])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());
}

#[test]
fn bounded_add_sub_relu_rejects_branches_fanout_and_pins() {
    let mut branched = Graph::default();
    let left = branched.input(vec![2], PcuScalarType::F32).unwrap();
    let right = branched.input(vec![2], PcuScalarType::F32).unwrap();
    let left_branch = branched.add(left, right).unwrap();
    let right_branch = branched.sub(left, right).unwrap();
    let joined = branched.add(left_branch, right_branch).unwrap();
    let output = branched.relu(joined).unwrap();
    let rejected = branched
        .execution_plan_for_outputs(&[output])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());

    let mut fanout = Graph::default();
    let left = fanout.input(vec![2], PcuScalarType::F32).unwrap();
    let right = fanout.input(vec![2], PcuScalarType::F32).unwrap();
    let intermediate = fanout.add(left, right).unwrap();
    let terminal = fanout.sub(intermediate, right).unwrap();
    let extra = fanout.add(intermediate, left).unwrap();
    let output = fanout.relu(terminal).unwrap();
    let rejected = fanout
        .execution_plan_for_outputs(&[output, extra])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());

    let mut pinned = Graph::default();
    let left = pinned.input(vec![2], PcuScalarType::F32).unwrap();
    let right = pinned.input(vec![2], PcuScalarType::F32).unwrap();
    let intermediate = pinned.add(left, right).unwrap();
    let terminal = pinned.sub(intermediate, right).unwrap();
    let output = pinned.relu(terminal).unwrap();
    let rejected = pinned
        .execution_plan_for_outputs(&[intermediate, output])
        .unwrap()
        .select_lowering_with_grouping(
            TensorArithmeticRewritePolicy::Disabled,
            TensorArithmeticCapability::Strict,
            TensorPointwiseGroupingPolicy::BoundedAddSubRelu,
        );
    assert!(rejected.bounded_pointwise_fusion_groups().is_empty());
}

#[test]
fn selected_add_relu_group_rejects_foreign_values_and_shape_mismatches() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2], PcuScalarType::F32).unwrap();
    let wrong_shape = graph.input(vec![1, 2], PcuScalarType::F32).unwrap();
    assert!(matches!(
        graph.add(left, wrong_shape),
        Err(TensorError::ShapeMismatch { .. })
    ));

    let mut other_graph = Graph::default();
    let foreign = other_graph.input(vec![2], PcuScalarType::F32).unwrap();
    assert_eq!(
        graph.execution_plan_for_outputs(&[foreign]).unwrap_err(),
        TensorError::UnknownValue(foreign)
    );
    assert!(matches!(
        graph.add(left, foreign),
        Err(TensorError::UnknownValue(value)) if value == foreign
    ));
}

#[test]
fn empty_graph_has_empty_execution_plan() {
    let graph = Graph::default();
    let plan = graph.execution_plan();
    assert!(plan.node_order().is_empty());
    assert_eq!(plan.peak_live_bytes(), Some(0));
}

#[test]
fn unsupported_assessment_has_no_implicit_fallback() {
    struct RejectAll;
    impl TensorOperationAssessor for RejectAll {
        fn assess_node(&self, _graph: &Graph, _node: NodeDescriptor<'_>) -> TensorOperationSupport {
            TensorOperationSupport::Unsupported {
                reason: TensorUnsupportedReason::Operation,
            }
        }
    }

    let mut graph = Graph::default();
    graph.input(vec![1], PcuScalarType::F32).unwrap();
    let assessment = graph.assess_with(&RejectAll);
    assert!(!assessment.is_supported());
    assert_eq!(assessment.unsupported_nodes().count(), 1);
    assert!(matches!(
        assessment.nodes[0].support,
        TensorOperationSupport::Unsupported {
            reason: TensorUnsupportedReason::Operation
        }
    ));
}

fn build() -> (Graph, ValueId, ValueId, ValueId, ValueId) {
    let mut g = Graph::default();
    let x = g.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let w = g.input(vec![2, 1], PcuScalarType::F32).unwrap();
    let b = g.input(vec![2, 1], PcuScalarType::F32).unwrap();
    let affine = g.matmul(x, w).unwrap();
    let prediction = g.add(affine, b).unwrap();
    let target = g
        .constant_typed::<f32>(Tensor::new(vec![2, 1], vec![1.0, -1.0]).unwrap())
        .erase();
    let loss = g.mean_squared_error(prediction, target).unwrap();
    (g, x, w, b, loss)
}

#[test]
fn linear_regression_value_and_finite_difference_gradients() {
    let (g, x, w, b, loss) = build();
    let xv = Tensor::<f32>::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let wv = Tensor::<f32>::new(vec![2, 1], vec![0.25, -0.5]).unwrap();
    let bv = Tensor::<f32>::new(vec![2, 1], vec![0.1, 0.2]).unwrap();
    let inputs = [
        (x, TensorValue::from(xv.clone())),
        (w, TensorValue::from(wv.clone())),
        (b, TensorValue::from(bv.clone())),
    ];
    let execution = g.evaluate(&inputs).unwrap();
    let analytic = execution.gradients(&g, loss).unwrap()[w.index]
        .as_ref()
        .unwrap()
        .data
        .clone();
    let eps = 1e-3;
    for (index, analytic_gradient) in analytic.iter().enumerate() {
        let mut plus = wv.clone();
        plus.data[index] += eps;
        let mut minus = wv.clone();
        minus.data[index] -= eps;
        let eval_loss = |candidate_w: Tensor| {
            let eval = g
                .evaluate(&[
                    (x, TensorValue::from(xv.clone())),
                    (w, TensorValue::from(candidate_w)),
                    (b, TensorValue::from(bv.clone())),
                ])
                .unwrap();
            eval.value_typed::<f32>(loss).unwrap().data[0]
        };
        let numeric = (eval_loss(plus) - eval_loss(minus)) / (2.0 * eps);
        assert!(
            (analytic_gradient - numeric).abs() < 1e-3,
            "{index}: {analytic_gradient} != {numeric}"
        );
    }
    assert_eq!(execution.value_typed::<f32>(loss).unwrap().shape(), &[]);

    let gradients = execution.gradients(&g, loss).unwrap();
    let learning_rate = 0.05;
    let updated_w = Tensor::new(
        wv.shape.clone(),
        wv.data
            .iter()
            .zip(&gradients[w.index].as_ref().unwrap().data)
            .map(|(value, grad)| value - learning_rate * grad)
            .collect(),
    )
    .unwrap();
    let updated_b = Tensor::new(
        bv.shape.clone(),
        bv.data
            .iter()
            .zip(&gradients[b.index].as_ref().unwrap().data)
            .map(|(value, grad)| value - learning_rate * grad)
            .collect(),
    )
    .unwrap();
    let updated = g
        .evaluate(&[
            (x, TensorValue::from(xv)),
            (w, TensorValue::from(updated_w)),
            (b, TensorValue::from(updated_b)),
        ])
        .unwrap();
    assert!(
        updated.value_typed::<f32>(loss).unwrap().data[0]
            < execution.value_typed::<f32>(loss).unwrap().data[0]
    );
}

#[test]
fn checked_shape_and_operator_shapes() {
    assert_eq!(
        Tensor::<f32>::new(vec![usize::MAX, 2], vec![]),
        Err(TensorError::ShapeOverflow)
    );
    let mut g = Graph::default();
    let a = g.input(vec![2, 3], PcuScalarType::F32).unwrap();
    let b = g.input(vec![4, 2], PcuScalarType::F32).unwrap();
    assert!(matches!(
        g.matmul(a, b),
        Err(TensorError::MatMulShape { .. })
    ));
}

#[test]
fn graph_plan_exposes_borrowed_operations_shapes_and_append_order() {
    let mut graph = Graph::default();
    let input = graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    let weights = graph.input(vec![2, 1], PcuScalarType::F32).unwrap();
    let product = graph.matmul(input, weights).unwrap();
    let activated = graph.relu(product).unwrap();
    let bias = graph
        .constant_typed::<f32>(Tensor::new(vec![2, 1], vec![0.5, -0.5]).unwrap())
        .erase();
    let prediction = graph.add(activated, bias).unwrap();
    let target = graph
        .constant_typed::<f32>(Tensor::new(vec![2, 1], vec![1.0, 0.0]).unwrap())
        .erase();
    let loss = graph.mean_squared_error(prediction, target).unwrap();

    let plan: Vec<_> = graph.nodes().collect();
    assert_eq!(plan.len(), 8);
    assert_eq!(
        plan.iter().map(|node| node.value).collect::<Vec<_>>(),
        [
            input, weights, product, activated, bias, prediction, target, loss
        ]
    );
    assert_eq!(plan[0].op, OpDescriptor::Input);
    assert_eq!(plan[0].shape, [2, 2]);
    assert_eq!(plan[1].shape, [2, 1]);
    assert_eq!(
        plan[2].op,
        OpDescriptor::MatMul {
            left: input,
            right: weights,
            transpose_left: false,
            transpose_right: false,
        }
    );
    assert_eq!(plan[2].shape, [2, 1]);
    assert_eq!(plan[3].op, OpDescriptor::Relu { input: product });
    assert_eq!(
        plan[4].op,
        OpDescriptor::Constant(&TensorValue::from(
            Tensor::<f32>::new(vec![2, 1], vec![0.5, -0.5]).unwrap()
        ))
    );
    assert_eq!(
        plan[5].op,
        OpDescriptor::Add {
            left: activated,
            right: bias
        }
    );
    assert_eq!(plan[6].shape, [2, 1]);
    assert_eq!(
        plan[7].op,
        OpDescriptor::MeanSquaredError { prediction, target }
    );
    assert_eq!(plan[7].shape, []);
}

#[test]
fn graph_plan_value_ids_keep_graph_identity() {
    let mut first = Graph::default();
    let first_id = first.input(vec![1], PcuScalarType::F32).unwrap();
    let mut second = Graph::default();
    let second_id = second.input(vec![1], PcuScalarType::F32).unwrap();

    assert_ne!(first_id, second_id);
    assert_eq!(first.nodes().next().unwrap().value, first_id);
    assert_eq!(second.nodes().next().unwrap().value, second_id);
    assert_eq!(
        second.shape(first_id),
        Err(TensorError::UnknownValue(first_id))
    );
}

#[test]
fn reference_route_requires_explicit_assessor() {
    let (graph, ..) = build();
    let assessment = graph.assess_with(&TensorReferenceAssessor);
    assert!(assessment.is_supported());
    assert_eq!(assessment.nodes.len(), graph.nodes().len());
    assert!(assessment.nodes.iter().all(|node| matches!(
        node.support,
        TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Reference,
            workspace_bytes: None
        }
    )));
}

#[test]
fn composed_loss_propagates_incoming_gradient() {
    let (mut g, x, w, b, loss) = build();
    let doubled_loss = g.add(loss, loss).unwrap();
    let inputs = [
        (
            x,
            TensorValue::from(Tensor::<f32>::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
        ),
        (
            w,
            TensorValue::from(Tensor::<f32>::new(vec![2, 1], vec![0.25, -0.5]).unwrap()),
        ),
        (
            b,
            TensorValue::from(Tensor::<f32>::new(vec![2, 1], vec![0.1, 0.2]).unwrap()),
        ),
    ];
    let execution = g.evaluate(&inputs).unwrap();
    let gradients = execution.gradients(&g, doubled_loss).unwrap();
    let base_gradient = execution.gradients(&g, loss).unwrap();
    for (actual, base) in gradients[w.index]
        .as_ref()
        .unwrap()
        .data
        .iter()
        .zip(&base_gradient[w.index].as_ref().unwrap().data)
    {
        assert!((actual - 2.0 * base).abs() < 1e-6);
    }
}

#[test]
fn execution_and_input_bindings_are_graph_checked() {
    let (mut graph, x, w, b, loss) = build();
    let inputs = [
        (
            x,
            TensorValue::from(Tensor::<f32>::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap()),
        ),
        (
            w,
            TensorValue::from(Tensor::<f32>::new(vec![2, 1], vec![0.25, -0.5]).unwrap()),
        ),
        (
            b,
            TensorValue::from(Tensor::<f32>::new(vec![2, 1], vec![0.1, 0.2]).unwrap()),
        ),
    ];

    let duplicated = [
        inputs[0].clone(),
        inputs[0].clone(),
        inputs[1].clone(),
        inputs[2].clone(),
    ];
    assert_eq!(
        graph.evaluate(&duplicated).unwrap_err(),
        TensorError::DuplicateInput(x)
    );
    let extra = [(loss, TensorValue::from(Tensor::<f32>::scalar(0.0)))];
    assert_eq!(
        graph
            .evaluate(&[
                inputs[0].clone(),
                inputs[1].clone(),
                inputs[2].clone(),
                extra[0].clone()
            ])
            .unwrap_err(),
        TensorError::ExtraInput(loss)
    );

    let execution = graph.evaluate(&inputs).unwrap();
    let mut other_graph = Graph::default();
    let foreign_id = other_graph.input(vec![2, 2], PcuScalarType::F32).unwrap();
    assert_eq!(
        other_graph.shape(x).unwrap_err(),
        TensorError::UnknownValue(x)
    );
    assert_eq!(
        other_graph.add(foreign_id, x).unwrap_err(),
        TensorError::UnknownValue(x)
    );
    let other_graph = Graph::default();
    assert_eq!(
        execution.gradients(&other_graph, loss).unwrap_err(),
        TensorError::WrongGraph
    );
    let new_scalar = graph.constant_typed::<f32>(Tensor::scalar(1.0)).erase();
    let _ = graph.add(loss, new_scalar).unwrap();
    assert_eq!(
        execution.gradients(&graph, loss).unwrap_err(),
        TensorError::GraphChanged
    );
}
