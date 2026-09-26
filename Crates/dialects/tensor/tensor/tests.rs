use super::*;
use fusion_pcu::{
    PcuMemoryAccess,
    PcuMemoryOverlap,
    PcuMemoryRange,
    PcuMemoryResource,
    PcuMemoryPoolId,
    PcuMemoryResourceOrigin,
};

#[test]
#[allow(clippy::cognitive_complexity)] // One test covers schedule selection and its numeric contract.
fn sgd_rewrite_is_opt_in_inspectable_and_does_not_change_reference_rounding() {
    let mut graph = Graph::default();
    let weights = graph.input([1]).unwrap();
    let gradient = graph.input([1]).unwrap();
    let rate_value = 1.0 + 2.0_f32.powi(-23);
    let gradient_value = 1.0 - 2.0_f32.powi(-23);
    let rate = graph.constant(Tensor::new([1], vec![rate_value]).unwrap());
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
    let foreign_value = other_graph.input([1]).unwrap();
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
            (weights, Tensor::new([1], vec![1.0]).unwrap()),
            (gradient, Tensor::new([1], vec![gradient_value]).unwrap()),
        ])
        .unwrap();
    let separate = execution.value(updated).unwrap().data()[0];
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
    let weights = graph.input([1]).unwrap();
    let gradient = graph.input([1]).unwrap();
    let rate = graph.constant(Tensor::new([1], vec![0.25]).unwrap());
    let scaled = graph.mul(rate, gradient).unwrap();
    let intervening = graph.relu(weights).unwrap();
    let updated = graph.sub(intervening, scaled).unwrap();
    let source = graph.execution_plan_for_outputs(&[updated]).unwrap();
    assert!(
        !source
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
    let weights = graph.input([2]).unwrap();
    let gradient = graph.input([2]).unwrap();
    let rate = graph.constant(Tensor::new([2], vec![0.25, 0.5]).unwrap());
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
    let weights = graph.input([2]).unwrap();
    let gradient = graph.input([2]).unwrap();
    let rate = graph.constant(Tensor::new([2], vec![0.25, 0.5]).unwrap());
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
    let weights = graph.input([2]).unwrap();
    let gradient = graph.input([2]).unwrap();
    let rate = graph.constant(Tensor::new([2], vec![0.25, 0.25]).unwrap());
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
    let weights = graph.input([1]).unwrap();
    let gradient = graph.input([1]).unwrap();
    let rate = graph.constant(Tensor::new([1], vec![f32::NAN]).unwrap());
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
    let weights = graph.input([4]).unwrap();
    let gradient = graph.input([4]).unwrap();
    let rate = graph.constant(splat);
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
    let input = graph.input([4]).unwrap();
    let uniform = graph.uniform([4], 2.5).unwrap();
    let sum = graph.add(input, uniform).unwrap();

    let uniform_node = graph.nodes().nth(1).unwrap();
    assert_eq!(uniform_node.value, uniform);
    assert_eq!(uniform_node.shape, &[4]);
    assert_eq!(uniform_node.op, OpDescriptor::Uniform { value: 2.5 });

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
    let uniform = graph.uniform([2, 2], -3.0).unwrap();
    let empty = graph.uniform([2, 0], 7.0).unwrap();
    let scalar = graph.uniform([], 4.0).unwrap();
    assert_eq!(
        graph.uniform([usize::MAX, 2], 0.0),
        Err(TensorError::ShapeOverflow)
    );

    let execution = graph.evaluate(&[]).unwrap();
    assert_eq!(execution.value(uniform).unwrap().shape(), &[2, 2]);
    assert_eq!(execution.value(uniform).unwrap().data(), &[-3.0; 4]);
    assert_eq!(execution.value(empty).unwrap().shape(), &[2, 0]);
    assert!(execution.value(empty).unwrap().data().is_empty());
    assert_eq!(execution.value(scalar).unwrap().shape(), &[]);
    assert_eq!(execution.value(scalar).unwrap().data(), &[4.0]);
}

struct TestMemory(PcuMemoryOverlap);

impl PcuMemoryResource for TestMemory {
    fn pool(&self) -> PcuMemoryPoolId {
        PcuMemoryPoolId(0)
    }
    fn size_bytes(&self) -> u64 {
        64
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

#[test]
fn storage_constraints_follow_liveness_and_reject_unknown_overlap() {
    let mut graph = Graph::default();
    let left = graph.input(vec![4]).unwrap();
    let right = graph.input(vec![4]).unwrap();
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
    let a = TestMemory(PcuMemoryOverlap::Disjoint);
    let b = TestMemory(PcuMemoryOverlap::Unknown);
    assert_eq!(constraint.validate(&[(left, &a), (sum, &a)]), Ok(()));
    assert_eq!(constraint.validate_resources(&a, &a), Ok(()));
    assert_eq!(
        constraint.validate(&[(left, &a), (sum, &b)]),
        Err(TensorStorageValidationError::UnknownOverlap { left, right: sum })
    );
    let overlapping = TestMemory(PcuMemoryOverlap::Overlapping);
    assert!(matches!(
        constraint.validate_resources(&a, &overlapping),
        Err(TensorStorageValidationError::Overlapping { .. })
    ));
    assert!(matches!(
        constraint.validate(&[(left, &a), (sum, &overlapping)]),
        Err(TensorStorageValidationError::Overlapping { .. })
    ));
}

fn assert_gradient_graph_matches_reference(
    graph: &mut Graph,
    inputs: &[(ValueId, Tensor)],
    loss: ValueId,
    values: &[ValueId],
) {
    let gradient_values = graph.backward_mse(loss).unwrap();
    let execution = graph.evaluate(inputs).unwrap();
    let reference = execution.gradients(graph, loss).unwrap();
    for value in values {
        let graph_gradient = execution
            .value(gradient_values[value.index].unwrap())
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
    let samples = graph.input([2, 3]).unwrap();
    let weights = graph.input([3, 2]).unwrap();
    let prediction = graph.matmul(samples, weights).unwrap();
    let squared = graph.mul(prediction, prediction).unwrap();
    let target = graph.input([2, 2]).unwrap();
    let loss = graph.mean_squared_error(squared, target).unwrap();
    let inputs = [
        (
            samples,
            Tensor::new([2, 3], vec![1.0, 2.0, -1.0, 0.5, -2.0, 3.0]).unwrap(),
        ),
        (
            weights,
            Tensor::new([3, 2], vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.2]).unwrap(),
        ),
        (
            target,
            Tensor::new([2, 2], vec![0.0, 1.0, -1.0, 0.5]).unwrap(),
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
    let left = graph.input([3, 2]).unwrap();
    let right = graph.input([4, 3]).unwrap();
    let prediction = graph.matmul_transposed(left, right, true, true).unwrap();
    let target = graph.input([2, 4]).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    let inputs = [
        (
            left,
            Tensor::new([3, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5]).unwrap(),
        ),
        (
            right,
            Tensor::new(
                [4, 3],
                vec![
                    0.2, -0.1, 0.3, 0.4, 0.5, -0.2, 0.1, -0.3, 0.6, 0.7, 0.2, 0.8,
                ],
            )
            .unwrap(),
        ),
        (target, Tensor::new([2, 4], vec![0.0; 8]).unwrap()),
    ];
    assert_gradient_graph_matches_reference(&mut graph, &inputs, loss, &[left, right]);
}

#[test]
fn relu_backward_checks_shapes_and_matches_strict_derivative() {
    let mut graph = Graph::default();
    let input = graph.input([2]).unwrap();
    let upstream = graph.input([2]).unwrap();
    let wrong_shape = graph.constant(Tensor::new([1, 2], vec![0.0, 0.0]).unwrap());
    assert_eq!(
        graph.relu_backward(input, wrong_shape),
        Err(TensorError::ShapeMismatch {
            left: vec![2],
            right: vec![1, 2]
        })
    );
    let derivative = graph.relu_backward(input, upstream).unwrap();
    let inputs = [
        (input, Tensor::new([2], vec![f32::NAN, 0.0]).unwrap()),
        (upstream, Tensor::new([2], vec![3.0, 4.0]).unwrap()),
    ];
    assert_eq!(
        graph
            .evaluate(&inputs)
            .unwrap()
            .value(derivative)
            .unwrap()
            .data(),
        &[0.0, 0.0]
    );
}

#[test]
fn backward_mse_graph_matches_reference_for_two_hidden_relu_layers() {
    let mut graph = Graph::default();
    let samples = graph.input([2, 3]).unwrap();
    let weights1 = graph.input([3, 4]).unwrap();
    let hidden1 = graph.matmul(samples, weights1).unwrap();
    let activated1 = graph.relu(hidden1).unwrap();
    let weights2 = graph.input([4, 4]).unwrap();
    let hidden2 = graph.matmul(activated1, weights2).unwrap();
    let activated2 = graph.relu(hidden2).unwrap();
    let weights3 = graph.input([4, 2]).unwrap();
    let prediction = graph.matmul(activated2, weights3).unwrap();
    let target = graph.input([2, 2]).unwrap();
    let loss = graph.mean_squared_error(prediction, target).unwrap();
    let inputs = [
        (
            samples,
            Tensor::new([2, 3], vec![1.0, -2.0, 0.5, -1.0, 0.25, 2.0]).unwrap(),
        ),
        (
            weights1,
            Tensor::new(
                [3, 4],
                vec![
                    0.2, -0.3, 0.5, 0.1, -0.4, 0.6, 0.2, -0.1, 0.3, 0.2, -0.5, 0.4,
                ],
            )
            .unwrap(),
        ),
        (
            weights2,
            Tensor::new(
                [4, 4],
                vec![
                    0.1, 0.2, -0.3, 0.4, -0.2, 0.5, 0.1, -0.1, 0.3, -0.4, 0.2, 0.6, 0.5, 0.1, -0.2,
                    0.3,
                ],
            )
            .unwrap(),
        ),
        (
            weights3,
            Tensor::new([4, 2], vec![0.2, -0.1, 0.4, 0.3, -0.5, 0.2, 0.1, 0.6]).unwrap(),
        ),
        (
            target,
            Tensor::new([2, 2], vec![0.0, 1.0, -0.5, 0.25]).unwrap(),
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
    let left = graph.input(vec![2, 2]).unwrap();
    let right = graph.input(vec![2, 2]).unwrap();
    let sub = graph.sub(left, right).unwrap();
    let mul = graph.mul(sub, right).unwrap();
    let shape_mismatch = graph.input(vec![2]).unwrap();
    assert!(matches!(
        graph.mul(left, shape_mismatch),
        Err(TensorError::ShapeMismatch { .. })
    ));
    let execution = graph
        .evaluate(&[
            (
                left,
                Tensor::new(vec![2, 2], vec![3.0, 5.0, 7.0, 9.0]).unwrap(),
            ),
            (
                right,
                Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ),
            (
                shape_mismatch,
                Tensor::new(vec![2], vec![0.0, 0.0]).unwrap(),
            ),
        ])
        .unwrap();
    assert_eq!(execution.value(sub).unwrap().data(), &[2.0, 3.0, 4.0, 5.0]);
    assert_eq!(
        execution.value(mul).unwrap().data(),
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
    let weights = graph.input([3]).unwrap();
    let gradient = graph.input([3]).unwrap();
    let updated = graph.sgd_update(weights, gradient, 0.25).unwrap();
    let wrong_shape = graph.input([2]).unwrap();
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
            (weights, Tensor::new([3], vec![1.0, 2.0, -3.0]).unwrap()),
            (gradient, Tensor::new([3], vec![0.4, -2.0, 8.0]).unwrap()),
            (wrong_shape, Tensor::new([2], vec![0.0, 0.0]).unwrap()),
        ])
        .unwrap();
    for (actual, expected) in execution
        .value(updated)
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
    let left = graph.input(vec![3, 2]).unwrap();
    let right = graph.input(vec![3, 4]).unwrap();
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
                Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            ),
            (
                right,
                Tensor::new(
                    vec![3, 4],
                    vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                )
                .unwrap(),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution.value(product).unwrap().data(),
        &[1.0, 3.0, 5.0, 5.0, 2.0, 4.0, 6.0, 6.0]
    );
    let mut scalar_graph = Graph::default();
    let x = scalar_graph.input(vec![3, 2]).unwrap();
    let y = scalar_graph.input(vec![3, 1]).unwrap();
    let transposed = scalar_graph.matmul_transposed(x, y, true, false).unwrap();
    let target = scalar_graph.constant(Tensor::new(vec![2, 1], vec![0.0, 0.0]).unwrap());
    let loss = scalar_graph.mean_squared_error(transposed, target).unwrap();
    let execution = scalar_graph
        .evaluate(&[
            (
                x,
                Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
            ),
            (y, Tensor::new(vec![3, 1], vec![1.0, 1.0, 1.0]).unwrap()),
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
    let input = graph.input(vec![2, 2]).unwrap();
    let constant = graph.constant(Tensor::new(vec![2, 2], vec![1.0; 4]).unwrap());
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
    let input = graph.input(vec![2]).unwrap();
    let constant = graph.constant(Tensor::new(vec![2], vec![1.0, 1.0]).unwrap());
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

    let inputs = [(input, Tensor::new(vec![2], vec![-1.0, 2.0]).unwrap())];
    let planned = plan.execute_reference(&inputs).unwrap();
    let ordinary = graph.evaluate(&inputs).unwrap();
    assert_eq!(planned.value(output).unwrap().data(), &[0.0, 6.0]);
    assert_eq!(
        planned.value(output).unwrap(),
        ordinary.value(output).unwrap()
    );
}

#[test]
fn selected_output_plan_prunes_unrelated_nodes_and_pins_outputs() {
    let mut graph = Graph::default();
    let input = graph.input(vec![2]).unwrap();
    let constant = graph.constant(Tensor::new(vec![2], vec![1.0, 1.0]).unwrap());
    let selected = graph.add(input, constant).unwrap();
    let unrelated = graph.input(vec![2]).unwrap();
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
    let left = graph.input(vec![2, 2]).unwrap();
    let right = graph.input(vec![2, 2]).unwrap();
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
            Tensor::new(vec![2, 2], vec![-2.0, 1.0, 0.0, 3.0]).unwrap(),
        ),
        (
            right,
            Tensor::new(vec![2, 2], vec![1.0, 2.0, -0.0, -4.0]).unwrap(),
        ),
    ];
    assert_eq!(
        graph.evaluate(&inputs).unwrap().value(relu).unwrap().data(),
        &[0.0, 3.0, 0.0, 0.0]
    );
}

#[test]
fn selected_add_relu_group_respects_fanout_and_output_pins() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2]).unwrap();
    let right = graph.input(vec![2]).unwrap();
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
    let value = graph.input(vec![3]).unwrap();
    let scale = graph.uniform(vec![3], 0.5).unwrap();
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
    let a = graph.input(vec![2]).unwrap();
    let b = graph.input(vec![2]).unwrap();
    let c = graph.input(vec![2]).unwrap();
    let d = graph.input(vec![2]).unwrap();
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
    let mut value = too_many_steps.input(vec![1]).unwrap();
    let factor = too_many_steps.input(vec![1]).unwrap();
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
        .map(|_| too_many_leaves.input(vec![1]).unwrap())
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
    let left = graph.input(vec![4]).unwrap();
    let right = graph.input(vec![4]).unwrap();
    let bias = graph.input(vec![4]).unwrap();
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
    let value = graph.input(vec![3]).unwrap();
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
    let left = graph.input(vec![4]).unwrap();
    let uniform = graph.uniform(vec![4], 0.25).unwrap();
    let bias = graph.input(vec![4]).unwrap();
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
fn bounded_add_sub_identity_keeps_requested_terminal_for_pin_and_fanout() {
    let mut graph = Graph::default();
    let left = graph.input(vec![2]).unwrap();
    let right = graph.input(vec![2]).unwrap();
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
    let left = pinned_graph.input(vec![2]).unwrap();
    let right = pinned_graph.input(vec![2]).unwrap();
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
    let first = graph.input(vec![1]).unwrap();
    let second = graph.uniform(vec![1], 0.5).unwrap();
    let third = graph.input(vec![1]).unwrap();
    let fourth = graph.input(vec![1]).unwrap();
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
    let lhs = too_many_steps.input(vec![1]).unwrap();
    let rhs = too_many_steps.input(vec![1]).unwrap();
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
        .map(|_| too_many_leaves.input(vec![1]).unwrap())
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
    let left = graph.input(vec![1]).unwrap();
    let right = graph.input(vec![1]).unwrap();
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
    let first = over_limit_graph.input(vec![1]).unwrap();
    let second = over_limit_graph.input(vec![1]).unwrap();
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
        .map(|_| too_many_leaves_graph.input(vec![1]).unwrap())
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
    let left = branched.input(vec![2]).unwrap();
    let right = branched.input(vec![2]).unwrap();
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
    let left = fanout.input(vec![2]).unwrap();
    let right = fanout.input(vec![2]).unwrap();
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
    let left = pinned.input(vec![2]).unwrap();
    let right = pinned.input(vec![2]).unwrap();
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
    let left = graph.input(vec![2]).unwrap();
    let wrong_shape = graph.input(vec![1, 2]).unwrap();
    assert!(matches!(
        graph.add(left, wrong_shape),
        Err(TensorError::ShapeMismatch { .. })
    ));

    let mut other_graph = Graph::default();
    let foreign = other_graph.input(vec![2]).unwrap();
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
    graph.input(vec![1]).unwrap();
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
    let x = g.input(vec![2, 2]).unwrap();
    let w = g.input(vec![2, 1]).unwrap();
    let b = g.input(vec![2, 1]).unwrap();
    let affine = g.matmul(x, w).unwrap();
    let prediction = g.add(affine, b).unwrap();
    let target = g.constant(Tensor::new(vec![2, 1], vec![1.0, -1.0]).unwrap());
    let loss = g.mean_squared_error(prediction, target).unwrap();
    (g, x, w, b, loss)
}

#[test]
fn linear_regression_value_and_finite_difference_gradients() {
    let (g, x, w, b, loss) = build();
    let xv = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let wv = Tensor::new(vec![2, 1], vec![0.25, -0.5]).unwrap();
    let bv = Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap();
    let inputs = [(x, xv.clone()), (w, wv.clone()), (b, bv.clone())];
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
                .evaluate(&[(x, xv.clone()), (w, candidate_w), (b, bv.clone())])
                .unwrap();
            eval.value(loss).unwrap().data[0]
        };
        let numeric = (eval_loss(plus) - eval_loss(minus)) / (2.0 * eps);
        assert!(
            (analytic_gradient - numeric).abs() < 1e-3,
            "{index}: {analytic_gradient} != {numeric}"
        );
    }
    assert_eq!(execution.value(loss).unwrap().shape(), &[]);

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
        .evaluate(&[(x, xv), (w, updated_w), (b, updated_b)])
        .unwrap();
    assert!(updated.value(loss).unwrap().data[0] < execution.value(loss).unwrap().data[0]);
}

#[test]
fn checked_shape_and_operator_shapes() {
    assert_eq!(
        Tensor::new(vec![usize::MAX, 2], vec![]),
        Err(TensorError::ShapeOverflow)
    );
    let mut g = Graph::default();
    let a = g.input(vec![2, 3]).unwrap();
    let b = g.input(vec![4, 2]).unwrap();
    assert!(matches!(
        g.matmul(a, b),
        Err(TensorError::MatMulShape { .. })
    ));
}

#[test]
fn graph_plan_exposes_borrowed_operations_shapes_and_append_order() {
    let mut graph = Graph::default();
    let input = graph.input(vec![2, 2]).unwrap();
    let weights = graph.input(vec![2, 1]).unwrap();
    let product = graph.matmul(input, weights).unwrap();
    let activated = graph.relu(product).unwrap();
    let bias = graph.constant(Tensor::new(vec![2, 1], vec![0.5, -0.5]).unwrap());
    let prediction = graph.add(activated, bias).unwrap();
    let target = graph.constant(Tensor::new(vec![2, 1], vec![1.0, 0.0]).unwrap());
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
        OpDescriptor::Constant(&Tensor::new(vec![2, 1], vec![0.5, -0.5]).unwrap())
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
    let first_id = first.input(vec![1]).unwrap();
    let mut second = Graph::default();
    let second_id = second.input(vec![1]).unwrap();

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
            Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ),
        (w, Tensor::new(vec![2, 1], vec![0.25, -0.5]).unwrap()),
        (b, Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap()),
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
            Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
        ),
        (w, Tensor::new(vec![2, 1], vec![0.25, -0.5]).unwrap()),
        (b, Tensor::new(vec![2, 1], vec![0.1, 0.2]).unwrap()),
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
    let extra = [(loss, Tensor::scalar(0.0))];
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
    let foreign_id = other_graph.input(vec![2, 2]).unwrap();
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
    let new_scalar = graph.constant(Tensor::scalar(1.0));
    let _ = graph.add(loss, new_scalar).unwrap();
    assert_eq!(
        execution.gradients(&graph, loss).unwrap_err(),
        TensorError::GraphChanged
    );
}
