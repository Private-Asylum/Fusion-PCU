//! Source differentiation uses authoritative graph identities and forward policies.
#[rustfmt::skip]
use super::super::{
    PcuExecutionError,
    PcuTensorGraphCapture,
};
#[rustfmt::skip]
use crate::{
    global::PcuSourceShape,
    PcuNumericalMode,
};

const fn shape(rows: usize, columns: usize) -> PcuSourceShape {
    PcuSourceShape::FixedMatrix { rows, columns }
}

#[crate::pcu(crate_path = crate, flag(strict))]
fn train<T: crate::PcuScalar, const R: usize, const K: usize>(
    samples: &[[T; K]; R],
    weights: &[[T; 1]; K],
    target: &[[T; 1]; R],
) -> Result<crate::PcuTensor<T>, PcuExecutionError> {
    let prediction = pcu::matmul(samples, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    let gradient = pcu::gradient(&loss, weights);
    pcu::sgd_update(weights, &gradient, 0.25_f32)
}

#[test]
fn actual_annotated_training_captures_backward_and_update_as_tensor_ir() {
    let (mut capture, inputs) =
        PcuTensorGraphCapture::new::<f32, 3>([shape(2, 2), shape(2, 1), shape(2, 1)]).unwrap();
    let output = train::__pcu_capture_entry::<f32, 2, 2>(&mut capture, inputs).unwrap();
    assert_eq!(capture.gradient_sets.len(), 1);
    let descriptor = capture.graph.node(output.value.erase()).unwrap();
    assert!(matches!(
        descriptor.op,
        crate::dialect::tensor::OpDescriptor::SgdUpdate { .. }
    ));
    let (graph, output_id) = capture.finish(output).unwrap();
    let input_ids = inputs.map(|input| input.value.erase());
    let execution = graph
        .evaluate_checked(&[
            (
                input_ids[0],
                crate::dialect::tensor::TensorValue::F32(
                    crate::dialect::tensor::Tensor::new([2, 2], alloc::vec![1.0, 0.0, 0.0, 1.0])
                        .unwrap(),
                ),
            ),
            (
                input_ids[1],
                crate::dialect::tensor::TensorValue::F32(
                    crate::dialect::tensor::Tensor::new([2, 1], alloc::vec![2.0, -1.0]).unwrap(),
                ),
            ),
            (
                input_ids[2],
                crate::dialect::tensor::TensorValue::F32(
                    crate::dialect::tensor::Tensor::new([2, 1], alloc::vec![1.0, 1.0]).unwrap(),
                ),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution.value_typed::<f32>(output_id).unwrap().data(),
        &[1.75, -0.5]
    );
}

#[cfg(feature = "cpu")]
#[test]
fn ordinary_training_calls_keep_borrowed_weights_and_escaped_updates_live() {
    let _guard = crate::global::policy::TEST_LOCK.lock().unwrap();
    crate::global::configure(crate::global::PcuExecutionPolicy {
        backend: crate::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let samples = [[1.0_f32, 0.0], [0.0, 1.0]];
    let weights = [[2.0_f32], [-1.0]];
    let target = [[1.0_f32], [1.0]];
    let first = train(&samples, &weights, &target).unwrap();
    let second = train::<f32, 2, 2>(&samples, &first, &target).unwrap();
    crate::global::clear_thread_cache().unwrap();
    let mut stack = [99.0_f32; 3];
    first.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.75_f32, -0.5, 99.0].map(f32::to_bits)
    );
    second.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.5625_f32, -0.125, 99.0].map(f32::to_bits)
    );
    drop(first);
    second.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f32::to_bits),
        [1.5625_f32, -0.125, 99.0].map(f32::to_bits)
    );
    crate::global::use_defaults().unwrap();
}

#[test]
fn repeated_derivative_reuses_nodes_and_new_targets_keep_forward_policy() {
    let (mut capture, [input, weights, target]) =
        PcuTensorGraphCapture::new::<f64, 3>([shape(2, 2), shape(2, 1), shape(2, 1)]).unwrap();
    capture.numerical_mode.set(PcuNumericalMode::Strict);
    let predicted = capture.matmul(input, weights).unwrap();
    let loss = capture.mean_squared_error(predicted, target).unwrap();
    capture.numerical_mode.set(PcuNumericalMode::Boundary);
    let weight_gradient = capture.gradient(loss, weights).unwrap();
    let node_count = capture.graph.nodes().len();
    let repeated = capture.gradient(loss, weights).unwrap();
    assert_eq!(repeated.value.erase(), weight_gradient.value.erase());
    assert_eq!(capture.graph.nodes().len(), node_count);
    let input_gradient = capture.gradient(loss, input).unwrap();
    assert!(capture.graph.nodes().len() > node_count);
    assert_eq!(capture.gradient_sets.len(), 1);
    assert_eq!(
        capture.graph.shape(weight_gradient.value.erase()).unwrap(),
        [2, 1]
    );
    assert_eq!(
        capture.graph.shape(input_gradient.value.erase()).unwrap(),
        [2, 2]
    );
    assert_eq!(
        capture
            .graph
            .node(weight_gradient.value.erase())
            .unwrap()
            .numerical_mode,
        Some(PcuNumericalMode::Strict),
    );
    assert_eq!(capture.numerical_mode.get(), PcuNumericalMode::Boundary);
}

#[test]
fn foreign_loss_or_target_rejects_before_graph_changes() {
    let (mut capture, [prediction, target]) =
        PcuTensorGraphCapture::new::<f32, 2>([shape(1, 1); 2]).unwrap();
    let loss = capture.mean_squared_error(prediction, target).unwrap();
    let (mut foreign, [foreign_input, foreign_target]) =
        PcuTensorGraphCapture::new::<f32, 2>([shape(1, 1); 2]).unwrap();
    let foreign_loss = foreign
        .mean_squared_error(foreign_input, foreign_target)
        .unwrap();
    let before = capture.graph.nodes().len();
    for (root, wrt) in [(foreign_loss, target), (loss, foreign_input)] {
        assert!(matches!(
            capture.gradient(root, wrt),
            Err(PcuExecutionError::InvalidTensorSourcePlan)
        ));
        assert_eq!(capture.graph.nodes().len(), before);
        assert!(capture.gradient_sets.is_empty());
    }
}

#[test]
fn non_mse_and_unsupported_division_reject_without_fabricating_derivatives() {
    let (mut capture, [prediction, target]) =
        PcuTensorGraphCapture::new::<f64, 2>([shape(1, 1); 2]).unwrap();
    let before = capture.graph.nodes().len();
    assert!(matches!(
        capture.gradient(prediction, target),
        Err(PcuExecutionError::TensorBuild(
            crate::dialect::tensor::TensorError::GradientRootNotMse(_)
        ))
    ));
    assert_eq!(capture.graph.nodes().len(), before);
    let divided = capture.div(prediction, target).unwrap();
    let loss = capture.mean_squared_error(divided, target).unwrap();
    let before = capture.graph.nodes().len();
    assert!(matches!(
        capture.gradient(loss, prediction),
        Err(PcuExecutionError::TensorBuild(
            crate::dialect::tensor::TensorError::UnsupportedGradient(_)
        ))
    ));
    assert_eq!(capture.graph.nodes().len(), before);
    assert!(capture.gradient_sets.is_empty());
}

#[test]
fn disconnected_target_is_an_error_instead_of_an_invented_zero() {
    let (mut capture, [prediction, target, disconnected]) =
        PcuTensorGraphCapture::new::<f32, 3>([shape(1, 1); 3]).unwrap();
    let loss = capture.mean_squared_error(prediction, target).unwrap();
    let before = capture.graph.nodes().len();
    assert!(matches!(
        capture.gradient(loss, disconnected),
        Err(PcuExecutionError::TensorBuild(
            crate::dialect::tensor::TensorError::UnsupportedGradient(_)
        ))
    ));
    assert_eq!(capture.graph.nodes().len(), before);
    assert!(capture.gradient_sets.is_empty());
    // A later connected request captures an actual reverse graph, never a host zero.
    assert!(capture.gradient(loss, prediction).is_ok());
    assert_eq!(capture.gradient_sets.len(), 1);
}

#[cfg(feature = "cpu")]
#[test]
fn same_generic_source_trains_f64_without_narrowing_to_f32() {
    let _guard = crate::global::policy::TEST_LOCK.lock().unwrap();
    crate::global::configure(crate::global::PcuExecutionPolicy {
        backend: crate::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    // Values below are exact binary fractions whose low bits disappear in F32.
    let low = 2.0_f64.powi(-40);
    let samples = [[1.0_f64, 0.0], [0.0, 1.0]];
    let weights = [[2.0_f64 + low], [-1.0 - low]];
    let target = [[1.0_f64], [1.0]];
    let updated = train(&samples, &weights, &target).unwrap();
    crate::global::clear_thread_cache().unwrap();
    let mut stack = [99.0_f64; 3];
    updated.read_into(&mut stack).unwrap();
    assert_eq!(
        stack.map(f64::to_bits),
        [1.75_f64 + low * 0.75, -0.5 - low * 0.75, 99.0].map(f64::to_bits)
    );
    crate::global::use_defaults().unwrap();
}

#[crate::pcu(crate_path = crate, flag(strict))]
fn weight_only(
    input: &[f32; 1],
    weights: &[f32; 1],
    target: &[f32; 1],
) -> Result<crate::PcuTensor<f32>, PcuExecutionError> {
    let prediction = pcu::mul(input, weights);
    let loss = pcu::mean_squared_error(&prediction, target);
    pcu::gradient(&loss, weights)
}

#[cfg(feature = "cpu")]
#[test]
fn ordinary_requested_gradient_does_not_compute_overflowing_unrequested_derivative() {
    let _guard = crate::global::policy::TEST_LOCK.lock().unwrap();
    crate::global::configure(crate::global::PcuExecutionPolicy {
        backend: crate::global::PcuBackendChoice::Cpu,
        ..Default::default()
    })
    .unwrap();
    let gradient = weight_only(&[0.0], &[f32::MAX], &[1.0]).unwrap();
    crate::global::clear_thread_cache().unwrap();
    let mut stack = [99.0_f32; 2];
    gradient.read_into(&mut stack).unwrap();
    // dL/dweight = -2 * 0; dL/dinput = -2 * MAX was never requested.
    assert_eq!(
        stack.map(f32::to_bits),
        [(-0.0_f32).to_bits(), 99.0_f32.to_bits()]
    );
    crate::global::use_defaults().unwrap();
}

#[crate::pcu(crate_path = crate, flag(strict))]
#[allow(clippy::type_complexity)] // The frontend must see the three explicit owner roles.
fn train_two(
    input: &[f64; 1],
    first: &[f64; 1],
    second: &[f64; 1],
    target: &[f64; 1],
) -> Result<
    (
        crate::PcuTensor<f64>,
        crate::PcuTensor<f64>,
        crate::PcuTensor<f64>,
    ),
    PcuExecutionError,
> {
    let hidden = pcu::mul(input, first);
    let prediction = pcu::mul(&hidden, second);
    let loss = pcu::mean_squared_error(&prediction, target);
    let (second_gradient, first_gradient) = pcu::gradients(&loss, (second, first));
    let updated_first = pcu::sgd_update(first, &first_gradient, 0.001_953_125_f32);
    let updated_second = pcu::sgd_update(second, &second_gradient, 0.001_953_125_f32);
    (updated_first, updated_second, loss)
}

#[test]
fn authentic_tuple_training_captures_one_shared_reverse_pass() {
    let (mut capture, inputs) =
        PcuTensorGraphCapture::new::<f64, 4>([PcuSourceShape::FixedArray { length: 1 }; 4])
            .unwrap();
    let outputs = train_two::__pcu_capture_entry(&mut capture, inputs).unwrap();
    assert_eq!(capture.gradient_sets.len(), 1);
    // Four inputs, two forward multiplies, MSE, six shared reverse nodes,
    // then the two requested weight updates. No derivative of the sample.
    assert_eq!(capture.graph.nodes().len(), 15);
    let before = capture.graph.nodes().len();
    let loss = outputs[2];
    let gradients = capture.gradients(loss, [inputs[2], inputs[1]]).unwrap();
    assert_eq!(capture.graph.nodes().len(), before);
    assert_eq!(gradients[0].capture_id, capture.capture_id);
    assert_eq!(gradients[1].capture_id, capture.capture_id);
    let (graph, outputs) = capture.finish_outputs(outputs).unwrap();
    let execution = graph
        .evaluate_checked(&[
            (
                inputs[0].value.erase(),
                crate::dialect::tensor::TensorValue::F64(
                    crate::dialect::tensor::Tensor::new([1], alloc::vec![2.0]).unwrap(),
                ),
            ),
            (
                inputs[1].value.erase(),
                crate::dialect::tensor::TensorValue::F64(
                    crate::dialect::tensor::Tensor::new([1], alloc::vec![3.0]).unwrap(),
                ),
            ),
            (
                inputs[2].value.erase(),
                crate::dialect::tensor::TensorValue::F64(
                    crate::dialect::tensor::Tensor::new([1], alloc::vec![4.0]).unwrap(),
                ),
            ),
            (
                inputs[3].value.erase(),
                crate::dialect::tensor::TensorValue::F64(
                    crate::dialect::tensor::Tensor::new([1], alloc::vec![1.0]).unwrap(),
                ),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution.value_typed::<f64>(outputs[0]).unwrap().data(),
        &[2.28125]
    );
    assert_eq!(
        execution.value_typed::<f64>(outputs[1]).unwrap().data(),
        &[3.460_937_5]
    );
    assert_eq!(
        execution.value_typed::<f64>(outputs[2]).unwrap().data(),
        &[529.0]
    );
}

#[test]
fn grouped_capture_preflights_all_targets_without_populating_cache() {
    let (mut capture, [prediction, target, disconnected]) =
        PcuTensorGraphCapture::new::<f32, 3>([shape(1, 1); 3]).unwrap();
    let loss = capture.mean_squared_error(prediction, target).unwrap();
    let (_, [foreign]) = PcuTensorGraphCapture::new::<f32, 1>([shape(1, 1)]).unwrap();
    let before = capture.graph.nodes().len();
    for targets in [[prediction, foreign], [prediction, disconnected]] {
        assert!(capture.gradients(loss, targets).is_err());
        assert_eq!(capture.graph.nodes().len(), before);
        assert!(capture.gradient_sets.is_empty());
    }
    assert!(capture.gradients(loss, [prediction, target]).is_ok());
    assert_eq!(capture.gradient_sets.len(), 1);
}
