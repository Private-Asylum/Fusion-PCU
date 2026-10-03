#[rustfmt::skip]
use super::{
    Graph,
    PcuNumericalMode,
    TensorError,
    backward_mse,
    with_source_scope,
};
#[rustfmt::skip]
use crate::{
    PcuCompoundArithmeticPolicy,
    PcuFloatUnderflowPolicy,
    PcuNumericalOptions,
    PcuPrecisionPolicy,
    PcuReproducibility,
    dialect::tensor::{OpDescriptor, Tensor, TensorValue},
};
use alloc::vec;

fn unrelated_defaults() -> PcuNumericalOptions {
    PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        precision: PcuPrecisionPolicy::BackendOptimized,
        reproducibility: PcuReproducibility::PortableV1,
    }
}

#[test]
fn reverse_graph_keeps_forward_contract_after_defaults_change() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let input = graph.input_typed::<f64>([2, 2]).unwrap();
    let weights = graph.input_typed::<f64>([2, 1]).unwrap();
    let target = graph.input_typed::<f64>([2, 1]).unwrap();
    let projected = graph.matmul_typed(input, weights).unwrap();
    let prediction = graph.relu_typed(projected).unwrap();
    let loss = graph.mean_squared_error_typed(prediction, target).unwrap();
    let original_len = graph.nodes.len();
    graph.set_numerical_mode(PcuNumericalMode::Boundary);
    graph.set_numerical_options(unrelated_defaults());
    let gradients = backward_mse(&mut graph, loss.erase()).unwrap();
    assert_eq!(graph.numerical_mode(), PcuNumericalMode::Boundary);
    assert_eq!(graph.numerical_options(), unrelated_defaults());
    for node in graph.nodes().skip(original_len) {
        assert_eq!(node.numerical_options, PcuNumericalOptions::default());
        if matches!(node.op, OpDescriptor::MatMul { .. }) {
            assert_eq!(node.numerical_mode, Some(PcuNumericalMode::Strict));
        }
    }
    let execution = graph
        .evaluate_checked(&[
            (
                input.erase(),
                TensorValue::F64(Tensor::new([2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap()),
            ),
            (
                weights.erase(),
                TensorValue::F64(Tensor::new([2, 1], vec![2.0, -1.0]).unwrap()),
            ),
            (
                target.erase(),
                TensorValue::F64(Tensor::new([2, 1], vec![1.0, 1.0]).unwrap()),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution
            .value_typed::<f64>(gradients[weights.erase().index].unwrap())
            .unwrap()
            .data(),
        &[1.0, 0.0]
    );
}

#[test]
fn each_derivative_uses_its_own_origin_not_the_loss_scope() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f32>([2, 2]).unwrap();
    let weights = graph.input_typed::<f32>([2, 1]).unwrap();
    let target = graph.input_typed::<f32>([2, 1]).unwrap();
    let native_options = PcuNumericalOptions {
        compound_arithmetic: PcuCompoundArithmeticPolicy::BackendDefined,
        ..PcuNumericalOptions::default()
    };
    graph.set_numerical_options(native_options);
    let projected = graph.matmul_typed(input, weights).unwrap();
    graph
        .set_value_float_underflow_policy(
            projected.erase(),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        )
        .unwrap();
    graph.set_numerical_options(PcuNumericalOptions::default());
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let loss = graph.mean_squared_error_typed(projected, target).unwrap();
    graph
        .set_value_float_underflow_policy(
            loss.erase(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    let original_len = graph.nodes.len();
    backward_mse(&mut graph, loss.erase()).unwrap();
    let mut matrix_derivatives = 0;
    let mut loss_derivatives = 0;
    for node in graph.nodes().skip(original_len) {
        match node.op {
            OpDescriptor::MatMul { .. } => {
                matrix_derivatives += 1;
                assert_eq!(node.numerical_options, native_options);
                assert_eq!(node.numerical_mode, Some(PcuNumericalMode::Boundary));
                assert_eq!(
                    node.float_underflow_policy,
                    Some(PcuFloatUnderflowPolicy::AllowGradualUnderflow)
                );
            }
            OpDescriptor::Sub { .. } | OpDescriptor::Mul { .. } => {
                loss_derivatives += 1;
                assert_eq!(node.numerical_options, PcuNumericalOptions::default());
                assert_eq!(
                    node.float_underflow_policy,
                    Some(PcuFloatUnderflowPolicy::RejectSubnormalResult)
                );
            }
            _ => {}
        }
    }
    assert_eq!(matrix_derivatives, 2);
    assert_eq!(loss_derivatives, 4);
    assert_eq!(graph.numerical_mode(), PcuNumericalMode::Strict);
    assert_eq!(graph.numerical_options(), PcuNumericalOptions::default());
}

#[test]
fn failed_append_restores_defaults_and_preserves_existing_contracts() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f64>([1]).unwrap();
    let source = graph.mul_typed(input, input).unwrap();
    let foreign = Graph::default().input_typed::<f64>([1]).unwrap();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    graph.set_numerical_options(unrelated_defaults());
    let original_len = graph.nodes.len();
    let result = with_source_scope(&mut graph, source.erase().index, |graph| {
        graph.mul_typed(input, foreign)
    });
    assert_eq!(result, Err(TensorError::UnknownValue(foreign.erase())));
    assert_eq!(graph.nodes.len(), original_len);
    assert_eq!(graph.numerical_mode(), PcuNumericalMode::Strict);
    assert_eq!(graph.numerical_options(), unrelated_defaults());
    assert_eq!(
        graph.node(source.erase()).unwrap().numerical_options,
        PcuNumericalOptions::default()
    );
}

#[test]
fn unrequested_input_gradient_can_fault_while_weight_derivative_is_valid() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let input = graph.input_typed::<f32>([1]).unwrap();
    let weight = graph.input_typed::<f32>([1]).unwrap();
    let target = graph.input_typed::<f32>([1]).unwrap();
    let prediction = graph.mul_typed(input, weight).unwrap();
    let loss = graph.mean_squared_error_typed(prediction, target).unwrap();
    let gradients = graph.backward_mse(loss.erase()).unwrap();
    let execution = graph.evaluate_checked(&[
        (
            input.erase(),
            TensorValue::F32(Tensor::new([1], vec![0.0]).unwrap()),
        ),
        (
            weight.erase(),
            TensorValue::F32(Tensor::new([1], vec![f32::MAX]).unwrap()),
        ),
        (
            target.erase(),
            TensorValue::F32(Tensor::new([1], vec![1.0]).unwrap()),
        ),
    ]);
    // dL/dweight = -2 * 0 is valid. The all-target transform also computes
    // dL/dinput = -2 * MAX, which is a checked overflow even though not requested.
    assert!(
        matches!(execution, Err(TensorError::ArithmeticFault { value, kind: crate::PcuExecutionFaultKind::ArithmeticOverflow, .. }) if Some(value) == gradients[input.erase().index])
    );
}

#[test]
fn selected_weight_derivative_omits_fatal_branch_without_erasing_forward_effects() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let input = graph.input_typed::<f32>([1]).unwrap();
    let weight = graph.input_typed::<f32>([1]).unwrap();
    let target = graph.input_typed::<f32>([1]).unwrap();
    let prediction = graph.mul_typed(input, weight).unwrap();
    let loss = graph.mean_squared_error_typed(prediction, target).unwrap();
    let gradient = graph
        .backward_mse_for(loss.erase(), weight.erase())
        .unwrap();
    let inputs = [
        (
            input.erase(),
            TensorValue::F32(Tensor::new([1], vec![0.0]).unwrap()),
        ),
        (
            weight.erase(),
            TensorValue::F32(Tensor::new([1], vec![f32::MAX]).unwrap()),
        ),
        (
            target.erase(),
            TensorValue::F32(Tensor::new([1], vec![1.0]).unwrap()),
        ),
    ];
    let execution = graph.evaluate_checked(&inputs).unwrap();
    assert_eq!(
        execution.value_typed::<f32>(gradient).unwrap().data()[0].to_bits(),
        (-0.0_f32).to_bits()
    );
    let one = graph.uniform_typed([1], 1.0_f32).unwrap();
    let forward_fault = graph.div_typed(one, input).unwrap();
    assert!(
        matches!(graph.evaluate_checked(&inputs), Err(TensorError::ArithmeticFault { value, kind: crate::PcuExecutionFaultKind::DivideByZero, .. }) if value == forward_fault.erase())
    );
}

#[test]
fn selected_foreign_disconnected_and_unsupported_targets_never_mutate_graph() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<f64>([1]).unwrap();
    let target = graph.input_typed::<f64>([1]).unwrap();
    let isolated = graph.input_typed::<f64>([1]).unwrap();
    let foreign = Graph::default().input_typed::<f64>([1]).unwrap();
    let loss = graph.mean_squared_error_typed(input, target).unwrap();
    let before = graph.nodes().len();
    assert_eq!(
        graph.backward_mse_for(loss.erase(), foreign.erase()),
        Err(TensorError::UnknownValue(foreign.erase()))
    );
    assert_eq!(
        graph.backward_mse_for(loss.erase(), isolated.erase()),
        Err(TensorError::UnsupportedGradient(isolated.erase()))
    );
    assert_eq!(graph.nodes().len(), before);
    let divided = graph.div_typed(input, target).unwrap();
    let loss = graph.mean_squared_error_typed(divided, target).unwrap();
    let before = graph.nodes().len();
    assert_eq!(
        graph.backward_mse_for(loss.erase(), input.erase()),
        Err(TensorError::UnsupportedGradient(divided.erase()))
    );
    assert_eq!(graph.nodes().len(), before);
}

#[test]
fn selected_mse_derivative_does_not_materialize_an_unused_root_seed() {
    // A requested prediction derivative consumes only the typed 2/n scale.
    // A separate root Uniform(1) has no semantic consumer or storage lifetime.
    macro_rules! verify {
        ($scalar:ty, $variant:ident) => {{
            let mut graph = Graph::default();
            graph.set_numerical_mode(PcuNumericalMode::Strict);
            let prediction = graph.input_typed::<$scalar>([2]).unwrap();
            let target = graph.input_typed::<$scalar>([2]).unwrap();
            let loss = graph.mean_squared_error_typed(prediction, target).unwrap();
            let original_len = graph.nodes().len();
            let derivative = graph
                .backward_mse_for(loss.erase(), prediction.erase())
                .unwrap();
            let appended: alloc::vec::Vec<_> = graph.nodes().skip(original_len).collect();
            assert_eq!(appended.len(), 3);
            assert_eq!(appended[0].shape, &[2]);
            let scale: $scalar = 1.0;
            assert!(matches!(appended[0].op,
                OpDescriptor::Uniform {
                    value: crate::dialect::tensor::TensorScalarValue::$variant(value)
                } if value.to_bits() == scale.to_bits()));
            assert!(matches!(appended[1].op, OpDescriptor::Sub { .. }));
            assert!(matches!(appended[2].op, OpDescriptor::Mul { .. }));
            let execution = graph
                .evaluate_checked(&[
                    (prediction.erase(), TensorValue::$variant(
                        Tensor::new([2], vec![2.0, 3.0]).unwrap())),
                    (target.erase(), TensorValue::$variant(
                        Tensor::new([2], vec![1.0, 1.0]).unwrap())),
                ])
                .unwrap();
            assert_eq!(execution.value_typed::<$scalar>(derivative).unwrap().data(), &[1.0, 2.0]);
        }};
    }
    verify!(f32, F32);
    verify!(f64, F64);
}

#[test]
fn captured_mse_gradient_scales_retain_the_declared_destination_precision() {
    macro_rules! verify {
        ($scalar:ty, $variant:ident, $bits:expr) => {{
            let mut graph = Graph::default();
            let prediction = graph.input_typed::<$scalar>([3]).unwrap();
            let target = graph.input_typed::<$scalar>([3]).unwrap();
            let loss = graph.mean_squared_error_typed(prediction, target).unwrap();
            let before = graph.nodes().len();
            graph.backward_mse_for(loss.erase(), prediction.erase()).unwrap();
            assert!(matches!(graph.nodes().nth(before).unwrap().op,
                OpDescriptor::Uniform {
                    value: crate::dialect::tensor::TensorScalarValue::$variant(value)
                } if value.to_bits() == $bits));
        }};
    }
    verify!(f32, F32, 0x3f2a_aaab);
    verify!(f64, F64, 0x3fe5_5555_5555_5555);
}

#[test]
fn captured_sgd_derivative_retains_a_subnormal_rate_as_an_exact_constant() {
    macro_rules! verify {
        ($scalar:ty, $variant:ident, $bits:expr) => {{
            let mut graph = Graph::default();
            let weights = graph.input_typed::<$scalar>([1]).unwrap();
            let update_gradient = graph.input_typed::<$scalar>([1]).unwrap();
            let target = graph.input_typed::<$scalar>([1]).unwrap();
            let updated = graph.sgd_update_typed(weights, update_gradient, f32::from_bits(1)).unwrap();
            let loss = graph.mean_squared_error_typed(updated, target).unwrap();
            let before = graph.nodes().len();
            graph.backward_mse_for(loss.erase(), update_gradient.erase()).unwrap();
            assert!(graph.nodes().skip(before).any(|node| matches!(node.op,
                OpDescriptor::Uniform {
                    value: crate::dialect::tensor::TensorScalarValue::$variant(value)
                } if value.to_bits() == $bits)));
        }};
    }
    verify!(f32, F32, 0x8000_0001);
    verify!(f64, F64, 0xb6a0_0000_0000_0000);
}

#[test]
fn explicitly_requested_loss_derivative_retains_the_observable_seed() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let prediction = graph.input_typed::<f64>([1]).unwrap();
    let target = graph.input_typed::<f64>([1]).unwrap();
    let loss = graph.mean_squared_error_typed(prediction, target).unwrap();
    let original_len = graph.nodes().len();
    let derivative = graph.backward_mse_for(loss.erase(), loss.erase()).unwrap();
    assert_eq!(graph.nodes().len(), original_len + 1);
    assert!(graph.shape(derivative).unwrap().is_empty());
    let execution = graph
        .evaluate_checked(&[
            (
                prediction.erase(),
                TensorValue::F64(Tensor::new([1], vec![2.0]).unwrap()),
            ),
            (
                target.erase(),
                TensorValue::F64(Tensor::new([1], vec![1.0]).unwrap()),
            ),
        ])
        .unwrap();
    assert_eq!(
        execution.value_typed::<f64>(derivative).unwrap().data(),
        &[1.0]
    );
}
