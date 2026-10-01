//! Exceptional intermediates, ordered faults, IEEE underflow and typed rate regression tests.
use super::*;
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuNumericalMode,
    dialect::tensor::{Graph, TensorCheckedReferenceAssessor, TensorOperationAssessor, TensorOperationSupport},
};

fn graph_f32(weights: &[f32], gradient: &[f32], rate: f32) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let weights = graph.constant_typed(Tensor::new([weights.len()], weights.to_vec()).unwrap());
    let gradient = graph.constant_typed(Tensor::new([gradient.len()], gradient.to_vec()).unwrap());
    let output = graph.sgd_update_typed(weights, gradient, rate).unwrap();
    (graph, output.erase())
}

#[test]
fn tiny_inexact_intermediate_faults_even_when_final_output_would_be_normal() {
    let (mut graph, output) = graph_f32(&[1.0], &[f32::from_bits(1)], 0.5);
    assert!(
        matches!(graph.evaluate_checked(&[]), Err(TensorError::CompoundArithmeticFault {
        value, element_index: 0, reduction_index: 0, step: TensorArithmeticStep::Multiply,
        kind: PcuExecutionFaultKind::ArithmeticUnderflow,
    }) if value == output)
    );
    graph
        .set_value_float_underflow_policy(output, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
        .unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f32>(output)
            .unwrap()
            .data(),
        &[1.0]
    );
}

#[test]
fn exact_subnormal_is_not_ieee_underflow_but_tight_policy_rejects_it() {
    let (mut graph, output) = graph_f32(&[0.0], &[f32::from_bits(1)], 1.0);
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f32>(output)
            .unwrap()
            .data()[0]
            .to_bits(),
        0x8000_0001
    );
    graph
        .set_value_float_underflow_policy(output, PcuFloatUnderflowPolicy::RejectSubnormalResult)
        .unwrap();
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::CompoundArithmeticFault {
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            ..
        })
    ));
}

#[test]
fn subtraction_overflow_is_distinct_and_first_element_precedes_later_multiply_faults() {
    let (graph, _) = graph_f32(&[f32::MAX, 0.0], &[-f32::MAX, f32::MAX], 2.0);
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::CompoundArithmeticFault {
            element_index: 0,
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            ..
        })
    ));
    let (graph, _) = graph_f32(&[f32::MAX, f32::NAN], &[-f32::MAX, f32::MAX], 1.0);
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::CompoundArithmeticFault {
            element_index: 0,
            reduction_index: 0,
            step: TensorArithmeticStep::Subtract,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            ..
        })
    ));
}

#[test]
fn invalid_weight_is_subtract_fault_and_invalid_gradient_is_multiply_fault() {
    for (weight, gradient, expected) in [
        (f32::NAN, 1.0, TensorArithmeticStep::Subtract),
        (1.0, f32::INFINITY, TensorArithmeticStep::Multiply),
        (f32::NAN, f32::NAN, TensorArithmeticStep::Multiply),
    ] {
        let (graph, _) = graph_f32(&[weight], &[gradient], 0.0);
        assert!(
            matches!(graph.evaluate_checked(&[]), Err(TensorError::CompoundArithmeticFault {
            step, kind: PcuExecutionFaultKind::InvalidFloatingOperand, ..
        }) if step == expected)
        );
    }
}

#[test]
fn no_contraction_and_signed_zero_have_exact_observable_results() {
    let rate = 1.000_000_1_f32;
    let gradient = 0.999_999_94_f32;
    let (graph, output) = graph_f32(&[1.0], &[gradient], rate);
    let result = graph.evaluate_checked(&[]).unwrap();
    let actual = result.value_typed::<f32>(output).unwrap().data()[0];
    assert_eq!(actual.to_bits(), 0.0_f32.to_bits());
    assert_ne!(actual.to_bits(), (-rate).mul_add(gradient, 1.0).to_bits());
    let (graph, output) = graph_f32(&[-0.0], &[0.0], 1.0);
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f32>(output)
            .unwrap()
            .data()[0]
            .to_bits(),
        (-0.0_f32).to_bits()
    );
}

#[test]
fn binary64_uses_exact_widened_rate_and_the_same_fault_steps() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let weights = graph.constant_typed(Tensor::new([2], alloc::vec![3.0_f64, -0.0]).unwrap());
    let gradient = graph.constant_typed(Tensor::new([2], alloc::vec![1.0_f64, 0.0]).unwrap());
    let rate = 0.1_f32;
    let output = graph.sgd_update_typed(weights, gradient, rate).unwrap();
    let result = graph.evaluate_checked(&[]).unwrap();
    let data = result.value_typed::<f64>(output.erase()).unwrap().data();
    assert_eq!(data[0].to_bits(), (3.0 - f64::from(rate)).to_bits());
    assert_ne!(data[0].to_bits(), (3.0_f64 - 0.1).to_bits());
    assert_eq!(data[1].to_bits(), (-0.0_f64).to_bits());
    let gradient =
        graph.constant_typed(Tensor::new([2], alloc::vec![f64::from_bits(1), 1.0]).unwrap());
    graph.sgd_update_typed(weights, gradient, 0.5).unwrap();
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::CompoundArithmeticFault {
            element_index: 0,
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            ..
        })
    ));
}

#[test]
fn boundary_stays_uncertified_and_discarded_strict_updates_retain_fault_effects() {
    let (mut graph, output) = graph_f32(&[1.0], &[1.0], 0.5);
    let input = graph.nodes().next().unwrap().value;
    assert!(
        graph
            .execution_plan_for_outputs(&[input])
            .unwrap()
            .nodes()
            .any(|node| node.value == output)
    );
    assert!(matches!(
        TensorCheckedReferenceAssessor.assess_node(&graph, graph.node(output).unwrap()),
        TensorOperationSupport::Supported { .. }
    ));
    graph
        .set_value_numerical_mode(output, PcuNumericalMode::Boundary)
        .unwrap();
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::UnsupportedNumericalMode {
            mode: PcuNumericalMode::Boundary,
            ..
        })
    ));
    assert!(matches!(
        TensorCheckedReferenceAssessor.assess_node(&graph, graph.node(output).unwrap()),
        TensorOperationSupport::Unsupported { .. }
    ));
}

#[test]
fn scalar_and_empty_shape_are_valid_reference_values_and_nonfinite_rates_reject_at_capture() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let scalar = graph.constant_typed(Tensor::scalar(2.0_f64));
    let output = graph.sgd_update_typed(scalar, scalar, 0.5).unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f64>(output.erase())
            .unwrap()
            .data(),
        &[1.0]
    );
    let empty = graph.constant_typed(Tensor::<f32>::new([0], alloc::vec![]).unwrap());
    graph.sgd_update_typed(empty, empty, 1.0).unwrap();
    assert!(graph.evaluate_checked(&[]).is_ok());
    for rate in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            graph.sgd_update_typed(scalar, scalar, rate),
            Err(TensorError::InvalidLearningRate)
        );
    }
}
