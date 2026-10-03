use super::*;
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuNumericalMode,
};
#[rustfmt::skip]
use crate::dialect::tensor::{
    Graph,
    TensorOperationAssessor,
    TensorCheckedReferenceAssessor,
    TensorOperationSupport,
};
use alloc::vec;

fn graph(prediction: &[f32], target: &[f32]) -> (Graph, ValueId) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let prediction =
        graph.constant_typed(Tensor::new([prediction.len()], prediction.to_vec()).unwrap());
    let target = graph.constant_typed(Tensor::new([target.len()], target.to_vec()).unwrap());
    let output = graph
        .mean_squared_error_typed(prediction, target)
        .unwrap()
        .erase();
    (graph, output)
}

#[test]
fn strict_loss_has_ordered_individual_faults_and_no_boundary_certificate() {
    for (prediction, target, index, step, kind) in [
        (
            vec![f32::MAX],
            vec![-f32::MAX],
            0,
            TensorArithmeticStep::Subtract,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            vec![f32::MAX],
            vec![0.0],
            0,
            TensorArithmeticStep::Multiply,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            vec![1.0, f32::NAN],
            vec![0.0, 0.0],
            1,
            TensorArithmeticStep::Subtract,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ),
        (
            vec![f32::from_bits(1)],
            vec![0.0],
            0,
            TensorArithmeticStep::Multiply,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
    ] {
        let (graph, output) = graph(&prediction, &target);
        assert!(
            matches!(graph.evaluate_checked(&[]),Err(TensorError::CompoundArithmeticFault {
            value,element_index:0,reduction_index,step:actual,kind:actual_kind
        }) if value==output && reduction_index==index && actual==step && actual_kind==kind)
        );
    }
    let (mut graph, output) = graph(&[3.0, 5.0], &[1.0, 1.0]);
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f32>(output)
            .unwrap()
            .data(),
        &[10.0]
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
        Err(TensorError::UnsupportedNumericalMode { .. })
    ));
}

#[test]
fn binary64_loss_and_exact_subnormal_follow_the_same_contract() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let p = graph.constant_typed(Tensor::new([2], vec![3.0_f64, 5.0]).unwrap());
    let t = graph.constant_typed(Tensor::new([2], vec![1.0_f64, 1.0]).unwrap());
    let output = graph.mean_squared_error_typed(p, t).unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f64>(output.erase())
            .unwrap()
            .data(),
        &[10.0]
    );
    let (mut graph, output) = self::graph(&[f32::from_bits(1)], &[0.0]);
    graph
        .set_value_float_underflow_policy(output, PcuFloatUnderflowPolicy::AllowGradualUnderflow)
        .unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f32>(output)
            .unwrap()
            .data()[0]
            .to_bits(),
        0
    );
}

#[test]
fn discarded_loss_preserves_checked_effects_and_deterministic_stays_unavailable() {
    let (mut graph, output) = graph(&[f32::NAN], &[0.0]);
    let input = graph.nodes().next().unwrap().value;
    assert!(
        graph
            .execution_plan_for_outputs(&[input])
            .unwrap()
            .nodes()
            .any(|node| node.value == output)
    );
    let mut options = graph.numerical_options();
    options.reproducibility = crate::PcuReproducibility::PortableV1;
    graph.set_value_numerical_options(output, options).unwrap();
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::UnsupportedNumericalOptions { .. })
    ));
}
