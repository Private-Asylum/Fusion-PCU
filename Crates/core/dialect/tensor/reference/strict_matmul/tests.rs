//! Strict compound numerical contract controls and fault-location regression tests.
use super::*;
use alloc::vec::Vec;
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    PcuNumericalMode,
    dialect::tensor::{Graph, TensorValue},
};

fn run_f32(
    left: &[f32],
    right: &[f32],
    policy: PcuFloatUnderflowPolicy,
) -> Result<Vec<f32>, TensorError> {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.constant_typed(Tensor::new([1, left.len()], left.to_vec()).unwrap());
    let b = graph.constant_typed(Tensor::new([right.len(), 1], right.to_vec()).unwrap());
    let out = graph.matmul_typed(a, b).unwrap();
    graph
        .set_value_float_underflow_policy(out.erase(), policy)
        .unwrap();
    let values = graph.evaluate_checked(&[])?;
    Ok(values
        .value_typed::<f32>(out.erase())
        .unwrap()
        .data()
        .to_vec())
}

#[test]
fn intermediate_tiny_inexact_product_is_checked_even_when_final_cell_would_be_normal() {
    let error = run_f32(
        &[f32::from_bits(1), 1.0],
        &[0.5, 1.0],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        TensorError::CompoundArithmeticFault {
            element_index: 0,
            reduction_index: 0,
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            ..
        }
    ));
    assert_eq!(
        run_f32(
            &[f32::from_bits(1), 1.0],
            &[0.5, 1.0],
            PcuFloatUnderflowPolicy::AllowGradualUnderflow
        )
        .unwrap(),
        vec![1.0]
    );
}

#[test]
fn exact_subnormal_and_exact_zero_are_independent_policy_controls() {
    let tiny = f32::from_bits(1);
    assert_eq!(
        run_f32(&[tiny], &[1.0], PcuFloatUnderflowPolicy::IeeeAfterRounding).unwrap()[0].to_bits(),
        tiny.to_bits()
    );
    assert!(matches!(
        run_f32(
            &[tiny],
            &[1.0],
            PcuFloatUnderflowPolicy::RejectSubnormalResult
        ),
        Err(TensorError::CompoundArithmeticFault {
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            ..
        })
    ));
    assert_eq!(
        run_f32(
            &[1.0, -1.0],
            &[1.0, 1.0],
            PcuFloatUnderflowPolicy::RejectSubnormalResult
        )
        .unwrap(),
        vec![0.0]
    );
}

#[test]
fn ordered_overflow_checks_precede_later_cancellation_and_detect_final_faults() {
    for right in [vec![1.0, 1.0, -1.0], vec![1.0, 1.0, 0.0]] {
        assert!(matches!(
            run_f32(
                &[f32::MAX; 3],
                &right,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            Err(TensorError::CompoundArithmeticFault {
                reduction_index: 1,
                step: TensorArithmeticStep::Add,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
                ..
            })
        ));
    }
    assert!(matches!(
        run_f32(
            &[f32::MAX],
            &[2.0],
            PcuFloatUnderflowPolicy::IeeeAfterRounding
        ),
        Err(TensorError::CompoundArithmeticFault {
            reduction_index: 0,
            step: TensorArithmeticStep::Multiply,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            ..
        })
    ));
}

#[test]
fn first_fault_is_row_major_cell_then_increasing_reduction_then_multiply_add() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.constant_typed(Tensor::new([2, 2], vec![1.0, f32::MAX, f32::MAX, 1.0]).unwrap());
    let b = graph.constant_typed(Tensor::new([2, 2], vec![1.0, 2.0, 2.0, 2.0]).unwrap());
    let out = graph.matmul_typed(a, b).unwrap();
    assert!(
        matches!(graph.evaluate_checked(&[]), Err(TensorError::CompoundArithmeticFault {
        value, element_index: 0, reduction_index: 1, step: TensorArithmeticStep::Multiply,
        kind: PcuExecutionFaultKind::ArithmeticOverflow,
    }) if value == out.erase())
    );
}

#[test]
fn binary64_strict_controls_and_transpose_use_the_same_checked_sequence() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.constant_typed(Tensor::new([2, 1], vec![f64::from_bits(1), 1.0]).unwrap());
    let b = graph.constant_typed(Tensor::new([1, 2], vec![0.5_f64, 1.0]).unwrap());
    let out = graph.matmul_transposed_typed(a, b, true, true).unwrap();
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::CompoundArithmeticFault {
            reduction_index: 0,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow,
            ..
        })
    ));
    graph
        .set_value_float_underflow_policy(
            out.erase(),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        )
        .unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f64>(out.erase())
            .unwrap()
            .data(),
        &[1.0]
    );
    let a = graph.constant_typed(Tensor::new([1, 1], vec![f64::MIN_POSITIVE]).unwrap());
    let b = graph.constant_typed(Tensor::new([1, 1], vec![0.5_f64]).unwrap());
    let out = graph.matmul_typed(a, b).unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f64>(out.erase())
            .unwrap()
            .data()[0]
            .to_bits(),
        (f64::MIN_POSITIVE * 0.5).to_bits()
    );
}

#[test]
fn checked_entry_does_not_certify_raw_boundary_compounds_and_metadata_is_orthogonal() {
    let mut graph = Graph::default();
    let a = graph.constant_typed(Tensor::new([1, 1], vec![1.0_f32]).unwrap());
    let out = graph.matmul_typed(a, a).unwrap();
    assert_eq!(
        graph.node(out.erase()).unwrap().numerical_mode,
        Some(PcuNumericalMode::Boundary)
    );
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::UnsupportedNumericalMode {
            mode: PcuNumericalMode::Boundary,
            ..
        })
    ));
    graph
        .set_value_float_underflow_policy(
            out.erase(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    graph
        .set_value_numerical_mode(out.erase(), PcuNumericalMode::Strict)
        .unwrap();
    assert_eq!(
        graph.node(out.erase()).unwrap().float_underflow_policy,
        Some(PcuFloatUnderflowPolicy::RejectSubnormalResult)
    );
    assert!(graph.evaluate_checked(&[]).is_ok());
    assert!(
        graph
            .execution_plan_for_outputs(&[a.erase()])
            .unwrap()
            .nodes()
            .any(|node| node.value == out.erase())
    );
    assert!(matches!(
        graph.evaluate(&[]).unwrap().value(out.erase()).unwrap(),
        TensorValue::F32(_)
    ));
}

#[test]
fn multiplication_rounding_is_not_erased_by_fma_and_nonfinite_operands_fail() {
    let higher = 1.0 + f32::EPSILON;
    let lower = 1.0 - f32::EPSILON;
    let result = run_f32(
        &[-1.0, higher],
        &[1.0, lower],
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
    )
    .unwrap();
    assert_eq!(result[0].to_bits(), 0.0_f32.to_bits());
    assert_ne!(higher.mul_add(lower, -1.0).to_bits(), result[0].to_bits());
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(matches!(
            run_f32(
                &[invalid],
                &[0.0],
                PcuFloatUnderflowPolicy::AllowGradualUnderflow
            ),
            Err(TensorError::CompoundArithmeticFault {
                reduction_index: 0,
                step: TensorArithmeticStep::Multiply,
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                ..
            })
        ));
    }
}

#[test]
fn binary64_overflow_is_checked_before_cancellation_and_strict_loss_is_supported() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph.constant_typed(Tensor::new([1, 3], vec![f64::MAX; 3]).unwrap());
    let b = graph.constant_typed(Tensor::new([3, 1], vec![1.0_f64, 1.0, -1.0]).unwrap());
    graph.matmul_typed(a, b).unwrap();
    assert!(matches!(
        graph.evaluate_checked(&[]),
        Err(TensorError::CompoundArithmeticFault {
            reduction_index: 1,
            step: TensorArithmeticStep::Add,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
            ..
        })
    ));
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let value = graph.constant_typed(Tensor::new([1], vec![1.0_f32]).unwrap());
    let loss = graph
        .mean_squared_error(value.erase(), value.erase())
        .unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&[])
            .unwrap()
            .value_typed::<f32>(loss)
            .unwrap()
            .data(),
        &[0.0]
    );
}

#[test]
fn discarded_boundary_compounds_retain_potential_fault_and_admission_effects() {
    let mut graph = Graph::default();
    let left = graph.input_typed::<f32>([1, 1]).unwrap();
    let discarded = graph.matmul_typed(left, left).unwrap();
    let plan = graph.execution_plan_for_outputs(&[left.erase()]).unwrap();
    assert!(plan.nodes().any(|node| node.value == discarded.erase()));
    let node = plan
        .nodes()
        .find(|node| node.value == discarded.erase())
        .unwrap();
    let support = crate::dialect::tensor::TensorOperationAssessor::assess_node(
        &crate::dialect::tensor::TensorCheckedReferenceAssessor,
        &graph,
        node,
    );
    assert!(matches!(
        support,
        crate::dialect::tensor::TensorOperationSupport::Unsupported { .. }
    ));
}
