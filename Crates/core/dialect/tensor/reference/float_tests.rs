//! Checked binary32 reference fault and selected-effect acceptance.
use super::*;
use crate::PcuExecutionFaultKind;

#[test]
fn binary32_faults_are_attributed_without_publishing_outputs() {
    for (operation, left, right, expected) in [
        (
            BinaryOp::Add,
            f32::MAX,
            f32::MAX,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            BinaryOp::Sub,
            -f32::MAX,
            f32::MAX,
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            BinaryOp::Mul,
            f32::from_bits(1),
            0.5,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        (
            BinaryOp::Mul,
            f32::MIN_POSITIVE,
            f32::from_bits(0x3f7f_ffff),
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        (
            BinaryOp::Div,
            f32::from_bits(1),
            2.0,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        ),
        (BinaryOp::Div, 1.0, 0.0, PcuExecutionFaultKind::DivideByZero),
        (
            BinaryOp::Div,
            f32::MAX,
            f32::from_bits(1),
            PcuExecutionFaultKind::ArithmeticOverflow,
        ),
        (
            BinaryOp::Add,
            f32::INFINITY,
            0.0,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ),
    ] {
        let mut graph = Graph::default();
        let lhs = graph.input_typed::<f32>([2]).unwrap();
        let rhs = graph.input_typed::<f32>([2]).unwrap();
        let output = match operation {
            BinaryOp::Add => graph.add_typed(lhs, rhs),
            BinaryOp::Sub => graph.sub_typed(lhs, rhs),
            BinaryOp::Mul => graph.mul_typed(lhs, rhs),
            BinaryOp::Div => graph.div_typed(lhs, rhs),
        }
        .unwrap();
        let inputs = [
            (
                lhs.erase(),
                TensorValue::F32(Tensor::new([2], vec![1.0, left]).unwrap()),
            ),
            (
                rhs.erase(),
                TensorValue::F32(Tensor::new([2], vec![1.0, right]).unwrap()),
            ),
        ];
        assert!(
            matches!(graph.evaluate(&inputs), Err(TensorError::ArithmeticFault {
            value, element_index: 1, kind,
        }) if value == output.erase() && kind == expected)
        );
    }
}

#[test]
fn selected_plan_retains_discarded_float_checked_effects() {
    let mut graph = Graph::default();
    let lhs = graph.input_typed::<f32>([1]).unwrap();
    let rhs = graph.input_typed::<f32>([1]).unwrap();
    let discarded = graph.mul_typed(lhs, rhs).unwrap();
    let plan = graph.execution_plan_for_outputs(&[lhs.erase()]).unwrap();
    assert!(plan.nodes().any(|node| node.value == discarded.erase()));
}
