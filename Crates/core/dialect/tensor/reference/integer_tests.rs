use super::*;
#[rustfmt::skip]
use crate::{
    PcuExecutionFaultKind,
    dialect::tensor::{
        TensorArithmeticCapability,
        TensorArithmeticRewritePolicy,
        TensorElement,
        TensorPointwiseGroupingPolicy,
        TensorSelectedOperation,
    },
};

fn assert_integer_fault<T: TensorElement + PcuCheckedInteger>(
    left_values: [T; 2],
    right_values: [T; 2],
    operation: BinaryOp,
    expected_kind: PcuExecutionFaultKind,
) {
    let mut graph = Graph::default();
    let left = graph.input_typed::<T>([2]).unwrap();
    let right = graph.input_typed::<T>([2]).unwrap();
    let output = match operation {
        BinaryOp::Add => graph.add_typed(left, right),
        BinaryOp::Sub => graph.sub_typed(left, right),
        BinaryOp::Mul => graph.mul_typed(left, right),
        BinaryOp::Div => panic!("integer division is not admitted by checked tensors"),
    }
    .unwrap();
    let assessment = graph.assess_with(&crate::dialect::tensor::TensorReferenceAssessor);
    assert!(matches!(
        assessment.nodes[output.erase().index].support,
        crate::dialect::tensor::TensorOperationSupport::Supported {
            route: crate::dialect::tensor::TensorExecutionRoute::Reference,
            ..
        }
    ));
    let inputs = [
        (
            left.erase(),
            T::into_value(Tensor::new([2], left_values.to_vec()).unwrap()),
        ),
        (
            right.erase(),
            T::into_value(Tensor::new([2], right_values.to_vec()).unwrap()),
        ),
    ];
    assert!(matches!(
        graph.evaluate(&inputs),
        Err(TensorError::ArithmeticFault {
            value,
            element_index: 1,
            kind,
        }) if value == output.erase() && kind == expected_kind
    ));
}

macro_rules! check_width {
    ($ty:ty) => {{
        assert_integer_fault(
            [0, <$ty>::MAX],
            [0, 1],
            BinaryOp::Add,
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
        assert_integer_fault(
            [0, <$ty>::MIN],
            [0, 1],
            BinaryOp::Sub,
            PcuExecutionFaultKind::ArithmeticUnderflow,
        );
        assert_integer_fault(
            [0, <$ty>::MAX],
            [0, 2],
            BinaryOp::Mul,
            PcuExecutionFaultKind::ArithmeticOverflow,
        );
    }};
}

#[test]
fn checked_integer_add_sub_mul_fault_at_first_element_for_all_widths() {
    check_width!(u8);
    check_width!(u16);
    check_width!(u32);
    check_width!(u64);
    check_width!(i8);
    check_width!(i16);
    check_width!(i32);
    check_width!(i64);
}

#[test]
fn selected_plan_retains_unrequested_integer_fault_effects_and_dependencies() {
    let mut graph = Graph::default();
    let left = graph.input_typed::<u8>([1]).unwrap();
    let right = graph.input_typed::<u8>([1]).unwrap();
    let faulting = graph.add_typed(left, right).unwrap();
    let requested = graph.input_typed::<f32>([1]).unwrap();

    let plan = graph
        .execution_plan_for_outputs(&[requested.erase()])
        .unwrap();

    assert_eq!(plan.output_values(), &[requested.erase()]);
    assert!(plan.node_order().contains(&faulting.erase()));
    assert!(plan.input_values().contains(&left.erase()));
    assert!(plan.input_values().contains(&right.erase()));
    assert!(plan.input_values().contains(&requested.erase()));
}

#[test]
fn checked_integer_chains_preserve_faulting_intermediates_in_fusion_policies() {
    let mut graph = Graph::default();
    let input = graph.input_typed::<u8>([1]).unwrap();
    let one = graph.constant_typed(Tensor::splat([1], 1_u8).unwrap());
    let two = graph.constant_typed(Tensor::splat([1], 2_u8).unwrap());
    let zero = graph.constant_typed(Tensor::splat([1], 0_u8).unwrap());

    let sum = graph.add_typed(input, one).unwrap();
    let canceled = graph.sub_typed(sum, one).unwrap();
    let product = graph.mul_typed(input, two).unwrap();
    let masked = graph.mul_typed(product, zero).unwrap();
    let outputs = [canceled.erase(), masked.erase()];
    let plan = graph.execution_plan_for_outputs(&outputs).unwrap();

    let add_sub = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::BoundedAddSubIdentity,
    );
    assert!(add_sub.bounded_pointwise_fusion_groups().is_empty());
    assert!(
        add_sub
            .operations()
            .iter()
            .all(|operation| matches!(operation, TensorSelectedOperation::Node(_)))
    );

    let multiply = plan.select_lowering_with_grouping(
        TensorArithmeticRewritePolicy::AllowContractedArithmetic,
        TensorArithmeticCapability::ContractedMultiplyAdd,
        TensorPointwiseGroupingPolicy::BoundedMulIdentity,
    );
    assert!(multiply.bounded_mul_fusion_groups().is_empty());
    assert!(
        multiply
            .operations()
            .iter()
            .all(|operation| matches!(operation, TensorSelectedOperation::Node(_)))
    );

    let inputs = [(
        input.erase(),
        TensorValue::from(Tensor::new([1], vec![u8::MAX]).unwrap()),
    )];
    assert!(matches!(
        graph.evaluate(&inputs),
        Err(TensorError::ArithmeticFault {
            value,
            element_index: 0,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
        }) if value == sum.erase()
    ));
}
