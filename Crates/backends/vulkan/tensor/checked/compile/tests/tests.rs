use super::*;
#[rustfmt::skip]
use fusion_pcu::{PcuExecutionFaultKind, PcuFloatUnderflowPolicy};
use fusion_pcu::dialect::tensor::Graph;

#[test]
fn cold_pointwise_retains_actual_operation_fault_law() {
    for scalar in [
        PcuScalarType::F16,
        PcuScalarType::BF16,
        PcuScalarType::F8E4M3FN,
        PcuScalarType::F8E5M2,
        PcuScalarType::F32,
        PcuScalarType::F64,
    ] {
        let mut graph = Graph::default();
        let left = graph.input([7], scalar).unwrap();
        let right = graph.input([7], scalar).unwrap();
        let add = graph.add(left, right).unwrap();
        let div = graph.div(left, right).unwrap();
        let relu = graph.relu(left).unwrap();
        let backward = graph.relu_backward(left, right).unwrap();
        for policy in [
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for value in [add, div, relu, backward] {
                let mut node = graph.node(value).unwrap();
                node.float_underflow_policy = Some(policy);
                let lowered = lower(node, 7).unwrap();
                assert_eq!(
                    lowered
                        .fault_law
                        .allows(PcuExecutionFaultKind::DivideByZero, false),
                    value == div
                );
                assert!(
                    lowered
                        .fault_law
                        .allows(PcuExecutionFaultKind::InvalidFloatingOperand, false)
                );
                assert!(
                    !lowered
                        .fault_law
                        .allows(PcuExecutionFaultKind::ArithmeticOverflow, true)
                );
                assert_eq!(
                    lowered
                        .fault_law
                        .allows(PcuExecutionFaultKind::ArithmeticUnderflow, false),
                    policy == PcuFloatUnderflowPolicy::RejectSubnormalResult
                        || (value == div && policy == PcuFloatUnderflowPolicy::IeeeAfterRounding)
                );
            }
        }
    }
}

#[test]
fn cold_unsigned_tensor_subtraction_cannot_report_overflow_or_domain_faults() {
    for scalar in [
        PcuScalarType::U8,
        PcuScalarType::U16,
        PcuScalarType::U32,
        PcuScalarType::U64,
        PcuScalarType::U128,
        PcuScalarType::U256,
        PcuScalarType::U512,
    ] {
        let mut graph = Graph::default();
        let left = graph.input([7], scalar).unwrap();
        let right = graph.input([7], scalar).unwrap();
        let value = graph.sub(left, right).unwrap();
        let lowered = lower(graph.node(value).unwrap(), 7).unwrap();
        assert!(
            lowered
                .fault_law
                .allows(PcuExecutionFaultKind::ArithmeticUnderflow, false)
        );
        for kind in [
            PcuExecutionFaultKind::ArithmeticOverflow,
            PcuExecutionFaultKind::DivideByZero,
            PcuExecutionFaultKind::SignedDivisionOverflow,
            PcuExecutionFaultKind::InvalidFloatingOperand,
        ] {
            assert!(!lowered.fault_law.allows(kind, false));
        }
    }
}
