#[rustfmt::skip]
use fusion_pcu::{
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticStep,
    TensorError,
};
use super::StrictMatMulSpec;

#[test]
fn strict_factory_is_independent_of_underflow_and_preserves_ordered_abi() {
    for scalar in [PcuScalarType::F32, PcuScalarType::F64] {
        let mut graph = Graph::default();
        graph.set_numerical_mode(PcuNumericalMode::Strict);
        let left = graph.input([3, 2], scalar).unwrap();
        let right = graph.input([4, 3], scalar).unwrap();
        let output = graph.matmul_transposed(left, right, true, true).unwrap();
        graph
            .set_value_float_underflow_policy(
                output,
                PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            )
            .unwrap();
        let spec = StrictMatMulSpec::from_node(&graph, graph.node(output).unwrap()).unwrap();
        assert_eq!(spec.shape().invocation_count().get(), 8);
        let width = if scalar == PcuScalarType::F32 { 4 } else { 8 };
        assert_eq!(
            spec.requirements().map(|r| r.min_required_bytes),
            [6 * width, 12 * width, 8 * width]
        );
        let source = spec.source();
        assert!(source.contains("for (unsigned long long k = 0; k < 3ull; ++k)"));
        assert!(source.contains("left[k * 2ull + row]"));
        assert!(source.contains("right[column * 3ull + k]"));
        assert!(source.contains("2u, 2u)"));
        assert!(source.contains("0u, 2u)"));
        assert!(source.contains("atomicMin(fault"));
        assert!(!source.contains("fma("));
        assert!(!source.contains("float accumulator"));
    }
}

#[test]
fn default_mode_does_not_silently_admit_a_strict_factory() {
    let mut graph = Graph::default();
    let left = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let right = graph.input([3, 4], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    assert!(StrictMatMulSpec::from_node(&graph, graph.node(output).unwrap()).is_none());
    graph
        .set_value_numerical_mode(output, PcuNumericalMode::Strict)
        .unwrap();
    assert!(StrictMatMulSpec::from_node(&graph, graph.node(output).unwrap()).is_some());
}

#[test]
fn compound_fault_preserves_cell_reduction_and_arithmetic_step() {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let left = graph.input([2, 3], PcuScalarType::F32).unwrap();
    let right = graph.input([3, 2], PcuScalarType::F32).unwrap();
    let output = graph.matmul(left, right).unwrap();
    let spec = StrictMatMulSpec::from_node(&graph, graph.node(output).unwrap()).unwrap();
    let fault = PcuExecutionFault {
        kind: PcuExecutionFaultKind::ArithmeticOverflow,
        invocation_id: (2 * 3 + 1) * 2 + 1,
        recovered: false,
    };
    assert_eq!(
        spec.fault_error(output, fault).unwrap(),
        TensorError::CompoundArithmeticFault {
            value: output,
            element_index: 2,
            reduction_index: 1,
            step: TensorArithmeticStep::Add,
            kind: PcuExecutionFaultKind::ArithmeticOverflow,
        }
    );
}
