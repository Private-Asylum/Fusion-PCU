#![cfg(feature = "tensor")]
//! Graph admission, policy capture, and fault liveness for the half references.
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuBf16Bits,
    PcuCheckedFloat,
    PcuExecutionFaultKind,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy,
    PcuScalarType,
    dialect::tensor::{
        Graph,
        Tensor,
        TensorCheckedReferenceAssessor,
        TensorElement,
        TensorError,
        TensorOperationAssessor,
        TensorOperationSupport,
        TensorValue,
    },
};

fn input<T: TensorElement>(data: [T; 2]) -> TensorValue {
    TensorValue::from_tensor(Tensor::new([2], data.to_vec()).unwrap())
}

fn assert_low_graph<T: TensorElement + PcuCheckedFloat + Eq + core::fmt::Debug>(
    one: T,
    two: T,
    three: T,
    four: T,
    negative_one: T,
    positive_zero: T,
) {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input_typed::<T>([2]).unwrap();
    let right = graph.input_typed::<T>([2]).unwrap();
    let add = graph.add_typed(left, right).unwrap();
    let sub = graph.sub_typed(left, right).unwrap();
    let mul = graph.mul_typed(left, right).unwrap();
    let div = graph.div_typed(left, right).unwrap();
    let relu = graph.relu_typed(left).unwrap();
    let backward = graph.relu_backward(left.erase(), right.erase()).unwrap();
    for node in graph.nodes() {
        assert!(matches!(
            TensorCheckedReferenceAssessor.assess_node(&graph, node),
            TensorOperationSupport::Supported { .. }
        ));
    }
    let execution = graph
        .evaluate_checked(&[
            (left.erase(), input([two, negative_one])),
            (right.erase(), input([one, one])),
        ])
        .unwrap();
    for (id, expected) in [
        (add.erase(), [three, positive_zero]),
        (sub.erase(), [one, two.pcu_checked_neg().unwrap()]),
        (mul.erase(), [two, negative_one]),
        (div.erase(), [two, negative_one]),
        (relu.erase(), [two, positive_zero]),
        (backward, [one, positive_zero]),
    ] {
        assert_eq!(execution.value_typed::<T>(id).unwrap().data(), expected);
    }
    // Low-format elementwise admission does not advertise compound accumulation.
    let left_matrix = graph.input_typed::<T>([2, 1]).unwrap();
    let right_matrix = graph.input_typed::<T>([1, 2]).unwrap();
    assert!(matches!(
        graph.matmul(left_matrix.erase(), right_matrix.erase()),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
    assert!(matches!(
        graph.sgd_update(left.erase(), right.erase(), 0.5),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
    assert!(matches!(
        graph.mean_squared_error(left.erase(), right.erase()),
        Err(TensorError::UnsupportedScalarType { .. })
    ));
    assert_eq!(two.pcu_checked_mul(two).unwrap(), four);
}

#[test]
fn half_and_bfloat_graphs_keep_their_scalar_identity_and_exact_results() {
    assert_low_graph(
        PcuF16Bits::from_bits(0x3c00),
        PcuF16Bits::from_bits(0x4000),
        PcuF16Bits::from_bits(0x4200),
        PcuF16Bits::from_bits(0x4400),
        PcuF16Bits::from_bits(0xbc00),
        PcuF16Bits::from_bits(0),
    );
    assert_low_graph(
        PcuBf16Bits::from_bits(0x3f80),
        PcuBf16Bits::from_bits(0x4000),
        PcuBf16Bits::from_bits(0x4040),
        PcuBf16Bits::from_bits(0x4080),
        PcuBf16Bits::from_bits(0xbf80),
        PcuBf16Bits::from_bits(0),
    );
}

#[test]
fn named_fp8_graphs_keep_exact_formats_and_checked_selection() {
    assert_low_graph(
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0x40),
        PcuF8E4M3FnBits::from_bits(0x44),
        PcuF8E4M3FnBits::from_bits(0x48),
        PcuF8E4M3FnBits::from_bits(0xb8),
        PcuF8E4M3FnBits::from_bits(0),
    );
    assert_low_graph(
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0x40),
        PcuF8E5M2Bits::from_bits(0x42),
        PcuF8E5M2Bits::from_bits(0x44),
        PcuF8E5M2Bits::from_bits(0xbc),
        PcuF8E5M2Bits::from_bits(0),
    );
}

fn fp8_faults<T: TensorElement + PcuCheckedFloat + Eq + core::fmt::Debug>(
    tiny: T,
    one: T,
    half: T,
    zero: T,
    maximum: T,
    two: T,
) {
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input_typed::<T>([2]).unwrap();
    let right = graph.input_typed::<T>([2]).unwrap();
    let result = graph.mul_typed(left, right).unwrap();
    let inputs = [
        (left.erase(), input([tiny; 2])),
        (right.erase(), input([half; 2])),
    ];
    assert!(
        matches!(graph.evaluate_checked(&inputs), Err(TensorError::ArithmeticFault {
        value, element_index: 0, kind: PcuExecutionFaultKind::ArithmeticUnderflow,
    }) if value == result.erase())
    );
    graph
        .set_value_float_underflow_policy(
            result.erase(),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        )
        .unwrap();
    assert_eq!(
        graph
            .evaluate_checked(&inputs)
            .unwrap()
            .value_typed::<T>(result.erase())
            .unwrap()
            .data(),
        [zero; 2]
    );
    graph
        .set_value_float_underflow_policy(
            result.erase(),
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        )
        .unwrap();
    assert!(matches!(graph.evaluate_checked(&[
        (left.erase(), input([one, tiny])), (right.erase(), input([one; 2])),
    ]), Err(TensorError::ArithmeticFault { value, element_index: 1,
        kind: PcuExecutionFaultKind::ArithmeticUnderflow }) if value == result.erase()));
    assert!(matches!(graph.evaluate_checked(&[
        (left.erase(), input([one, maximum])), (right.erase(), input([one, two])),
    ]), Err(TensorError::ArithmeticFault { value, element_index: 1,
        kind: PcuExecutionFaultKind::ArithmeticOverflow }) if value == result.erase()));
    let retry = graph
        .evaluate_checked(&[
            (left.erase(), input([one; 2])),
            (right.erase(), input([two; 2])),
        ])
        .unwrap();
    assert_eq!(
        retry.value_typed::<T>(result.erase()).unwrap().data(),
        [two; 2]
    );
}

#[test]
fn named_fp8_graphs_capture_underflow_and_preserve_fault_indices() {
    fp8_faults(
        PcuF8E4M3FnBits::from_bits(1),
        PcuF8E4M3FnBits::from_bits(0x38),
        PcuF8E4M3FnBits::from_bits(0x30),
        PcuF8E4M3FnBits::from_bits(0),
        PcuF8E4M3FnBits::from_bits(0x7e),
        PcuF8E4M3FnBits::from_bits(0x40),
    );
    fp8_faults(
        PcuF8E5M2Bits::from_bits(1),
        PcuF8E5M2Bits::from_bits(0x3c),
        PcuF8E5M2Bits::from_bits(0x38),
        PcuF8E5M2Bits::from_bits(0),
        PcuF8E5M2Bits::from_bits(0x7b),
        PcuF8E5M2Bits::from_bits(0x40),
    );
}

#[test]
fn half_division_fault_and_explicit_underflow_policy_survive_graph_capture() {
    let tiny = PcuF16Bits::from_bits(1);
    let half = PcuF16Bits::from_bits(0x3800);
    let mut graph = Graph::try_new().unwrap();
    let left = graph.input_typed::<PcuF16Bits>([2]).unwrap();
    let right = graph.input_typed::<PcuF16Bits>([2]).unwrap();
    let product = graph.mul_typed(left, right).unwrap();
    let inputs = [
        (left.erase(), input([tiny; 2])),
        (right.erase(), input([half; 2])),
    ];
    assert!(matches!(graph.evaluate_checked(&inputs),
        Err(TensorError::ArithmeticFault { value, element_index: 0,
            kind: PcuExecutionFaultKind::ArithmeticUnderflow }) if value == product.erase()));
    graph
        .set_value_float_underflow_policy(
            product.erase(),
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        )
        .unwrap();
    let execution = graph.evaluate_checked(&inputs).unwrap();
    assert_eq!(
        execution
            .value_typed::<PcuF16Bits>(product.erase())
            .unwrap()
            .data(),
        [PcuF16Bits::from_bits(0); 2]
    );

    let mut division = Graph::try_new().unwrap();
    let left = division.input_typed::<PcuBf16Bits>([2]).unwrap();
    let right = division.input_typed::<PcuBf16Bits>([2]).unwrap();
    let result = division.div_typed(left, right).unwrap();
    let inputs = [
        (left.erase(), input([PcuBf16Bits::from_bits(0x3f80); 2])),
        (
            right.erase(),
            input([PcuBf16Bits::from_bits(0x3f80), PcuBf16Bits::from_bits(0)]),
        ),
    ];
    assert!(matches!(division.evaluate_checked(&inputs),
        Err(TensorError::ArithmeticFault { value, element_index: 1,
            kind: PcuExecutionFaultKind::DivideByZero }) if value == result.erase()));
}

#[test]
fn discarded_half_work_remains_an_observable_fault_effect() {
    let mut graph = Graph::try_new().unwrap();
    let source = graph.input_typed::<PcuF16Bits>([1]).unwrap();
    let discarded = graph.relu_typed(source).unwrap();
    let terminal = graph.input_typed::<PcuF16Bits>([1]).unwrap();
    let selected = graph
        .execution_plan_for_outputs(&[terminal.erase()])
        .unwrap();
    // The final output is unrelated, but faulting arithmetic must not be pruned.
    assert_eq!(selected.node_order().len(), graph.nodes().count());
    assert_eq!(
        graph.node(source.erase()).unwrap().scalar_type,
        PcuScalarType::F16
    );
    let inputs = [
        (
            source.erase(),
            TensorValue::from_tensor(
                Tensor::new([1], vec![PcuF16Bits::from_bits(0x7c00)]).unwrap(),
            ),
        ),
        (
            terminal.erase(),
            TensorValue::from_tensor(Tensor::new([1], vec![PcuF16Bits::from_bits(0)]).unwrap()),
        ),
    ];
    assert!(matches!(graph.evaluate_checked(&inputs),
        Err(TensorError::ArithmeticFault { value, element_index: 0,
            kind: PcuExecutionFaultKind::InvalidFloatingOperand })
            if value == discarded.erase()));
}
