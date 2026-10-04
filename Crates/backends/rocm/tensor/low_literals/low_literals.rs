//! Dense low-format producers transport raw bits; arithmetic keeps its existing policy.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBf16Bits, PcuF16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits,
    PcuFloatUnderflowPolicy, PcuNumericalMode,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph, Tensor, TensorElement, TensorExecutionRoute, TensorOperandRepresentation,
    TensorOperationSupport,
};
fn admits<T: TensorElement>(one: T) {
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let input = graph.input_typed::<T>([3]).unwrap();
    let constant = graph.constant_typed(Tensor::new([3], vec![one; 3]).unwrap());
    let uniform = graph.uniform_typed([3], one).unwrap();
    let positive = graph.add_typed(input, input).unwrap();
    let negative = graph.add_typed(input, constant).unwrap();
    for value in [positive, negative] {
        graph
            .set_value_float_underflow_policy(
                value.erase(),
                PcuFloatUnderflowPolicy::RejectSubnormalResult,
            )
            .unwrap();
    }
    assert!(matches!(
        super::assess_tensor_node(&graph, graph.node(positive.erase()).unwrap()),
        TensorOperationSupport::Supported {
            route: TensorExecutionRoute::Synthesized,
            ..
        }
    ));
    for value in [constant.erase(), uniform.erase(), negative.erase()] {
        let node = graph.node(value).unwrap();
        if value != negative.erase() {
            assert_eq!(node.numerical_mode, None);
            assert_eq!(node.float_underflow_policy, None);
            assert!(super::rocm_supports_operand_representation(
                node,
                TensorOperandRepresentation::Dense
            ));
        }
        assert!(matches!(
            super::assess_tensor_node(&graph, node),
            TensorOperationSupport::Supported { .. }
        ));
    }
}
#[test]
fn four_low_dense_literals_and_consumers_are_admitted_without_literal_math_policy() {
    admits(PcuF16Bits::from_bits(0x3c00));
    admits(PcuBf16Bits::from_bits(0x3f80));
    admits(PcuF8E4M3FnBits::from_bits(0x38));
    admits(PcuF8E5M2Bits::from_bits(0x3c));
}

fn raw_bits<T: TensorElement>(values: [T; 4]) {
    let mut graph = Graph::default();
    let constant = graph.constant_typed(Tensor::new([4], values.to_vec()).unwrap());
    let node = graph.node(constant.erase()).unwrap();
    let layout = super::RocmPhysicalLayout::dense(u64::try_from(4 * T::ENCODED_SIZE).unwrap());
    let expected = values
        .into_iter()
        .flat_map(|value| value.encode_le().as_ref().to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        super::literal::encoded_low_float(node, layout).unwrap(),
        expected
    );
    assert!(
        super::literal::encoded_low_float(node, super::RocmPhysicalLayout::uniform_scalar())
            .is_err()
    );
    let mut forged = node;
    forged.shape = &[2, 2];
    assert!(super::literal::encoded_low_float(forged, layout).is_err());
    forged.scalar_type = fusion_pcu::PcuScalarType::F32;
    assert!(super::literal::encoded_low_float(forged, layout).is_err());
    assert!(super::literal::encoded_low_float(node, super::RocmPhysicalLayout::dense(1)).is_err());
    for value in values {
        let uniform = graph.uniform_typed([4], value).unwrap();
        let node = graph.node(uniform.erase()).unwrap();
        assert_eq!(node.numerical_mode, None);
        assert_eq!(node.float_underflow_policy, None);
        assert_eq!(
            super::literal::encoded_low_float(node, layout).unwrap(),
            value.encode_le().as_ref().repeat(4)
        );
    }
}

#[test]
fn low_literal_codec_preserves_nan_payloads_signed_zero_and_minimum_subnormal() {
    raw_bits([0x7c01, 0x7e55, 0x8000, 1].map(PcuF16Bits::from_bits));
    raw_bits([0x7f81, 0x7fc5, 0x8000, 1].map(PcuBf16Bits::from_bits));
    raw_bits([0x7f, 0xff, 0x80, 1].map(PcuF8E4M3FnBits::from_bits));
    raw_bits([0x7d, 0x7e, 0x80, 1].map(PcuF8E5M2Bits::from_bits));
}
