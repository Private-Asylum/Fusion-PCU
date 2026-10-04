//! Actual typed low-format immutable producers and consumers.
#[rustfmt::skip]
use super::{
    assess_tensor_node,
    Graph,
    Tensor,
    TensorOperationSupport,
};
#[test]
fn actual_low_float_producers_and_consumers_admit_typed_materialization() {
    macro_rules! check {
        ($ty:ty) => {{
            let value = <$ty>::from_bits(1);
            let mut graph = Graph::default();
            let input = graph.input_typed::<$ty>([3]).unwrap();
            let constant = graph.constant_typed(Tensor::new([3], vec![value; 3]).unwrap());
            let uniform = graph.uniform_typed([3], value).unwrap();
            let sum = graph.add_typed(input, constant).unwrap();
            let product = graph.mul_typed(sum, uniform).unwrap();
            assert!(matches!(
                assess_tensor_node(&graph, graph.node(input.erase()).unwrap()),
                TensorOperationSupport::Supported { .. }
            ));
            for typed in [constant, uniform, sum, product] {
                assert!(matches!(
                    assess_tensor_node(&graph, graph.node(typed.erase()).unwrap()),
                    TensorOperationSupport::Supported { .. }
                ));
            }
        }};
    }
    check!(fusion_pcu::PcuF16Bits);
    check!(fusion_pcu::PcuBf16Bits);
    check!(fusion_pcu::PcuF8E4M3FnBits);
    check!(fusion_pcu::PcuF8E5M2Bits);
}

#[test]
fn low_float_producer_codec_preserves_raw_exception_bits_and_refuses_bad_layouts() {
    use super::CudaPhysicalLayout;
    use super::TensorOperandRepresentation;
    macro_rules! check {
        ($ty:ty, $bits:ty, $patterns:expr) => {{
            let patterns: [$bits; 6] = $patterns;
            let values = patterns.map(<$ty>::from_bits);
            let expected: Vec<u8> = patterns
                .into_iter()
                .flat_map(<$bits>::to_le_bytes)
                .collect();
            let mut graph = Graph::default();
            let constant = graph.constant_typed(Tensor::new([2, 3], values.to_vec()).unwrap());
            let node = graph.node(constant.erase()).unwrap();
            let layout = CudaPhysicalLayout::dense(u64::try_from(expected.len()).unwrap());
            assert_eq!(
                super::literal::encoded_low_float(node, layout).unwrap(),
                expected
            );
            assert!(super::cuda_supports_operand_representation(
                node,
                TensorOperandRepresentation::Dense
            ));
            assert!(!super::cuda_supports_operand_representation(
                node,
                TensorOperandRepresentation::UniformScalar
            ));
            assert!(
                super::literal::encoded_low_float(node, CudaPhysicalLayout::uniform_scalar())
                    .is_err()
            );
            assert!(
                super::literal::encoded_low_float(
                    node,
                    CudaPhysicalLayout::dense(layout.physical_bytes - 1)
                )
                .is_err()
            );
            let mut forged = node;
            forged.shape = &[1, 6];
            assert!(super::literal::encoded_low_float(forged, layout).is_err());
            assert!(matches!(
                assess_tensor_node(&graph, forged),
                TensorOperationSupport::Unsupported { .. }
            ));
            let splat = graph.uniform_typed([2, 3], values[0]).unwrap();
            assert_eq!(
                super::literal::encoded_low_float(graph.node(splat.erase()).unwrap(), layout)
                    .unwrap(),
                patterns[0].to_le_bytes().repeat(6)
            );
            let empty = graph.uniform_typed([0], values[0]).unwrap();
            assert!(matches!(
                assess_tensor_node(&graph, graph.node(empty.erase()).unwrap()),
                TensorOperationSupport::Unsupported { .. }
            ));
            graph.set_numerical_options(fusion_pcu::PcuNumericalOptions {
                reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
                ..Default::default()
            });
            let portable = graph.uniform_typed([2, 3], values[0]).unwrap();
            assert!(!super::cuda_supports_operand_representation(
                graph.node(portable.erase()).unwrap(),
                TensorOperandRepresentation::Dense
            ));
            assert!(matches!(
                assess_tensor_node(&graph, graph.node(portable.erase()).unwrap()),
                TensorOperationSupport::Unsupported { .. }
            ));
        }};
    }
    check!(
        fusion_pcu::PcuF16Bits,
        u16,
        [0x7e42, 0xfe43, 0x8000, 1, 0x7c00, 0x03ff]
    );
    check!(
        fusion_pcu::PcuBf16Bits,
        u16,
        [0x7fc2, 0xffc3, 0x8000, 1, 0x7f80, 0x007f]
    );
    check!(
        fusion_pcu::PcuF8E4M3FnBits,
        u8,
        [0x7f, 0xff, 0x80, 1, 0x7e, 7]
    );
    check!(
        fusion_pcu::PcuF8E5M2Bits,
        u8,
        [0x7f, 0xff, 0x80, 1, 0x7c, 3]
    );
}
