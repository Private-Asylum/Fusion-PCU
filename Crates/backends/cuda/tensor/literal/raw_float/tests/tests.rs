#[rustfmt::skip]
use fusion_pcu::{
    PcuF128Bits,
    PcuF256Bits,
    PcuScalarType,
    PcuNumericalOptions,
    PcuReproducibility,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    Tensor,
    TensorScalarValue,
    TensorValue,
    TensorOperationSupport,
};
#[rustfmt::skip]
use super::super::super::{
    assess_tensor_node,
    cuda_supports_operand_representation,
    CudaPhysicalLayout,
};
use fusion_pcu::dialect::tensor::TensorOperandRepresentation;
use super::encoded_raw_float;

#[test]
fn real_wide_float_producers_admit_dense_bits_and_preserve_other_refusals() {
    let mut graph = Graph::default();
    let wide128 = PcuF128Bits::from_limbs_le([0x42, 0x7fff_0000_0000_0000]);
    let wide256 = PcuF256Bits::from_limbs_le([0x42, 0, 0, 0x7fff_f000_0000_0000]);
    for (scalar, value, uniform) in [
        (
            PcuScalarType::F128,
            TensorValue::F128(Tensor::new([3], vec![wide128; 3]).unwrap()),
            TensorScalarValue::F128(wide128),
        ),
        (
            PcuScalarType::F256,
            TensorValue::F256(Tensor::new([3], vec![wide256; 3]).unwrap()),
            TensorScalarValue::F256(wide256),
        ),
    ] {
        let constant = graph.constant_value(value);
        let splat = graph.uniform_value([3], uniform).unwrap();
        let input = graph.input([3], scalar).unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, graph.node(input).unwrap()),
            TensorOperationSupport::Supported { .. }
        ));
        for producer in [constant, splat] {
            assert!(matches!(
                assess_tensor_node(&graph, graph.node(producer).unwrap()),
                TensorOperationSupport::Supported { .. }
            ));
            let node = graph.node(producer).unwrap();
            assert!(cuda_supports_operand_representation(
                node,
                TensorOperandRepresentation::Dense
            ));
            assert!(!cuda_supports_operand_representation(
                node,
                TensorOperandRepresentation::UniformScalar
            ));
            let mut forged = node;
            forged.shape = &[2];
            assert!(matches!(
                assess_tensor_node(&graph, forged),
                TensorOperationSupport::Unsupported { .. }
            ));
        }
        // Core itself excludes arithmetic over these opaque float encodings.
        assert!(graph.add(constant, splat).is_err());
        assert!(graph.mul(constant, splat).is_err());
        let empty = graph.uniform_value([0], uniform).unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, graph.node(empty).unwrap()),
            TensorOperationSupport::Unsupported { .. }
        ));
        graph.set_numerical_options(PcuNumericalOptions {
            reproducibility: PcuReproducibility::PortableV1,
            ..Default::default()
        });
        let portable = graph.uniform_value([3], uniform).unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, graph.node(portable).unwrap()),
            TensorOperationSupport::Unsupported { .. }
        ));
        graph.set_numerical_options(PcuNumericalOptions::default());
    }
}

#[test]
fn raw_float_codec_preserves_every_limb_and_rejects_forged_geometry() {
    let patterns128 = [
        [0, 0],
        [0, 0x8000_0000_0000_0000],
        [1, 0],
        [0x42, 0x7fff_0000_0000_0000],
        [0xdead_beef, 0xffff_8000_0000_0000],
        [u64::MAX, 0x3ffe_0123_4567_89ab],
    ];
    let patterns256 = [
        [0, 0, 0, 0],
        [0, 0, 0, 0x8000_0000_0000_0000],
        [1, 0, 0, 0],
        [0x42, 0, 0, 0x7fff_f000_0000_0000],
        [0xdead_beef, 1, 2, 0xffff_f800_0000_0000],
        [
            u64::MAX,
            0x1234_5678_9abc_def0,
            0xfedc_ba98_7654_3210,
            0x3fff_e012_3456_789a,
        ],
    ];
    let mut graph = Graph::default();
    for (value, uniform, bytes, width) in [
        (
            TensorValue::F128(
                Tensor::new([6], patterns128.map(PcuF128Bits::from_limbs_le).to_vec()).unwrap(),
            ),
            TensorScalarValue::F128(PcuF128Bits::from_limbs_le(patterns128[3])),
            patterns128
                .into_iter()
                .flatten()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
            16,
        ),
        (
            TensorValue::F256(
                Tensor::new([6], patterns256.map(PcuF256Bits::from_limbs_le).to_vec()).unwrap(),
            ),
            TensorScalarValue::F256(PcuF256Bits::from_limbs_le(patterns256[3])),
            patterns256
                .into_iter()
                .flatten()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>(),
            32,
        ),
    ] {
        let matrix = match &value {
            TensorValue::F128(value) => {
                TensorValue::F128(Tensor::new([2, 3], value.data().to_vec()).unwrap())
            }
            TensorValue::F256(value) => {
                TensorValue::F256(Tensor::new([2, 3], value.data().to_vec()).unwrap())
            }
            _ => unreachable!(),
        };
        let constant = graph.constant_value(value);
        let node = graph.node(constant).unwrap();
        assert_eq!(
            encoded_raw_float(node, CudaPhysicalLayout::dense(bytes.len() as u64)).unwrap(),
            bytes
        );
        assert!(
            encoded_raw_float(node, CudaPhysicalLayout::dense(bytes.len() as u64 - 1)).is_err()
        );
        assert!(encoded_raw_float(node, CudaPhysicalLayout::uniform_scalar()).is_err());
        let mut forged = node;
        forged.shape = &[3, 2];
        assert!(encoded_raw_float(forged, CudaPhysicalLayout::dense(bytes.len() as u64)).is_err());
        forged.scalar_type = PcuScalarType::U32;
        assert!(encoded_raw_float(forged, CudaPhysicalLayout::dense(bytes.len() as u64)).is_err());
        let matrix = graph.constant_value(matrix);
        let matrix_node = graph.node(matrix).unwrap();
        assert_eq!(matrix_node.shape, [2, 3]);
        assert_eq!(
            encoded_raw_float(matrix_node, CudaPhysicalLayout::dense(bytes.len() as u64)).unwrap(),
            bytes
        );
        assert!(matches!(
            assess_tensor_node(&graph, matrix_node),
            TensorOperationSupport::Supported { .. }
        ));
        let splat = graph.uniform_value([6], uniform).unwrap();
        let expected = bytes[3 * width..4 * width].repeat(6);
        assert_eq!(
            encoded_raw_float(
                graph.node(splat).unwrap(),
                CudaPhysicalLayout::dense(bytes.len() as u64)
            )
            .unwrap(),
            expected
        );
        let empty = graph.uniform_value([0], uniform).unwrap();
        assert!(
            encoded_raw_float(graph.node(empty).unwrap(), CudaPhysicalLayout::dense(0))
                .unwrap()
                .is_empty()
        );
    }
}
