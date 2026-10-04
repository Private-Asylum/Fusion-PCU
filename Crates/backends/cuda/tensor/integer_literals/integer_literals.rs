//! Executable dense integer producer and representation contract witnesses.
#[rustfmt::skip]
use super::{
    assess_tensor_node,
    cuda_supports_operand_representation,
    Graph,
    Tensor,
    TensorOperandRepresentation,
    TensorOperationSupport,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuNumericalMode,
    PcuScalarType,
};

#[test]
fn integer_dense_operand_representations_match_executable_contract() {
    for scalar in [
        PcuScalarType::I8,
        PcuScalarType::U8,
        PcuScalarType::I16,
        PcuScalarType::U16,
        PcuScalarType::I32,
        PcuScalarType::U32,
        PcuScalarType::I64,
        PcuScalarType::U64,
        PcuScalarType::I128,
        PcuScalarType::U128,
        PcuScalarType::I256,
        PcuScalarType::U256,
        PcuScalarType::I512,
        PcuScalarType::U512,
    ] {
        let mut graph = Graph::default();
        let a = graph.input([65], scalar).unwrap();
        let b = graph.input([65], scalar).unwrap();
        let sum = graph.add(a, b).unwrap();
        let sub = graph.sub(sum, a).unwrap();
        let product = graph.mul(sub, b).unwrap();
        for value in [a, b, sum, sub, product] {
            let node = graph.node(value).unwrap();
            assert!(matches!(
                assess_tensor_node(&graph, node),
                TensorOperationSupport::Supported { .. }
            ));
            assert!(
                cuda_supports_operand_representation(node, TensorOperandRepresentation::Dense),
                "{scalar:?} {value:?}"
            );
            assert!(!cuda_supports_operand_representation(
                node,
                TensorOperandRepresentation::UniformScalar
            ));
        }
    }
}

#[test]
fn constant_uniform_producers_match_cpu_dense_integer_contract() {
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        let mut graph = Graph::default();
        graph.set_numerical_mode(mode);
        let x = graph.input_typed::<i32>([3]).unwrap();
        let constant = graph.constant_typed(Tensor::new([3], vec![2_i32; 3]).unwrap());
        let uniform = graph.uniform_typed([3], 3_i32).unwrap();
        let sum = graph.add_typed(x, constant).unwrap();
        let product = graph.mul_typed(sum, uniform).unwrap();
        for value in [x, constant, uniform, sum, product] {
            let node = graph.node(value.erase()).unwrap();
            assert!(
                matches!(
                    assess_tensor_node(&graph, node),
                    TensorOperationSupport::Supported { .. }
                ),
                "{mode:?} {:?}",
                node.op
            );
        }
    }
}

#[test]
fn exact_integer_literal_bytes_and_typed_disposition() {
    use fusion_pcu::{PcuNumericalOptions, PcuReproducibility, PcuScalar};
    use super::CudaPhysicalLayout;
    macro_rules! check {
        ($ty:ty, $bytes:expr) => {{
            let raw: [u8; $bytes] =
                std::array::from_fn(|i| u8::try_from((i * 37 + 0xa5) % 256).unwrap());
            let value = <$ty as PcuScalar>::decode_le(raw);
            let mut graph = Graph::default();
            let constant = graph.constant_typed(Tensor::new([2], vec![value; 2]).unwrap());
            let uniform = graph.uniform_typed([2], value).unwrap();
            for value in [constant, uniform] {
                let node = graph.node(value.erase()).unwrap();
                assert_eq!(
                    super::literal::encoded_integer(node, CudaPhysicalLayout::dense(2 * $bytes))
                        .unwrap(),
                    raw.repeat(2)
                );
                assert!(matches!(
                    assess_tensor_node(&graph, node),
                    TensorOperationSupport::Supported { .. }
                ));
                assert!(
                    super::literal::encoded_integer(
                        node,
                        CudaPhysicalLayout::dense(2 * $bytes - 1)
                    )
                    .is_err()
                );
                assert!(
                    super::literal::encoded_integer(node, CudaPhysicalLayout::uniform_scalar())
                        .is_err()
                );
                let mut wrong = node;
                wrong.shape = &[1, 2];
                if matches!(node.op, super::OpDescriptor::Constant(_)) {
                    assert!(
                        super::literal::encoded_integer(
                            wrong,
                            CudaPhysicalLayout::dense(2 * $bytes)
                        )
                        .is_err()
                    );
                }
                assert!(matches!(
                    assess_tensor_node(&graph, wrong),
                    TensorOperationSupport::Unsupported { .. }
                ));
            }
            graph.set_numerical_options(PcuNumericalOptions {
                reproducibility: PcuReproducibility::PortableV1,
                ..Default::default()
            });
            let portable = graph.uniform_typed([2], value).unwrap();
            let node = graph.node(portable.erase()).unwrap();
            assert!(matches!(
                assess_tensor_node(&graph, node),
                TensorOperationSupport::Unsupported { .. }
            ));
            assert!(!cuda_supports_operand_representation(
                node,
                TensorOperandRepresentation::Dense
            ));
            let zero = graph.uniform_typed([0], value).unwrap();
            assert!(matches!(
                assess_tensor_node(&graph, graph.node(zero.erase()).unwrap()),
                TensorOperationSupport::Unsupported { .. }
            ));
        }};
    }
    check!(i8, 1);
    check!(u8, 1);
    check!(i16, 2);
    check!(u16, 2);
    check!(i32, 4);
    check!(u32, 4);
    check!(i64, 8);
    check!(u64, 8);
    check!(i128, 16);
    check!(u128, 16);
    check!(fusion_pcu::PcuI256, 32);
    check!(fusion_pcu::PcuU256, 32);
    check!(fusion_pcu::PcuI512, 64);
    check!(fusion_pcu::PcuU512, 64);
}
