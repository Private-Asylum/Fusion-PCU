//! Exact dense literal transport; arithmetic policies apply only when consumed.
use fusion_pcu::dialect::tensor::TensorElement;
#[path = "raw_float/raw_float.rs"]
pub(super) mod raw_float;
#[path = "transport/transport.rs"]
mod transport;
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
};
#[rustfmt::skip]
use super::{
    NodeDescriptor,
    OpDescriptor,
    RocmMemoryResource,
    RocmPhysicalLayout,
    RocmPhysicalRepresentation,
    RocmTensorExecutionError,
};

fn encoded_f64(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Vec<u8>, RocmTensorExecutionError> {
    if node.scalar_type != fusion_pcu::PcuScalarType::F64
        || layout.representation != RocmPhysicalRepresentation::Dense
    {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    let count = node
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    let bytes = count
        .checked_mul(size_of::<f64>())
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    if u64::try_from(bytes).ok() != Some(layout.physical_bytes) {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    // Encoding is bit transport: neither conversion through F32 nor host float arithmetic.
    let encoded: Vec<u8> = match node.op {
        OpDescriptor::Constant(value) => value
            .as_typed::<f64>()
            .map_err(|_| RocmTensorExecutionError::InvalidPlan(node.value))?
            .data()
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
        OpDescriptor::Uniform { value } => std::iter::repeat_n(
            value
                .as_typed::<f64>()
                .map_err(|_| RocmTensorExecutionError::InvalidPlan(node.value))?
                .to_bits()
                .to_le_bytes(),
            count,
        )
        .flatten()
        .collect(),
        _ => return Err(RocmTensorExecutionError::InvalidPlan(node.value)),
    };
    if encoded.len() != bytes {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    Ok(encoded)
}

fn encoded_typed<T: TensorElement>(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Vec<u8>, RocmTensorExecutionError> {
    if node.scalar_type != T::TYPE || layout.representation != RocmPhysicalRepresentation::Dense {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    let count = node
        .shape
        .iter()
        .try_fold(1usize, |count, &dimension| count.checked_mul(dimension))
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    let bytes = count
        .checked_mul(T::ENCODED_SIZE)
        .ok_or(RocmTensorExecutionError::SizeOverflow)?;
    if u64::try_from(bytes).ok() != Some(layout.physical_bytes) {
        return Err(RocmTensorExecutionError::InvalidPlan(node.value));
    }
    let mut encoded = Vec::with_capacity(bytes);
    match node.op {
        OpDescriptor::Constant(value) => {
            let tensor = value
                .as_typed::<T>()
                .map_err(|_| RocmTensorExecutionError::InvalidPlan(node.value))?;
            if tensor.shape() != node.shape || tensor.data().len() != count {
                return Err(RocmTensorExecutionError::InvalidPlan(node.value));
            }
            for &value in tensor.data() {
                encoded.extend_from_slice(value.encode_le().as_ref());
            }
        }
        OpDescriptor::Uniform { value } => {
            let value = value
                .as_typed::<T>()
                .map_err(|_| RocmTensorExecutionError::InvalidPlan(node.value))?
                .encode_le();
            for _ in 0..count {
                encoded.extend_from_slice(value.as_ref());
            }
        }
        _ => return Err(RocmTensorExecutionError::InvalidPlan(node.value)),
    }
    Ok(encoded)
}

// The closed match deliberately excludes floating and compact integer representations.
pub(super) fn encoded_integer(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Vec<u8>, RocmTensorExecutionError> {
    use fusion_pcu::PcuScalarType;
    match node.scalar_type {
        PcuScalarType::U8 => encoded_typed::<u8>(node, layout),
        PcuScalarType::I8 => encoded_typed::<i8>(node, layout),
        PcuScalarType::U16 => encoded_typed::<u16>(node, layout),
        PcuScalarType::I16 => encoded_typed::<i16>(node, layout),
        PcuScalarType::U32 => encoded_typed::<u32>(node, layout),
        PcuScalarType::I32 => encoded_typed::<i32>(node, layout),
        PcuScalarType::U64 => encoded_typed::<u64>(node, layout),
        PcuScalarType::I64 => encoded_typed::<i64>(node, layout),
        PcuScalarType::U128 => encoded_typed::<u128>(node, layout),
        PcuScalarType::I128 => encoded_typed::<i128>(node, layout),
        PcuScalarType::U256 => encoded_typed::<fusion_pcu::PcuU256>(node, layout),
        PcuScalarType::I256 => encoded_typed::<fusion_pcu::PcuI256>(node, layout),
        PcuScalarType::U512 => encoded_typed::<fusion_pcu::PcuU512>(node, layout),
        PcuScalarType::I512 => encoded_typed::<fusion_pcu::PcuI512>(node, layout),
        _ => Err(RocmTensorExecutionError::InvalidPlan(node.value)),
    }
}

// Raw low-format transport preserves every payload, including NaNs and signed zero.
pub(super) fn encoded_low_float(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
) -> Result<Vec<u8>, RocmTensorExecutionError> {
    use fusion_pcu::PcuScalarType;
    match node.scalar_type {
        PcuScalarType::F16 => encoded_typed::<fusion_pcu::PcuF16Bits>(node, layout),
        PcuScalarType::BF16 => encoded_typed::<fusion_pcu::PcuBf16Bits>(node, layout),
        PcuScalarType::F8E4M3FN => encoded_typed::<fusion_pcu::PcuF8E4M3FnBits>(node, layout),
        PcuScalarType::F8E5M2 => encoded_typed::<fusion_pcu::PcuF8E5M2Bits>(node, layout),
        _ => Err(RocmTensorExecutionError::InvalidPlan(node.value)),
    }
}

pub(super) fn upload_dense<P: PcuMemoryProvider<Resource = RocmMemoryResource>>(
    node: NodeDescriptor<'_>,
    layout: RocmPhysicalLayout,
    output: Option<&RocmMemoryResource>,
    pool: PcuMemoryPoolId,
    memory: &mut P,
) -> Result<RocmMemoryResource, RocmTensorExecutionError> {
    let bytes = transport::payload(node, layout)?;
    let (element_size, alignment) = super::scalar_layout(node.scalar_type)?;
    let alignment =
        usize::try_from(alignment).map_err(|_| RocmTensorExecutionError::SizeOverflow)?;
    let mut resource = if let Some(output) = output {
        output.clone_for_tensor_input()
    } else {
        super::allocate_tensor_for_size(memory, pool, node.shape, element_size, alignment)?
    };
    if resource.size_bytes() < layout.physical_bytes {
        return Err(RocmTensorExecutionError::OutputResourceMismatch);
    }
    memory.transfer_to(&mut resource, 0, bytes.bytes())?;
    if output.is_some_and(|output| !resource.same_binding(output)) {
        return Err(RocmTensorExecutionError::OutputResourceMismatch);
    }
    Ok(resource)
}

#[cfg(test)]
mod tests {
    use super::encoded_f64;
    use super::encoded_integer;
    use fusion_pcu::dialect::tensor::TensorElement;
    #[rustfmt::skip]
    use fusion_pcu::dialect::tensor::{
        TensorScalarValue,
        TensorValue,
    };
    #[rustfmt::skip]
    use super::super::{
        assess_tensor_node,
        Graph,
        RocmPhysicalLayout,
        Tensor,
        TensorOperationSupport,
    };

    fn integer_bits<T: TensorElement>(values: [T; 2]) {
        let mut graph = Graph::default();
        let constant = graph.constant_typed(Tensor::new([2], values.to_vec()).unwrap());
        let node = graph.node(constant.erase()).unwrap();
        let layout = RocmPhysicalLayout::dense(u64::try_from(2 * T::ENCODED_SIZE).unwrap());
        let expected = values
            .into_iter()
            .flat_map(|value| value.encode_le().as_ref().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(encoded_integer(node, layout).unwrap(), expected);
        assert!(encoded_integer(node, RocmPhysicalLayout::uniform_scalar()).is_err());
        let mut forged = node;
        forged.shape = &[1, 2];
        assert!(encoded_integer(forged, layout).is_err());
        forged.scalar_type = fusion_pcu::PcuScalarType::F32;
        assert!(encoded_integer(forged, layout).is_err());
        let uniform = graph.uniform_typed([2], values[1]).unwrap();
        assert_eq!(
            encoded_integer(graph.node(uniform.erase()).unwrap(), layout).unwrap(),
            values[1].encode_le().as_ref().repeat(2)
        );
    }

    #[test]
    fn all_integer_literals_preserve_extrema_and_exact_shape() {
        macro_rules! width {
            ($($ty:ty),+ $(,)?) => { $(integer_bits([<$ty>::MIN, <$ty>::MAX]);)+ };
        }
        width!(
            u8,
            i8,
            u16,
            i16,
            u32,
            i32,
            u64,
            i64,
            u128,
            i128,
            fusion_pcu::PcuI256,
            fusion_pcu::PcuI512
        );
        integer_bits([fusion_pcu::PcuU256::ZERO, fusion_pcu::PcuU256::MAX]);
        integer_bits([fusion_pcu::PcuU512::ZERO, fusion_pcu::PcuU512::MAX]);
    }

    #[test]
    fn dense_f64_literals_encode_raw_bits_and_reject_bad_layouts() {
        let patterns = [
            0,
            0x8000_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0x7ff0_0000_0000_1234,
            0xfff8_0000_0000_5678,
            1,
            0x8000_0000_0000_0001,
            0x3ff0_0000_0000_1000,
        ];
        let mut graph = Graph::default();
        let constant = graph.constant_value(TensorValue::F64(
            Tensor::new([patterns.len()], patterns.map(f64::from_bits).to_vec()).unwrap(),
        ));
        let node = graph.node(constant).unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, node),
            TensorOperationSupport::Supported { .. }
        ));
        assert_eq!(
            encoded_f64(node, RocmPhysicalLayout::dense(64)).unwrap(),
            patterns
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(encoded_f64(node, RocmPhysicalLayout::dense(63)).is_err());
        assert!(encoded_f64(node, RocmPhysicalLayout::uniform_scalar()).is_err());
        let mut wrong_shape = node;
        wrong_shape.shape = &[7];
        assert!(encoded_f64(wrong_shape, RocmPhysicalLayout::dense(56)).is_err());
        let scalar = graph
            .uniform_value([], TensorScalarValue::F64(-0.0))
            .unwrap();
        assert_eq!(
            encoded_f64(graph.node(scalar).unwrap(), RocmPhysicalLayout::dense(8)).unwrap(),
            (-0.0_f64).to_bits().to_le_bytes()
        );
        let empty = graph
            .uniform_value([0], TensorScalarValue::F64(1.0))
            .unwrap();
        assert!(
            encoded_f64(graph.node(empty).unwrap(), RocmPhysicalLayout::dense(0))
                .unwrap()
                .is_empty()
        );
        assert!(matches!(
            assess_tensor_node(&graph, graph.node(empty).unwrap()),
            TensorOperationSupport::Unsupported { .. }
        ));

        let uniform = graph
            .uniform_value(
                [17],
                TensorScalarValue::F64(f64::from_bits(0x3ff0_0000_0000_1000)),
            )
            .unwrap();
        assert_eq!(
            encoded_f64(graph.node(uniform).unwrap(), RocmPhysicalLayout::dense(136)).unwrap(),
            std::iter::repeat_n(0x3ff0_0000_0000_1000_u64.to_le_bytes(), 17)
                .flatten()
                .collect::<Vec<_>>()
        );
        let input = graph.input([8], fusion_pcu::PcuScalarType::F64).unwrap();
        assert!(encoded_f64(graph.node(input).unwrap(), RocmPhysicalLayout::dense(64)).is_err());
        graph.set_numerical_options(fusion_pcu::PcuNumericalOptions {
            reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
            ..Default::default()
        });
        let portable = graph
            .uniform_value([1], TensorScalarValue::F64(1.0))
            .unwrap();
        assert!(matches!(
            assess_tensor_node(&graph, graph.node(portable).unwrap()),
            TensorOperationSupport::Unsupported { .. }
        ));
    }
}
