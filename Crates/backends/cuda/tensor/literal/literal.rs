//! Exact dense F64 literal transport; arithmetic policies apply only when consumed.
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
    CudaMemoryResource,
    CudaPhysicalLayout,
    CudaPhysicalRepresentation,
    CudaTensorExecutionError,
};

fn encoded_f64(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
) -> Result<Vec<u8>, CudaTensorExecutionError> {
    if node.scalar_type != fusion_pcu::PcuScalarType::F64
        || layout.representation != CudaPhysicalRepresentation::Dense
    {
        return Err(CudaTensorExecutionError::InvalidPlan(node.value));
    }
    let count = node
        .shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or(CudaTensorExecutionError::SizeOverflow)?;
    let bytes = count
        .checked_mul(size_of::<f64>())
        .ok_or(CudaTensorExecutionError::SizeOverflow)?;
    if u64::try_from(bytes).ok() != Some(layout.physical_bytes) {
        return Err(CudaTensorExecutionError::InvalidPlan(node.value));
    }
    // Encoding is bit transport: neither conversion through F32 nor host float arithmetic.
    let encoded: Vec<u8> = match node.op {
        OpDescriptor::Constant(value) => value
            .as_typed::<f64>()
            .map_err(|_| CudaTensorExecutionError::InvalidPlan(node.value))?
            .data()
            .iter()
            .flat_map(|value| value.to_bits().to_le_bytes())
            .collect(),
        OpDescriptor::Uniform { value } => std::iter::repeat_n(
            value
                .as_typed::<f64>()
                .map_err(|_| CudaTensorExecutionError::InvalidPlan(node.value))?
                .to_bits()
                .to_le_bytes(),
            count,
        )
        .flatten()
        .collect(),
        _ => return Err(CudaTensorExecutionError::InvalidPlan(node.value)),
    };
    if encoded.len() != bytes {
        return Err(CudaTensorExecutionError::InvalidPlan(node.value));
    }
    Ok(encoded)
}

pub(super) fn upload_f64<P: PcuMemoryProvider<Resource = CudaMemoryResource>>(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
    output: Option<&CudaMemoryResource>,
    pool: PcuMemoryPoolId,
    memory: &mut P,
) -> Result<CudaMemoryResource, CudaTensorExecutionError> {
    let bytes = encoded_f64(node, layout)?;
    let mut resource = if let Some(output) = output {
        output.clone_for_tensor_input()
    } else {
        super::allocate_tensor_for_size(
            memory,
            pool,
            node.shape,
            size_of::<f64>(),
            align_of::<f64>(),
        )?
    };
    if resource.size_bytes() < layout.physical_bytes {
        return Err(CudaTensorExecutionError::OutputResourceMismatch);
    }
    memory.transfer_to(&mut resource, 0, &bytes)?;
    if output.is_some_and(|output| !resource.same_binding(output)) {
        return Err(CudaTensorExecutionError::OutputResourceMismatch);
    }
    Ok(resource)
}

#[cfg(test)]
mod tests {
    use super::encoded_f64;
    #[rustfmt::skip]
    use fusion_pcu::dialect::tensor::{
        TensorScalarValue,
        TensorValue,
    };
    #[rustfmt::skip]
    use super::super::{
        assess_tensor_node,
        Graph,
        CudaPhysicalLayout,
        Tensor,
        TensorOperationSupport,
    };

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
            encoded_f64(node, CudaPhysicalLayout::dense(64)).unwrap(),
            patterns
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect::<Vec<_>>()
        );
        assert!(encoded_f64(node, CudaPhysicalLayout::dense(63)).is_err());
        assert!(encoded_f64(node, CudaPhysicalLayout::uniform_scalar()).is_err());
        let mut wrong_shape = node;
        wrong_shape.shape = &[7];
        assert!(encoded_f64(wrong_shape, CudaPhysicalLayout::dense(56)).is_err());
        let scalar = graph
            .uniform_value([], TensorScalarValue::F64(-0.0))
            .unwrap();
        assert_eq!(
            encoded_f64(graph.node(scalar).unwrap(), CudaPhysicalLayout::dense(8)).unwrap(),
            (-0.0_f64).to_bits().to_le_bytes()
        );
        let empty = graph
            .uniform_value([0], TensorScalarValue::F64(1.0))
            .unwrap();
        assert!(
            encoded_f64(graph.node(empty).unwrap(), CudaPhysicalLayout::dense(0))
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
            encoded_f64(graph.node(uniform).unwrap(), CudaPhysicalLayout::dense(136)).unwrap(),
            std::iter::repeat_n(0x3ff0_0000_0000_1000_u64.to_le_bytes(), 17)
                .flatten()
                .collect::<Vec<_>>()
        );
        let input = graph.input([8], fusion_pcu::PcuScalarType::F64).unwrap();
        assert!(encoded_f64(graph.node(input).unwrap(), CudaPhysicalLayout::dense(64)).is_err());
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
