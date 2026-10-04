//! Exact dense integer producers; no host arithmetic or scalar literal IR is involved.
#[rustfmt::skip]
use fusion_pcu::{
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
    PcuScalar,
    PcuScalarType,
};
use fusion_pcu::dialect::tensor::TensorElement;
#[rustfmt::skip]
use super::super::{
    CudaMemoryResource,
    CudaPhysicalLayout,
    CudaPhysicalRepresentation,
    CudaTensorExecutionError,
    NodeDescriptor,
    OpDescriptor,
};

pub(super) fn encoded<T: PcuScalar + TensorElement>(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
) -> Result<Vec<u8>, CudaTensorExecutionError> {
    if node.scalar_type != <T as PcuScalar>::TYPE
        || layout.representation != CudaPhysicalRepresentation::Dense
    {
        return Err(CudaTensorExecutionError::InvalidPlan(node.value));
    }
    let count = node
        .shape
        .iter()
        .try_fold(1_usize, |n, &d| n.checked_mul(d))
        .ok_or(CudaTensorExecutionError::SizeOverflow)?;
    let bytes = count
        .checked_mul(T::ENCODED_SIZE)
        .ok_or(CudaTensorExecutionError::SizeOverflow)?;
    if u64::try_from(bytes).ok() != Some(layout.physical_bytes) {
        return Err(CudaTensorExecutionError::InvalidPlan(node.value));
    }
    let mut result = Vec::with_capacity(bytes);
    match node.op {
        OpDescriptor::Constant(value) => {
            let tensor = value
                .as_typed::<T>()
                .map_err(|_| CudaTensorExecutionError::InvalidPlan(node.value))?;
            if tensor.shape() != node.shape {
                return Err(CudaTensorExecutionError::InvalidPlan(node.value));
            }
            for &value in tensor.data() {
                result.extend_from_slice(value.encode_le().as_ref());
            }
        }
        OpDescriptor::Uniform { value } => {
            let value = value
                .as_typed::<T>()
                .map_err(|_| CudaTensorExecutionError::InvalidPlan(node.value))?
                .encode_le();
            for _ in 0..count {
                result.extend_from_slice(value.as_ref());
            }
        }
        _ => return Err(CudaTensorExecutionError::InvalidPlan(node.value)),
    }
    if result.len() != bytes {
        return Err(CudaTensorExecutionError::InvalidPlan(node.value));
    }
    Ok(result)
}

pub(in super::super) fn encoded_integer(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
) -> Result<Vec<u8>, CudaTensorExecutionError> {
    match node.scalar_type {
        PcuScalarType::I8 => encoded::<i8>(node, layout),
        PcuScalarType::U8 => encoded::<u8>(node, layout),
        PcuScalarType::I16 => encoded::<i16>(node, layout),
        PcuScalarType::U16 => encoded::<u16>(node, layout),
        PcuScalarType::I32 => encoded::<i32>(node, layout),
        PcuScalarType::U32 => encoded::<u32>(node, layout),
        PcuScalarType::I64 => encoded::<i64>(node, layout),
        PcuScalarType::U64 => encoded::<u64>(node, layout),
        PcuScalarType::I128 => encoded::<i128>(node, layout),
        PcuScalarType::U128 => encoded::<u128>(node, layout),
        PcuScalarType::I256 => encoded::<fusion_pcu::PcuI256>(node, layout),
        PcuScalarType::U256 => encoded::<fusion_pcu::PcuU256>(node, layout),
        PcuScalarType::I512 => encoded::<fusion_pcu::PcuI512>(node, layout),
        PcuScalarType::U512 => encoded::<fusion_pcu::PcuU512>(node, layout),
        _ => Err(CudaTensorExecutionError::InvalidPlan(node.value)),
    }
}

pub(in super::super) fn upload_integer<P: PcuMemoryProvider<Resource = CudaMemoryResource>>(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
    output: Option<&CudaMemoryResource>,
    pool: PcuMemoryPoolId,
    memory: &mut P,
) -> Result<CudaMemoryResource, CudaTensorExecutionError> {
    let bytes = encoded_integer(node, layout)?;
    let (size, alignment) = super::super::scalar_layout(node.scalar_type)?;
    let mut resource = if let Some(output) = output {
        output.clone_for_tensor_input()
    } else {
        super::super::allocate_tensor_for_size(
            memory,
            pool,
            node.shape,
            size,
            usize::try_from(alignment).map_err(|_| CudaTensorExecutionError::SizeOverflow)?,
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
