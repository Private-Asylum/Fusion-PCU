//! Canonical wide float encodings are immutable data, without float arithmetic.
#[rustfmt::skip]
use fusion_pcu::{
    PcuF128Bits,
    PcuF256Bits,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
    PcuScalarType,
};
#[rustfmt::skip]
use super::super::{
    CudaMemoryResource,
    CudaPhysicalLayout,
    CudaTensorExecutionError,
    NodeDescriptor,
};

pub(in super::super) fn encoded_raw_float(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
) -> Result<Vec<u8>, CudaTensorExecutionError> {
    match node.scalar_type {
        PcuScalarType::F128 => super::integer::encoded::<PcuF128Bits>(node, layout),
        PcuScalarType::F256 => super::integer::encoded::<PcuF256Bits>(node, layout),
        _ => Err(CudaTensorExecutionError::InvalidPlan(node.value)),
    }
}

pub(in super::super) fn upload_raw_float<P: PcuMemoryProvider<Resource = CudaMemoryResource>>(
    node: NodeDescriptor<'_>,
    layout: CudaPhysicalLayout,
    output: Option<&CudaMemoryResource>,
    pool: PcuMemoryPoolId,
    memory: &mut P,
) -> Result<CudaMemoryResource, CudaTensorExecutionError> {
    let bytes = encoded_raw_float(node, layout)?;
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

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
