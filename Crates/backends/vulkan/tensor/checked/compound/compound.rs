//! Cold exact shapes and unequal native spans for the ordered Strict compound table.
#[rustfmt::skip]
use fusion_pcu::{
    PcuNumericalMode,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    NodeDescriptor,
    OpDescriptor,
    TensorError,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_checked_compound_to_spirv,
    PcuSpirvCompoundOperation,
    PcuSpirvLoweringOptions,
};
#[rustfmt::skip]
use super::{
    Kernel,
    Slot,
    Step,
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanTensorError,
    VulkanPreparedTensorMap,
    TensorStatusPolicy,
};

pub(super) fn assess(node: NodeDescriptor<'_>) -> Result<(), PcuVulkanTensorError> {
    if !matches!(node.scalar_type, PcuScalarType::F32 | PcuScalarType::F64) {
        return Err(PcuVulkanTensorError::UnsupportedNode(node.value));
    }
    if node.numerical_mode != Some(PcuNumericalMode::Strict) {
        return Err(TensorError::UnsupportedNumericalMode {
            value: node.value,
            mode: node.numerical_mode.unwrap_or(PcuNumericalMode::Boundary),
        }
        .into());
    }
    Ok(())
}

pub(super) fn prepare(
    backend: &PcuVulkanBackend,
    node: NodeDescriptor<'_>,
    output: usize,
    inputs: [usize; 2],
    slots: &[Slot],
    nodes: &[NodeDescriptor<'_>],
) -> Result<Step, PcuVulkanTensorError> {
    assess(node)?;
    let operation = operation(
        node,
        [nodes[inputs[0]], nodes[inputs[1]]],
        slots[output].count,
    )?;
    let mut words = Vec::new();
    let (_, profile) = lower_checked_compound_to_spirv(
        node.scalar_type,
        operation,
        node.float_underflow_policy.unwrap_or_default(),
        PcuSpirvLoweringOptions::minimal_shader(),
        &mut words,
    )
    .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
    let lengths = profile.binding_bytes.map(|length| length as usize);
    for (slot, bytes) in [inputs[0], inputs[1], output].into_iter().zip(lengths) {
        if slots[slot].native.byte_len() != bytes {
            return Err(PcuVulkanError::InvalidArguments.into());
        }
    }
    let kernel = VulkanPreparedTensorMap::new(
        &backend.device,
        &words,
        profile.output_count,
        profile.output_count,
        [64, 1, 1],
        lengths,
        TensorStatusPolicy::compound(
            node.scalar_type,
            operation,
            node.float_underflow_policy.unwrap_or_default(),
        )?,
    )?;
    Ok(Step {
        kernel: Kernel::Compound(kernel),
        value: node.value,
        inputs,
        output,
    })
}

fn size(value: usize) -> Result<u32, PcuVulkanTensorError> {
    u32::try_from(value).map_err(|_| PcuVulkanError::BufferTooLarge.into())
}
fn operation(
    node: NodeDescriptor<'_>,
    inputs: [NodeDescriptor<'_>; 2],
    count: usize,
) -> Result<PcuSpirvCompoundOperation, PcuVulkanTensorError> {
    let [left, right] = inputs;
    Ok(match node.op {
        OpDescriptor::SgdUpdate { learning_rate, .. } => {
            if left.shape != node.shape || right.shape != node.shape {
                return Err(PcuVulkanError::InvalidArguments.into());
            }
            PcuSpirvCompoundOperation::Sgd {
                count: size(count)?,
                learning_rate,
            }
        }
        OpDescriptor::MeanSquaredError { .. } => {
            if left.shape != right.shape || !node.shape.is_empty() {
                return Err(PcuVulkanError::InvalidArguments.into());
            }
            let count = left
                .shape
                .iter()
                .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
                .ok_or(TensorError::ShapeOverflow)?;
            PcuSpirvCompoundOperation::MeanSquaredError {
                count: size(count)?,
            }
        }
        OpDescriptor::MatMul {
            transpose_left,
            transpose_right,
            ..
        } => {
            if left.shape.len() != 2 || right.shape.len() != 2 {
                return Err(TensorError::MatMulShape {
                    left: left.shape.to_vec(),
                    right: right.shape.to_vec(),
                }
                .into());
            }
            let rows = left.shape[usize::from(transpose_left)];
            let inner = left.shape[usize::from(!transpose_left)];
            let columns = right.shape[usize::from(!transpose_right)];
            if inner != right.shape[usize::from(transpose_right)] || node.shape != [rows, columns] {
                return Err(PcuVulkanError::InvalidArguments.into());
            }
            PcuSpirvCompoundOperation::MatMul {
                rows: size(rows)?,
                inner: size(inner)?,
                columns: size(columns)?,
                transpose_left,
                transpose_right,
            }
        }
        _ => return Err(PcuVulkanTensorError::UnsupportedNode(node.value)),
    })
}
