//! Exact pointwise type/operation admission and cold ordinary dispatch lowering.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedScalarFaultLaw,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchEntryPoint,
    PcuDispatchFeatureCaps,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIndex,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuDispatchValueId,
    PcuImplementationRequirements,
    PcuKernelId,
    PcuNumericalMode,
    PcuRangePolicy,
    PcuReproducibility,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    NodeDescriptor,
    OpDescriptor,
    TensorError,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    lower_checked_float_binary_to_spirv,
    lower_checked_float_unary_to_spirv,
    lower_checked_integer_to_spirv,
    lower_checked_relu_backward_to_spirv,
    PcuSpirvLoweringOptions,
};
#[rustfmt::skip]
use crate::{
    PcuVulkanError,
    PcuVulkanTensorError,
};

pub(super) struct Lowered {
    pub words: Vec<u32>,
    pub unary: bool,
    pub dispatch_extent: u32,
    pub local_size: [u32; 3],
    pub fault_law: PcuCheckedScalarFaultLaw,
}

pub(super) const fn float(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::F16
            | PcuScalarType::BF16
            | PcuScalarType::F8E4M3FN
            | PcuScalarType::F8E5M2
            | PcuScalarType::F32
            | PcuScalarType::F64
    )
}
pub(super) const fn integer(scalar: PcuScalarType) -> bool {
    matches!(
        scalar,
        PcuScalarType::I8
            | PcuScalarType::U8
            | PcuScalarType::I16
            | PcuScalarType::U16
            | PcuScalarType::I32
            | PcuScalarType::U32
            | PcuScalarType::I64
            | PcuScalarType::U64
            | PcuScalarType::I128
            | PcuScalarType::U128
            | PcuScalarType::I256
            | PcuScalarType::U256
            | PcuScalarType::I512
            | PcuScalarType::U512
    )
}

pub(super) fn assess(node: NodeDescriptor<'_>) -> Result<(), PcuVulkanTensorError> {
    if node.numerical_options.reproducibility != PcuReproducibility::Unspecified {
        return Err(TensorError::UnsupportedNumericalOptions {
            value: node.value,
            options: node.numerical_options,
        }
        .into());
    }
    let admitted = match node.op {
        OpDescriptor::Input | OpDescriptor::Constant(_) | OpDescriptor::Uniform { .. } => true,
        OpDescriptor::Add { .. } | OpDescriptor::Sub { .. } | OpDescriptor::Mul { .. } => {
            float(node.scalar_type) || integer(node.scalar_type)
        }
        OpDescriptor::Div { .. }
        | OpDescriptor::Relu { .. }
        | OpDescriptor::ReluBackward { .. } => float(node.scalar_type),
        OpDescriptor::SgdUpdate { .. }
        | OpDescriptor::MatMul { .. }
        | OpDescriptor::MeanSquaredError { .. } => return super::compound::assess(node),
    };
    if !admitted {
        return Err(PcuVulkanTensorError::UnsupportedNode(node.value));
    }
    Ok(())
}

fn arithmetic(node: NodeDescriptor<'_>) -> Result<PcuDispatchDataOp, PcuVulkanTensorError> {
    let float_op = match node.op {
        OpDescriptor::Add { .. } => PcuDispatchFloatBinaryOp::Add,
        OpDescriptor::Sub { .. } => PcuDispatchFloatBinaryOp::Sub,
        OpDescriptor::Mul { .. } => PcuDispatchFloatBinaryOp::Mul,
        OpDescriptor::Div { .. } => PcuDispatchFloatBinaryOp::Div,
        OpDescriptor::Relu { .. } => {
            return Ok(PcuDispatchDataOp::CheckedFloatUnary {
                value_type: PcuValueType::Scalar(node.scalar_type),
                op: PcuDispatchFloatUnaryOp::Relu,
                range_policy: PcuRangePolicy::Reject,
                underflow_policy: node.float_underflow_policy.unwrap_or_default(),
                result: PcuDispatchValueId(3),
                value: PcuDispatchValueId(1),
            });
        }
        _ => return Err(PcuVulkanTensorError::UnsupportedNode(node.value)),
    };
    Ok(if float(node.scalar_type) {
        PcuDispatchDataOp::CheckedFloatBinary {
            value_type: PcuValueType::Scalar(node.scalar_type),
            op: float_op,
            range_policy: PcuRangePolicy::Reject,
            underflow_policy: node.float_underflow_policy.unwrap_or_default(),
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }
    } else {
        PcuDispatchDataOp::CheckedIntegerBinary {
            value_type: PcuValueType::Scalar(node.scalar_type),
            op: match float_op {
                PcuDispatchFloatBinaryOp::Add => PcuDispatchIntegerBinaryOp::Add,
                PcuDispatchFloatBinaryOp::Sub => PcuDispatchIntegerBinaryOp::Sub,
                PcuDispatchFloatBinaryOp::Mul => PcuDispatchIntegerBinaryOp::Mul,
                PcuDispatchFloatBinaryOp::Div => {
                    return Err(PcuVulkanTensorError::UnsupportedNode(node.value));
                }
            },
            range_policy: PcuRangePolicy::Reject,
            result: PcuDispatchValueId(3),
            lhs: PcuDispatchValueId(1),
            rhs: PcuDispatchValueId(2),
        }
    })
}

pub(super) fn lower(
    node: NodeDescriptor<'_>,
    extent: u32,
) -> Result<Lowered, PcuVulkanTensorError> {
    if matches!(node.op, OpDescriptor::ReluBackward { .. }) {
        return backward(node, extent);
    }
    lower_map(node, extent)
}

fn lower_map(node: NodeDescriptor<'_>, extent: u32) -> Result<Lowered, PcuVulkanTensorError> {
    let unary = matches!(node.op, OpDescriptor::Relu { .. });
    let count = if unary { 2 } else { 3 };
    let bindings: Vec<_> = (0..count)
        .map(|index| {
            PcuBinding::value(
                None,
                0,
                index,
                PcuBindingStorageClass::Storage,
                if index == count - 1 {
                    PcuBindingAccess::ReadWrite
                } else {
                    PcuBindingAccess::ReadOnly
                },
                PcuValueType::Scalar(node.scalar_type),
            )
        })
        .collect();
    let load = |index| {
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(if index == 0 { 1 } else { 2 }),
            binding: PcuBindingRef::new(0, index),
            index: PcuDispatchIndex::InvocationId,
        })
    };
    let mut ops = vec![load(0)];
    if !unary {
        ops.push(load(1));
    }
    ops.extend([
        PcuDispatchOp::Data(arithmetic(node)?),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, count - 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(3),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ]);
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuImplementationRequirements {
            numerical_mode: node.numerical_mode.unwrap_or(PcuNumericalMode::Boundary),
            numerical_options: node.numerical_options,
            float_underflow: node.float_underflow_policy.unwrap_or_default(),
            range_policy: PcuRangePolicy::Reject,
        },
        id: PcuKernelId(0),
        entry: PcuDispatchEntryPoint {
            name: "checked_tensor_pointwise",
            logical_shape: [extent, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &ops,
        type_caps: PcuValueTypeCaps::for_scalar(node.scalar_type),
        feature_caps: PcuDispatchFeatureCaps::default(),
    };
    lower_words(&kernel, node.scalar_type, unary)
}

fn lower_words(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
    unary: bool,
) -> Result<Lowered, PcuVulkanTensorError> {
    let mut words = Vec::new();
    let options = PcuSpirvLoweringOptions::minimal_shader();
    let (dispatch_extent, local_size, fault_law) = if unary {
        let (_, profile) = lower_checked_float_unary_to_spirv(kernel, options, &mut words)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        (
            profile.dispatch_extent(),
            profile.local_size,
            PcuCheckedScalarFaultLaw::float_unary(
                profile.scalar,
                profile.operation,
                profile.range,
                profile.underflow,
            ),
        )
    } else if float(scalar) {
        let (_, profile) = lower_checked_float_binary_to_spirv(kernel, options, &mut words)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        (
            profile.dispatch_extent(),
            profile.local_size,
            PcuCheckedScalarFaultLaw::float_binary(
                profile.scalar,
                profile.operation,
                profile.range,
                profile.underflow,
            ),
        )
    } else {
        let (_, profile) = lower_checked_integer_to_spirv(kernel, options, &mut words)
            .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
        (
            profile.dispatch_extent(),
            profile.local_size,
            PcuCheckedScalarFaultLaw::integer_binary(
                profile.scalar,
                profile.operation,
                profile.range,
            ),
        )
    };
    Ok(Lowered {
        words,
        unary,
        dispatch_extent,
        local_size,
        fault_law: fault_law.ok_or(PcuVulkanError::UnsupportedPreparedProfile)?,
    })
}

fn backward(node: NodeDescriptor<'_>, extent: u32) -> Result<Lowered, PcuVulkanTensorError> {
    let mut words = Vec::new();
    lower_checked_relu_backward_to_spirv(
        node.scalar_type,
        node.float_underflow_policy.unwrap_or_default(),
        extent,
        PcuSpirvLoweringOptions::minimal_shader(),
        &mut words,
    )
    .map_err(|error| PcuVulkanError::SpirvLowering { error })?;
    let lanes = match node.scalar_type.bit_width() {
        8 => 4,
        16 => 2,
        _ => 1,
    };
    Ok(Lowered {
        words,
        unary: false,
        dispatch_extent: extent.div_ceil(lanes),
        local_size: [64, 1, 1],
        fault_law: PcuCheckedScalarFaultLaw::float_relu_backward(
            node.scalar_type,
            PcuRangePolicy::Reject,
            node.float_underflow_policy.unwrap_or_default(),
        )
        .ok_or(PcuVulkanError::UnsupportedPreparedProfile)?,
    })
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
