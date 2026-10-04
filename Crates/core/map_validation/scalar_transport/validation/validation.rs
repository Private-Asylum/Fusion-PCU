//! Allowlisted transport structure, access and typed SSA validation.

use super::super::PcuScalarTransportError as Error;
#[rustfmt::skip]
use crate::{
    validate_typed_dispatch_value_flow_with_scratch,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchControlOp,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchIndex,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuScalarType,
    PcuValueType,
    PcuValueTypeCaps,
};

pub(super) fn validate<'kernel>(
    kernel: &PcuDispatchKernelIr<'kernel>,
    scalar: PcuScalarType,
    scratch: &mut [Option<PcuValueType>],
) -> Result<(&'kernel [PcuDispatchOp<'kernel>], u32), Error> {
    validate_interface(kernel, scalar)?;
    let (body, indexed, logical_extent) = match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, body },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ] if *extent != 0 => (*body, PcuDispatchIndex::GridStrideId, *extent),
        ops if matches!(
            ops.last(),
            Some(PcuDispatchOp::Control(PcuDispatchControlOp::Return))
        ) =>
        {
            (
                &ops[..ops.len() - 1],
                PcuDispatchIndex::InvocationId,
                kernel.entry.logical_shape[0],
            )
        }
        _ => return Err(Error::MissingReturn),
    };
    let mut has_load = false;
    let mut has_store = false;
    for (position, instruction) in body.iter().copied().enumerate() {
        match instruction {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, index, .. }) => {
                if index != indexed && index != PcuDispatchIndex::BindingElementZero {
                    return Err(Error::InvalidIndex(position));
                }
                if access(kernel, binding)? == PcuBindingAccess::WriteOnly {
                    return Err(Error::InvalidBinding(binding));
                }
                has_load = true;
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, index, .. }) => {
                if index != indexed {
                    return Err(Error::InvalidIndex(position));
                }
                if access(kernel, binding)? == PcuBindingAccess::ReadOnly {
                    return Err(Error::InvalidBinding(binding));
                }
                has_store = true;
            }
            _ => return Err(Error::UnsupportedOperation(position)),
        }
    }
    if !has_load || !has_store {
        return Err(Error::MissingLoadOrStore);
    }
    validate_typed_dispatch_value_flow_with_scratch(kernel, scratch)
        .map_err(Error::InvalidValues)?;
    Ok((body, logical_extent))
}

fn validate_interface(
    kernel: &PcuDispatchKernelIr<'_>,
    scalar: PcuScalarType,
) -> Result<(), Error> {
    if !kernel.ports.is_empty() || !kernel.parameters.is_empty() || kernel.bindings.is_empty() {
        return Err(Error::UnsupportedInterface);
    }
    if kernel.entry.logical_shape[0] == 0
        || kernel.entry.logical_shape[1] != 1
        || kernel.entry.logical_shape[2] != 1
    {
        return Err(Error::InvalidLogicalShape);
    }
    let types = PcuValueTypeCaps::for_scalar(scalar) | PcuValueTypeCaps::SCALAR_VALUES;
    let features =
        PcuDispatchFeatureCaps::READ_ONLY_RESOURCES | PcuDispatchFeatureCaps::MUTABLE_RESOURCES;
    // Guard flat-map admission before recursive capability scanners.
    if kernel.has_nested_grid_stride_loop() {
        return Err(Error::UnsupportedOperation(0));
    }
    if kernel.required_type_support().bits() & !types.bits() != 0
        || kernel.required_feature_support().bits() & !features.bits() != 0
    {
        return Err(Error::UnsupportedRequirements);
    }
    for (position, binding) in kernel.bindings.iter().enumerate() {
        let reference = binding.reference();
        if kernel.bindings[..position]
            .iter()
            .any(|previous| previous.reference() == reference)
        {
            return Err(Error::DuplicateBinding(reference));
        }
        if binding.storage != PcuBindingStorageClass::Storage
            || binding.builtin.is_some()
            || binding.binding_type != PcuBindingType::Value(PcuValueType::Scalar(scalar))
        {
            return Err(Error::InvalidBinding(reference));
        }
    }
    Ok(())
}

fn access(
    kernel: &PcuDispatchKernelIr<'_>,
    binding: PcuBindingRef,
) -> Result<PcuBindingAccess, Error> {
    kernel
        .bindings
        .iter()
        .find(|declaration| declaration.reference() == binding)
        .map(|declaration| declaration.access)
        .ok_or(Error::InvalidBinding(binding))
}
