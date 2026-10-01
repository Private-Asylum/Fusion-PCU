//! Recognized scalar offer requests preserve schema and SSA faults before profile filtering.
//!
//! The bounded core verifier does not validate arbitrary graphics/control dialects. Work outside
//! its scalar region remains unsupported here, rather than being declared globally malformed.

#[rustfmt::skip]
use fusion_pcu::{
    validate_typed_dispatch_value_flow,
    PcuBindingAccess,
    PcuDispatchDataOp,
    PcuDispatchKernelIr,
    PcuDispatchOp,
    PcuTypedDispatchValidationError,
};
use crate::PcuVulkanError;

/// Returns false when this bounded scalar verifier cannot inspect the requested dialect.
pub(super) fn validate(kernel: &PcuDispatchKernelIr<'_>) -> Result<bool, PcuVulkanError> {
    if kernel.entry.logical_shape.contains(&0) {
        return Err(PcuVulkanError::InvalidOfferShape);
    }
    for (index, binding) in kernel.bindings.iter().enumerate() {
        if kernel.bindings[..index]
            .iter()
            .any(|prior| prior.reference() == binding.reference())
        {
            return Err(PcuVulkanError::InvalidOfferBinding(binding.reference()));
        }
    }
    match validate_typed_dispatch_value_flow(kernel) {
        Ok(()) => {}
        Err(
            PcuTypedDispatchValidationError::UnsupportedOperation(_)
            | PcuTypedDispatchValidationError::ValueOutOfRange(_)
            | PcuTypedDispatchValidationError::NonValueBinding(_),
        ) => return Ok(false),
        Err(error) => return Err(PcuVulkanError::InvalidOfferValueFlow(error)),
    }
    let ops = if let [PcuDispatchOp::GridStrideLoop { body, .. }, _] = kernel.ops {
        *body
    } else {
        kernel.ops
    };
    for op in ops {
        let (target, forbidden) = match op {
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad { binding, .. }) => {
                (*binding, PcuBindingAccess::WriteOnly)
            }
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore { binding, .. }) => {
                (*binding, PcuBindingAccess::ReadOnly)
            }
            _ => continue,
        };
        if kernel
            .bindings
            .iter()
            .any(|binding| binding.reference() == target && binding.access == forbidden)
        {
            return Err(PcuVulkanError::InvalidOfferBinding(target));
        }
    }
    Ok(true)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
