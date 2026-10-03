//! Normal exact selection over actual resource roles, separate from the canonical Portable ABI.
#[rustfmt::skip]
use fusion_pcu::{
    describe_checked_float_unary_map,
    PcuBindingAccess,
    PcuCheckedFloatUnaryMapDescription,
    PcuDispatchKernelIr,
    PcuReproducibility,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    PcuSpirvCheckedUnaryProfile,
    PcuSpirvError,
    PcuSpirvLoweringOptions,
    PcuSpirvModuleInfo,
    PcuSpirvSink,
};

/// Two actual native resources and retained original host-role/request metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuSpirvCheckedUnaryRoleProfile {
    pub native: PcuSpirvCheckedUnaryProfile,
    pub description: PcuCheckedFloatUnaryMapDescription,
}
impl PcuSpirvCheckedUnaryRoleProfile {
    #[must_use]
    pub const fn local_id(self) -> Option<u32> {
        match self.native.local_id() {
            Some(id) => Some(18176 + id - 96),
            None => None,
        }
    }
}
/// Admits only normal actual-role selection not already admitted by the legacy canonical validator.
/// Unread declarations must be readonly; their number does not add native buffers or descriptors.
/// # Errors
/// Rejects malformed role/SSA/header requests, Portable requests and unrepresentable native extents.
pub fn validate_checked_float_unary_roles_map(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuSpirvCheckedUnaryRoleProfile, PcuSpirvError> {
    if kernel
        .numerical_requirements
        .numerical_options
        .reproducibility
        != PcuReproducibility::Unspecified
        || super::validate_checked_float_unary_map(kernel).is_ok()
    {
        return Err(PcuSpirvError::UnsupportedNumericalRequirements);
    }
    let description = describe_checked_float_unary_map(kernel)
        .map_err(|_| PcuSpirvError::InvalidKernelSignature)?;
    if description.scalar == PcuScalarType::F64 && description.logical_extent > u32::MAX / 2 {
        return Err(PcuSpirvError::InvalidBinding);
    }
    if kernel.bindings.iter().any(|binding| {
        binding.reference() != description.output_binding
            && binding.access != PcuBindingAccess::ReadOnly
    }) {
        return Err(PcuSpirvError::InvalidBinding);
    }
    Ok(PcuSpirvCheckedUnaryRoleProfile {
        native: PcuSpirvCheckedUnaryProfile {
            scalar: description.scalar,
            operation: description.operation,
            underflow: description.requirements.float_underflow,
            range: description.requirements.range_policy,
            extent: description.logical_extent,
            broadcast: description.broadcast_input,
            local_size: [64, 1, 1],
            portable: false,
        },
        description,
    })
}
/// Emits the same U32-only module as its canonical numerical counterpart.
/// The host projection retains original declarations; no unread resource is synthesized in SPIR-V.
/// # Errors
/// Returns exact admission, version/capability or sink failure.
pub fn lower_checked_float_unary_roles_to_spirv<S: PcuSpirvSink>(
    kernel: &PcuDispatchKernelIr<'_>,
    options: PcuSpirvLoweringOptions,
    sink: &mut S,
) -> Result<(PcuSpirvModuleInfo, PcuSpirvCheckedUnaryRoleProfile), PcuSpirvError> {
    let profile = validate_checked_float_unary_roles_map(kernel)?;
    let info = super::emit_profile(profile.native, options, sink)?;
    Ok((info, profile))
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
