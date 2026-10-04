//! Separate cold Portable unary descriptor; no generic floating or graph admission.
#[rustfmt::skip]
use fusion_pcu::{
    describe_checked_float_unary_map,
    describe_portable_v1_unary_map,
    PcuDispatchKernelIr,
    PcuImplementationRequirements,
    PcuCheckedFloatUnaryMapDescription,
};
#[rustfmt::skip]
use crate::{
    MetalError,
    MetalSession,
};
use super::MetalPreparedFloatKernel;
/// Detached exact checked unary map with its original ordinary or requested Portable header.
/// Assessment describes eligibility; execution qualification is a separate provider contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalCheckedUnaryPlan {
    description: PcuCheckedFloatUnaryMapDescription,
    extent: usize,
    bytes: usize,
}
impl MetalCheckedUnaryPlan {
    /// Validates actual load/store roles, the exact unary body and physical byte bounds.
    ///
    /// # Errors
    /// Rejects malformed maps, unsupported encodings, endian or byte extent overflow.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let description =
            describe_checked_float_unary_map(kernel).map_err(|_| MetalError::Unsupported)?;
        let (extent, bytes) = checked_extents(description)?;
        Ok(Self {
            description,
            extent,
            bytes,
        })
    }
    /// Complete original request, without permission or reproducibility rewriting.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.description.requirements
    }
    /// Actual arithmetic and input/output roles, independent of declaration order.
    #[must_use]
    pub const fn description(&self) -> &PcuCheckedFloatUnaryMapDescription {
        &self.description
    }
}

/// Detached exact Portable unary request; no session, device compilation or caller IR borrow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalPortableUnaryPlan {
    description: PcuCheckedFloatUnaryMapDescription,
    extent: usize,
    bytes: usize,
}
impl MetalPortableUnaryPlan {
    /// Validates the neutral requested profile and checked host/status byte bounds.
    ///
    /// # Errors
    /// Rejects non-Portable/unproved requests, malformed roles or overflowing byte extents.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        if cfg!(target_endian = "big") {
            return Err(MetalError::Unsupported);
        }
        let description =
            describe_portable_v1_unary_map(kernel).map_err(|_| MetalError::Unsupported)?;
        let (extent, bytes) = checked_extents(description)?;
        Ok(Self {
            description,
            extent,
            bytes,
        })
    }
    /// Complete original request, independent of compile strategy or permission shortcuts.
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.description.requirements
    }
    /// Actual arithmetic/load/store roles and submitted versus visited geometry.
    #[must_use]
    pub const fn description(&self) -> &PcuCheckedFloatUnaryMapDescription {
        &self.description
    }
}
pub(super) fn prepare(
    session: &MetalSession,
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<MetalPreparedFloatKernel, MetalError> {
    crate::dispatch_shape::require_non_nested(kernel)?;
    let plan = MetalPortableUnaryPlan::assess(kernel)?;
    prepare_description(session, plan.description, plan.extent, plan.bytes)
}

pub(super) fn prepare_normal(
    session: &MetalSession,
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<MetalPreparedFloatKernel, MetalError> {
    crate::dispatch_shape::require_non_nested(kernel)?;
    super::require_scalar_numerics(kernel)?;
    let plan = MetalCheckedUnaryPlan::assess(kernel)?;
    prepare_description(session, plan.description, plan.extent, plan.bytes)
}

fn checked_extents(
    description: PcuCheckedFloatUnaryMapDescription,
) -> Result<(usize, usize), MetalError> {
    let extent =
        usize::try_from(description.logical_extent).map_err(|_| MetalError::InvalidExtent)?;
    let width = usize::from(description.scalar.bit_width()) / 8;
    let bytes = extent.checked_mul(width).ok_or(MetalError::InvalidExtent)?;
    isize::try_from(bytes).map_err(|_| MetalError::InvalidExtent)?;
    extent
        .checked_mul(4)
        .and_then(|status| isize::try_from(status).ok())
        .ok_or(MetalError::InvalidExtent)?;
    Ok((extent, bytes))
}

fn prepare_description(
    session: &MetalSession,
    description: PcuCheckedFloatUnaryMapDescription,
    extent: usize,
    bytes: usize,
) -> Result<MetalPreparedFloatKernel, MetalError> {
    let width = usize::from(description.scalar.bit_width()) / 8;
    let map = session
        .prepare_checked_float_unary_with_range(
            description.scalar,
            description.operation,
            description.requirements.float_underflow,
            description.requirements.range_policy,
        )?
        .with_broadcast(description.broadcast_input);
    Ok(MetalPreparedFloatKernel {
        map,
        input: description.input_binding,
        output: description.output_binding,
        extent,
        scalar: description.scalar,
        bytes,
        input_bytes: if description.broadcast_input {
            width
        } else {
            bytes
        },
        requirements: description.requirements,
    })
}
