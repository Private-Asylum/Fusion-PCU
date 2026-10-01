//! Exact cold implementation offers for the proved prepared host bit-map slice.

#[path = "validation/validation.rs"]
mod validation;

#[rustfmt::skip]
use fusion_pcu::{
    PcuCostBoundary,
    PcuDispatchKernelIr,
    PcuExecutorId,
    PcuImplementationCost,
    PcuImplementationId,
    PcuImplementationMechanism,
    PcuImplementationOffer,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuNumericalOptions,
    PcuRangePolicy,
    PcuScalarType,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    validate_float_bit_map,
    PcuSpirvBitOperation,
};
#[rustfmt::skip]
use crate::{
    PcuVulkanBackend,
    PcuVulkanError,
};

impl PcuImplementationOffers<PcuDispatchKernelIr<'_>> for PcuVulkanBackend {
    type Error = PcuVulkanError;

    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error> {
        if self.device_identity() != Some(request.device) || request.executor != PcuExecutorId(0) {
            return Err(PcuVulkanError::InvalidDiscoveryReference);
        }
        if !validation::validate(request.operation)? {
            return Ok(0);
        }
        // Prepared host storage currently uses native little-endian F32/F64 bytes. Match the
        // preparation gate: an offer must not promise execution on an unsupported host ABI.
        if cfg!(target_endian = "big")
            || request.boundary != PcuCostBoundary::Host
            || request.requirements.range_policy != PcuRangePolicy::Reject
            || request.requirements.numerical_options != PcuNumericalOptions::default()
        {
            return Ok(0);
        }
        let Ok(profile) = validate_float_bit_map(request.operation) else {
            return Ok(0);
        };
        if profile.scalar == PcuScalarType::F64 && !self.caps().shader_float64 {
            return Ok(0);
        }
        let local_id = match profile.operation {
            PcuSpirvBitOperation::Copy => 0,
            PcuSpirvBitOperation::CheckedNeg(underflow) => {
                if underflow != request.requirements.float_underflow {
                    return Err(PcuVulkanError::UnsupportedPreparedProfile);
                }
                1
            }
        } + if profile.scalar == PcuScalarType::F64 {
            2
        } else {
            0
        };
        match self.device.validate_bit_map_geometry(profile) {
            Ok(()) => {}
            Err(PcuVulkanError::DeviceLimitExceeded) => return Ok(0),
            Err(error) => return Err(error),
        }
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: request.device,
                    executor: request.executor,
                    local_id,
                    revision: 3,
                },
                kind: PcuImplementationMechanism::NativeKernel,
                requirements: request.requirements,
                // Logical status uses four bytes per element. Native allocation padding is
                // queried only when preparing buffers, so its physical workspace stays unknown.
                workspace_bytes: None,
                // Measurements have not become a universal calibration/model. Cold ranking
                // must retain unknown cost rather than inventing zero or warm heuristics.
                cost: PcuImplementationCost::unknown(PcuCostBoundary::Host),
            });
        }
        Ok(1)
    }
}
