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
    PcuReproducibility,
    PcuRangePolicy,
    PcuScalarType,
    PcuDispatchFloatBinaryOp,
};
#[rustfmt::skip]
use fusion_pcu_spirv::{
    validate_float_bit_map,
    validate_scalar_transport_map,
    validate_checked_float_unary_map,
    PcuSpirvBitOperation,
    validate_checked_float_binary_map,
    validate_checked_integer_map,
    validate_checked_div_rem_map,
    validate_checked_float_conversion_map,
};
#[rustfmt::skip]
use crate::{
    PcuVulkanBackend,
    PcuVulkanError,
    PcuVulkanPreparedComposed,
    PcuVulkanPreparedOrderedTransport,
};

impl PcuImplementationOffers<PcuDispatchKernelIr<'_>> for PcuVulkanBackend {
    type Error = PcuVulkanError;

    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error> {
        if !self.valid_host_request(request)? {
            return Ok(0);
        }
        let (local_id, revision, geometry) =
            if let Some(profile) = self.arithmetic_offer(request.operation) {
                profile
            } else if let Ok(profile) = validate_checked_float_binary_map(request.operation) {
                if profile.range != request.requirements.range_policy {
                    return Ok(0);
                }
                if profile.underflow != request.requirements.float_underflow {
                    return Err(PcuVulkanError::UnsupportedPreparedProfile);
                }
                let operation = match profile.operation {
                    PcuDispatchFloatBinaryOp::Add => 0,
                    PcuDispatchFloatBinaryOp::Sub => 1,
                    PcuDispatchFloatBinaryOp::Mul => 2,
                    PcuDispatchFloatBinaryOp::Div => 3,
                };
                (
                    binary_base(
                        profile.scalar,
                        profile.range,
                        request.requirements.numerical_options.reproducibility,
                    )
                    .ok_or(PcuVulkanError::UnsupportedPreparedProfile)?
                        + operation,
                    2,
                    self.device.validate_binary_geometry(profile),
                )
            } else if let Ok(profile) = validate_checked_float_unary_map(request.operation) {
                (
                    profile
                        .local_id()
                        .ok_or(PcuVulkanError::UnsupportedPreparedProfile)?,
                    1,
                    self.device.validate_unary_geometry(profile),
                )
            } else if let Some(profile) = self.transport_offer(request.operation) {
                profile
            } else if let Ok(profile) = validate_float_bit_map(request.operation) {
                let Some(id) = self.legacy_bit_map_id(request, profile)? else {
                    return Ok(0);
                };
                (id, 3, self.device.validate_bit_map_geometry(profile))
            } else {
                return Ok(0);
            };
        match geometry {
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
                    revision,
                },
                kind: PcuImplementationMechanism::NativeKernel,
                requirements: request.requirements,
                // Scalar and ordered-map status sizes differ. Native allocation padding is
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

const fn binary_base(
    scalar: PcuScalarType,
    range: PcuRangePolicy,
    reproducibility: PcuReproducibility,
) -> Option<u32> {
    if matches!(reproducibility, PcuReproducibility::PortableV1) {
        // The cold validator already requires exact shared descriptor membership.
        return match scalar {
            PcuScalarType::F16 => Some(64),
            PcuScalarType::BF16 => Some(68),
            PcuScalarType::F8E4M3FN => Some(72),
            PcuScalarType::F8E5M2 => Some(76),
            _ => None,
        };
    }
    match (scalar, range) {
        (PcuScalarType::F32, PcuRangePolicy::Reject) => Some(4),
        (PcuScalarType::F32, PcuRangePolicy::Clamp) => Some(8),
        (PcuScalarType::F64, PcuRangePolicy::Reject) => Some(12),
        (PcuScalarType::F64, PcuRangePolicy::Clamp) => Some(16),
        (PcuScalarType::F16, PcuRangePolicy::Reject) => Some(32),
        (PcuScalarType::BF16, PcuRangePolicy::Reject) => Some(36),
        (PcuScalarType::F8E4M3FN, PcuRangePolicy::Reject) => Some(40),
        (PcuScalarType::F8E5M2, PcuRangePolicy::Reject) => Some(44),
        (PcuScalarType::F16, PcuRangePolicy::Clamp) => Some(48),
        (PcuScalarType::BF16, PcuRangePolicy::Clamp) => Some(52),
        (PcuScalarType::F8E4M3FN, PcuRangePolicy::Clamp) => Some(56),
        (PcuScalarType::F8E5M2, PcuRangePolicy::Clamp) => Some(60),
        _ => None,
    }
}

impl PcuVulkanBackend {
    fn transport_offer(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<(u32, u64, Result<(), PcuVulkanError>)> {
        // Existing simple identity/broadcast profiles retain their exact IDs and revisions.
        if let Ok(profile) = validate_scalar_transport_map(kernel) {
            Some((
                profile.local_id()?,
                1,
                self.device.validate_scalar_transport_geometry(profile),
            ))
        } else if let Some(profile) = PcuVulkanPreparedOrderedTransport::admitted_profile(kernel) {
            Some((
                profile.local_id()?,
                1,
                self.device.validate_ordered_transport_geometry(&profile),
            ))
        } else {
            None
        }
    }

    fn legacy_bit_map_id(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        profile: fusion_pcu_spirv::PcuSpirvBitMapProfile,
    ) -> Result<Option<u32>, PcuVulkanError> {
        if request.requirements.range_policy != PcuRangePolicy::Reject
            || (profile.scalar == PcuScalarType::F64 && !self.caps().shader_float64)
        {
            return Ok(None);
        }
        let operation = match profile.operation {
            PcuSpirvBitOperation::Copy => 0,
            PcuSpirvBitOperation::CheckedNeg(underflow) => {
                if underflow != request.requirements.float_underflow {
                    return Err(PcuVulkanError::UnsupportedPreparedProfile);
                }
                1
            }
        };
        Ok(Some(
            operation
                + if profile.scalar == PcuScalarType::F64 {
                    2
                } else {
                    0
                },
        ))
    }

    fn valid_host_request(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
    ) -> Result<bool, PcuVulkanError> {
        if self.device_identity() != Some(request.device) || request.executor != PcuExecutorId(0) {
            return Err(PcuVulkanError::InvalidDiscoveryReference);
        }
        if !validation::validate(request.operation)? {
            return Ok(false);
        }
        if request.requirements != request.operation.numerical_requirements {
            return Ok(false);
        }
        // Prepared host storage uses native little-endian scalar bytes. Match the
        // preparation gate: an offer must not promise execution on an unsupported host ABI.
        if cfg!(target_endian = "big") || request.boundary != PcuCostBoundary::Host {
            return Ok(false);
        }
        Ok(true)
    }
}

impl PcuVulkanBackend {
    fn arithmetic_offer(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Option<(u32, u64, Result<(), PcuVulkanError>)> {
        if let Ok(profile) = fusion_pcu_spirv::validate_checked_float_unary_roles_map(kernel) {
            Some((
                profile.local_id()?,
                1,
                self.device.validate_unary_geometry(profile.native),
            ))
        } else if let Ok(profile) = validate_checked_float_conversion_map(kernel) {
            Some((
                profile.local_id(),
                1,
                self.device.validate_conversion_geometry(profile),
            ))
        } else if let Ok(profile) = validate_checked_div_rem_map(kernel) {
            Some((
                profile.local_id()?,
                1,
                self.device.validate_div_rem_geometry(profile),
            ))
        } else if let Ok(profile) = validate_checked_integer_map(kernel) {
            Some((
                profile.local_id()?,
                1,
                self.device.validate_integer_geometry(profile),
            ))
        } else if let Some(profile) = PcuVulkanPreparedComposed::admitted_profile(kernel) {
            let ordinal = match profile.scalar() {
                PcuScalarType::F16 => 0,
                PcuScalarType::BF16 => 1,
                PcuScalarType::F8E4M3FN => 2,
                PcuScalarType::F8E5M2 => 3,
                PcuScalarType::F32 => 4,
                PcuScalarType::F64 => 5,
                _ => return None,
            };
            Some((
                if profile.is_one_effect() {
                    18944
                } else {
                    17664
                } + ordinal,
                1,
                self.device.validate_composed_geometry(&profile),
            ))
        } else {
            None
        }
    }
}
