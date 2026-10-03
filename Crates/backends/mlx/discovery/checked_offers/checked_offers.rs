//! Exact detached dispatch offers, separately scoped from delegated native `MatMul` offers.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCostBoundary,
    PcuDispatchKernelIr,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationOffer,
    PcuImplementationId,
    PcuImplementationMechanism,
    PcuImplementationCost,
    PcuObjectKind,
    PcuObjectRef,
    validate_typed_dispatch_value_flow,
};
#[rustfmt::skip]
use super::{
    MlxDiscovery,
    MlxError,
    EXECUTOR,
};
impl PcuImplementationOffers<PcuDispatchKernelIr<'_>> for MlxDiscovery {
    type Error = MlxError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, MlxError> {
        self.validate(
            PcuObjectRef {
                provider: request.device.provider(),
                generation: request.device.generation(),
                kind: PcuObjectKind::Device,
                id: request.device.device_id(),
            },
            PcuObjectKind::Device,
        )?;
        output.fill(None);
        if request.executor != EXECUTOR
            || request.requirements != request.operation.numerical_requirements
        {
            return Err(MlxError::InvalidRequest(
                "checked offer executor/numerical envelope mismatch".into(),
            ));
        }
        validate_typed_dispatch_value_flow(request.operation)
            .map_err(|_| MlxError::InvalidRequest("checked offer malformed typed SSA".into()))?;
        if request.boundary != PcuCostBoundary::Host {
            return Ok(0);
        }
        let (local_id, revision) =
            if let Ok(plan) = crate::MlxCarrierPlan::assess(request.operation) {
                (plan.implementation_local_id(), 0x0003_0020_0003_0400)
            } else if let Ok(plan) = crate::MlxCheckedUnaryPlan::assess(request.operation) {
                let Some(identity) = unary_identity(plan) else {
                    return Ok(0);
                };
                identity
            } else if let Ok(plan) = crate::MlxCheckedBinaryPlan::assess(request.operation) {
                (
                    plan.implementation_local_id(),
                    if matches!(
                        plan.scalar_type(),
                        fusion_pcu::PcuScalarType::F32 | fusion_pcu::PcuScalarType::F64
                    ) {
                        0x0003_0020_0003_0600
                    } else {
                        0x0003_0020_0003_0200
                    },
                )
            } else if let Ok(plan) = crate::MlxCheckedIntegerPlan::assess(request.operation) {
                (plan.implementation_local_id(), 0x0003_0020_0003_0a00)
            } else if let Ok(plan) = crate::MlxCheckedDivRemPlan::assess(request.operation) {
                (
                    plan.implementation_local_id(),
                    division_revision(request.operation, false),
                )
            } else if let Ok(plan) = crate::MlxCheckedDivRemRolePlan::assess(request.operation) {
                (
                    plan.implementation_local_id(),
                    division_revision(request.operation, true),
                )
            } else {
                return Ok(0);
            };
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: request.device,
                    executor: EXECUTOR,
                    local_id,
                    revision,
                },
                kind: PcuImplementationMechanism::DelegatedRuntime,
                requirements: request.requirements,
                workspace_bytes: None,
                cost: PcuImplementationCost::unknown(request.boundary),
            });
        }
        Ok(1)
    }
}

const fn division_revision(kernel: &PcuDispatchKernelIr<'_>, roles: bool) -> u64 {
    if matches!(
        kernel
            .numerical_requirements
            .numerical_options
            .reproducibility,
        fusion_pcu::PcuReproducibility::PortableV1
    ) {
        0x0003_0020_0003_1100
    } else if roles {
        0x0003_0020_0003_0e00
    } else {
        0x0003_0020_0003_0c00
    }
}

fn unary_identity(plan: crate::MlxCheckedUnaryPlan) -> Option<(u32, u64)> {
    let format = match plan.scalar {
        fusion_pcu::PcuScalarType::F16 => 0,
        fusion_pcu::PcuScalarType::BF16 => 1,
        fusion_pcu::PcuScalarType::F8E4M3FN => 2,
        fusion_pcu::PcuScalarType::F8E5M2 => 3,
        fusion_pcu::PcuScalarType::F32 => 4,
        fusion_pcu::PcuScalarType::F64 => 5,
        _ => return None,
    };
    let role = u32::from(plan.operation == fusion_pcu::PcuDispatchFloatUnaryOp::Relu)
        + 2 * u32::from(plan.range == fusion_pcu::PcuRangePolicy::Clamp)
        + 4 * u32::from(plan.broadcast);
    if plan.requirements().numerical_options.reproducibility
        == fusion_pcu::PcuReproducibility::PortableV1
    {
        Some((0x1700 + format * 8 + role, 0x0003_0020_0003_1400))
    } else {
        Some((
            0x100 + format * 16 + role,
            if format >= 4 {
                0x0003_0020_0003_0700
            } else {
                0x0003_0020_0003_0100
            },
        ))
    }
}
