//! Exact source-closure offers, independent from native `MatMul` and mutable host dispatch.
#[rustfmt::skip]
use fusion_pcu::{
    PcuCostBoundary,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationOffer,
    PcuImplementationMechanism,
    PcuImplementationCost,
    PcuObjectRef,
    PcuObjectKind,
};
use fusion_pcu::dialect::tensor::TensorOwnedSelectedProgram;
#[rustfmt::skip]
use super::{
    MlxDiscovery,
    MlxError,
    EXECUTOR,
};
/// Authentic selected low-format graph with an explicit immutable numerical request envelope.
pub struct MlxCheckedTensorRequest<'a> {
    pub program: &'a TensorOwnedSelectedProgram,
}
impl PcuImplementationOffers<MlxCheckedTensorRequest<'_>> for MlxDiscovery {
    type Error = MlxError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, MlxCheckedTensorRequest<'_>>,
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
        if request.executor != EXECUTOR {
            return Err(MlxError::InvalidRequest(
                "unknown checked tensor executor".into(),
            ));
        }
        if !matches!(
            request.boundary,
            PcuCostBoundary::Host
                | PcuCostBoundary::HostInputsResidentOutput
                | PcuCostBoundary::Resident
        ) {
            return Ok(0);
        }
        let Ok(plan) = crate::MlxCheckedTensorPlan::assess_program(
            request.operation.program,
            request.requirements,
        ) else {
            return Ok(0);
        };
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: plan.implementation_id(request.device),
                kind: PcuImplementationMechanism::DelegatedRuntime,
                requirements: request.requirements,
                workspace_bytes: None,
                cost: PcuImplementationCost::unknown(request.boundary),
            });
        }
        Ok(1)
    }
}

/// Authentic selected six-format checked binary graph, independent of native library `MatMul`.
pub struct MlxTensorBinaryRequest<'a> {
    pub program: &'a TensorOwnedSelectedProgram,
}
impl PcuImplementationOffers<MlxTensorBinaryRequest<'_>> for MlxDiscovery {
    type Error = MlxError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, MlxTensorBinaryRequest<'_>>,
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
        if request.executor != EXECUTOR {
            return Err(MlxError::InvalidRequest(
                "unknown checked binary tensor executor".into(),
            ));
        }
        let Ok(plan) = crate::MlxCheckedTensorBinaryPlan::assess_program(
            request.operation.program,
            request.requirements,
        ) else {
            return Ok(0);
        };
        if !(matches!(
            request.boundary,
            PcuCostBoundary::HostInputsResidentOutput | PcuCostBoundary::Resident
        ) || request.boundary == PcuCostBoundary::MixedInputsResidentOutput
            && plan.input_values().len() == 2)
        {
            return Ok(0);
        }
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: plan.implementation_id(request.device),
                kind: PcuImplementationMechanism::DelegatedRuntime,
                requirements: request.requirements,
                workspace_bytes: None,
                cost: PcuImplementationCost::unknown(request.boundary),
            });
        }
        Ok(1)
    }
}

/// Authentic selected fourteen-width checked integer graph with immutable escaped output.
pub struct MlxTensorIntegerRequest<'a> {
    pub program: &'a TensorOwnedSelectedProgram,
}
impl PcuImplementationOffers<MlxTensorIntegerRequest<'_>> for MlxDiscovery {
    type Error = MlxError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, MlxTensorIntegerRequest<'_>>,
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
        if request.executor != EXECUTOR {
            return Err(MlxError::InvalidRequest(
                "unknown checked integer tensor executor".into(),
            ));
        }
        let Ok(plan) = crate::MlxCheckedTensorIntegerPlan::assess_program(
            request.operation.program,
            request.requirements,
        ) else {
            return Ok(0);
        };
        let Some(offer) = integer_offer(&plan, request.device, request.boundary) else {
            return Ok(0);
        };
        if let Some(slot) = output.first_mut() {
            *slot = Some(offer);
        }
        Ok(1)
    }
}
fn integer_offer(
    plan: &crate::MlxCheckedTensorIntegerPlan,
    device: fusion_pcu::PcuDeviceIdentity,
    boundary: PcuCostBoundary,
) -> Option<PcuImplementationOffer> {
    if !(matches!(
        boundary,
        PcuCostBoundary::HostInputsResidentOutput | PcuCostBoundary::Resident
    ) || boundary == PcuCostBoundary::MixedInputsResidentOutput
        && plan.input_values().len() == 2)
    {
        return None;
    }
    Some(PcuImplementationOffer {
        implementation: plan.implementation_id(device),
        kind: PcuImplementationMechanism::DelegatedRuntime,
        requirements: plan.requirements(),
        workspace_bytes: None,
        cost: PcuImplementationCost::unknown(boundary),
    })
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
