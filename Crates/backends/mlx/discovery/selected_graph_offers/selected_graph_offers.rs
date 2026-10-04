//! Exact structural numerical offers; native preparation remains a separate fallible step.
use fusion_pcu::{
    PcuCostBoundary, PcuImplementationCost, PcuImplementationMechanism, PcuImplementationOffer,
    PcuImplementationOffers, PcuImplementationRequest, PcuObjectKind, PcuObjectRef,
};
use super::{MlxDiscovery, MlxError};
use crate::{MlxSelectedTensorGraphPlan};
/// Sealed original source and full immutable effect envelope.
/// Offer IDs describe operation families; they are not graph/rate/shape cache keys.
pub struct MlxSelectedTensorGraphRequest<'a> {
    pub plan: &'a MlxSelectedTensorGraphPlan,
}
impl PcuImplementationOffers<MlxSelectedTensorGraphRequest<'_>> for MlxDiscovery {
    type Error = MlxError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, MlxSelectedTensorGraphRequest<'_>>,
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
        if request.executor != super::EXECUTOR
            || request.requirements != request.operation.plan.requirements()
        {
            return Err(MlxError::InvalidRequest(
                "selected graph executor or full requirements mismatch".into(),
            ));
        }
        let plan = request.operation.plan;
        if !(matches!(
            request.boundary,
            PcuCostBoundary::HostInputsResidentOutput | PcuCostBoundary::Resident
        ) || request.boundary == PcuCostBoundary::MixedInputsResidentOutput
            && plan.input_values().len() >= 2)
        {
            return Ok(0);
        }
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: plan.implementation_id(request.device),
                kind: PcuImplementationMechanism::DelegatedRuntime,
                requirements: plan.requirements(),
                workspace_bytes: None,
                cost: PcuImplementationCost::unknown(request.boundary),
            });
        }
        Ok(1)
    }
}
