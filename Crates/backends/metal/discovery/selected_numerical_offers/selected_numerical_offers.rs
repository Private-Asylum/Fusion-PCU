//! Exact structural numerical offers; native preparation remains a separate fallible step.
use fusion_pcu::{
    PcuCostBoundary, PcuExecutorId, PcuImplementationCost, PcuImplementationMechanism,
    PcuImplementationOffer, PcuImplementationOffers, PcuImplementationRequest, PcuObjectKind,
    PcuObjectRef,
};
use super::{MetalDiscovery, MetalError};
use crate::{MetalSelectedNumericalTensorPlan};
/// Sealed original source and full immutable effect envelope.
/// Offer IDs describe operation families; they are not graph/rate/shape cache keys.
pub struct MetalSelectedNumericalTensorRequest<'a> {
    pub plan: &'a MetalSelectedNumericalTensorPlan,
}
impl PcuImplementationOffers<MetalSelectedNumericalTensorRequest<'_>> for MetalDiscovery {
    type Error = MetalError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, MetalSelectedNumericalTensorRequest<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, MetalError> {
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
        if request.executor != PcuExecutorId(0)
            || request.requirements != request.operation.plan.requirements()
        {
            return Err(MetalError::Unsupported);
        }
        let plan = request.operation.plan;
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
                kind: PcuImplementationMechanism::NativeKernel,
                requirements: plan.requirements(),
                workspace_bytes: None,
                cost: PcuImplementationCost::unknown(request.boundary),
            });
        }
        Ok(1)
    }
}
