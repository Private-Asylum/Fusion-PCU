//! Exact source-host conversion offer; native owned conversion remains separately unsupported.
use fusion_pcu::{
    PcuCostBoundary, PcuDispatchKernelIr, PcuExecutorId, PcuImplementationCost,
    PcuImplementationId, PcuImplementationMechanism, PcuImplementationOffer,
    PcuImplementationOffers, PcuImplementationRequest, PcuObjectKind, PcuObjectRef,
    validate_typed_dispatch_value_flow,
};
use super::{MetalDiscovery, MetalError};
impl PcuImplementationOffers<PcuDispatchKernelIr<'_>> for MetalDiscovery {
    type Error = MetalError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
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
            || request.requirements != request.operation.numerical_requirements
        {
            return Err(MetalError::Unsupported);
        }
        crate::dispatch_shape::require_non_nested(request.operation)?;
        validate_typed_dispatch_value_flow(request.operation)
            .map_err(|_| MetalError::Unsupported)?;
        if request.boundary != PcuCostBoundary::Host {
            return Ok(0);
        }
        let Ok(plan) = crate::MetalCheckedConversionPlan::assess(request.operation) else {
            return Ok(0);
        };
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: request.device,
                    executor: PcuExecutorId(0),
                    local_id: plan.implementation_local_id(),
                    revision: 0x0000_0008_0000_0100,
                },
                kind: PcuImplementationMechanism::NativeKernel,
                requirements: request.requirements,
                workspace_bytes: None,
                cost: PcuImplementationCost::unknown(request.boundary),
            });
        }
        Ok(1)
    }
}
