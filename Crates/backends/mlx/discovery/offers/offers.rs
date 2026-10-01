//! Operation-specific offers preserve graph descriptor failures as errors.

#[rustfmt::skip]
use super::{
    MlxDiscovery,
    MlxError,
    PcuObjectRef,
    PcuObjectKind,
    EXECUTOR,
    MATMUL_REVISION,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuImplementationCost,
    PcuImplementationId,
    PcuImplementationMechanism,
    PcuImplementationOffer,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuRangePolicy,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    NodeDescriptor,
};

/// Borrowed authentic neutral graph operation, not an alternate MLX frontend.
pub struct MlxMatmulRequest<'a> {
    pub graph: &'a Graph,
    pub node: NodeDescriptor<'a>,
}

impl PcuImplementationOffers<MlxMatmulRequest<'_>> for MlxDiscovery {
    type Error = MlxError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, MlxMatmulRequest<'_>>,
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
        if request.executor != EXECUTOR {
            return Err(MlxError::InvalidRequest("unknown MLX executor".into()));
        }
        validate_graph_descriptor(request.operation)?;
        let node = request.operation.node;
        if node.numerical_mode != Some(request.requirements.numerical_mode)
            || node.numerical_options != request.requirements.numerical_options
            || node.float_underflow_policy != Some(request.requirements.float_underflow)
        {
            return Err(MlxError::InvalidRequest(
                "MLX numerical envelope differs from graph provenance".into(),
            ));
        }
        output.fill(None);
        if request.requirements.range_policy != PcuRangePolicy::Reject
            || crate::MlxMatmulPlan::assess(request.operation.graph, node).is_err()
        {
            return Ok(0);
        }
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: request.device,
                    executor: EXECUTOR,
                    local_id: 1,
                    revision: MATMUL_REVISION,
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

/// A descriptor must agree with the graph supplied in this exact request before capability
/// filtering. Valid non-MatMul nodes remain unsupported rather than malformed.
fn validate_graph_descriptor(request: &MlxMatmulRequest<'_>) -> Result<(), MlxError> {
    let expected = request
        .graph
        .node(request.node.value)
        .map_err(|_| MlxError::InvalidRequest("MLX offer node is absent from its graph".into()))?;
    if expected != request.node {
        return Err(MlxError::InvalidRequest(
            "MLX offer descriptor differs from its graph".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
