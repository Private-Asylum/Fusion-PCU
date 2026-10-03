//! Cold scalar integer offers tied to an externally validated CPU snapshot.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchKernelIr,
    PcuExecutorId,
    PcuHostKernelBackend,
    PcuImplementationCost,
    PcuImplementationId,
    PcuImplementationMechanism,
    PcuImplementationOffer,
    PcuImplementationOffers,
    PcuImplementationRequest,
};
#[rustfmt::skip]
use super::{
    PcuCpuCheckedInteger,
    PcuCpuCheckedIntegerError,
};

/// Snapshot identity mismatch while enumerating scalar integer implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuIntegerOfferError {
    DeviceMismatch,
    ExecutorMismatch,
    Provider(PcuCpuCheckedIntegerError),
}

/// One exact typed scalar integer backend bound to a provider's CPU discovery snapshot.
///
/// It advertises host execution only. Cold preparation admits the bounded exact integer
/// `PortableV1` descriptor; no discovery, tensor, division or resident ownership promise follows.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuIntegerOffers<T: PcuCheckedInteger> {
    backend: PcuCpuCheckedInteger<T>,
    device: PcuDeviceIdentity,
    executor: PcuExecutorId,
}

impl<T: PcuCheckedInteger> PcuCpuIntegerOffers<T> {
    #[must_use]
    pub const fn new(device: PcuDeviceIdentity, executor: PcuExecutorId) -> Self {
        Self {
            backend: PcuCpuCheckedInteger::new(),
            device,
            executor,
        }
    }
}

impl<T: PcuCheckedInteger> PcuImplementationOffers<PcuDispatchKernelIr<'_>>
    for PcuCpuIntegerOffers<T>
{
    type Error = PcuCpuIntegerOfferError;

    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error> {
        if request.device != self.device {
            return Err(PcuCpuIntegerOfferError::DeviceMismatch);
        }
        if request.executor != self.executor {
            return Err(PcuCpuIntegerOfferError::ExecutorMismatch);
        }
        let prepared = match self.backend.prepare_host_kernel(request.operation) {
            Ok(prepared) => prepared,
            Err(PcuCpuCheckedIntegerError::UnsupportedProfile) => return Ok(0),
            Err(error) => return Err(PcuCpuIntegerOfferError::Provider(error)),
        };
        if request.requirements != request.operation.numerical_requirements {
            return Ok(0);
        }
        if request.boundary != PcuCostBoundary::Host
            || request.requirements.range_policy != prepared.range_policy()
        {
            return Ok(0);
        }
        if let Some(first) = output.first_mut() {
            *first = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: self.device,
                    executor: self.executor,
                    // IDs 0..=3 are reserved for the Neg instruction implementations.
                    local_id: prepared.local_id(),
                    revision: prepared.implementation_revision(),
                },
                kind: PcuImplementationMechanism::NativeKernel,
                requirements: request.requirements,
                workspace_bytes: Some(0),
                cost: PcuImplementationCost::unknown(PcuCostBoundary::Host),
            });
        }
        Ok(1)
    }
}
