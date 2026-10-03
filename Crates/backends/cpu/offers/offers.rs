//! Cold implementation offers for the exact prepared checked F32/F64 Neg profiles.

#[rustfmt::skip]
use fusion_pcu::{
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
    PcuReproducibility,
    PcuRangePolicy,
};
#[rustfmt::skip]
use crate::{
    PcuCpuCheckedNeg,
    PcuCpuPreparedNegError,
};

/// An invalid target/policy envelope or a failure inspecting the CPU implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuNegOfferError {
    DeviceMismatch,
    ExecutorMismatch,
    UnderflowMismatch,
    Provider(PcuCpuPreparedNegError),
}

/// A selected CPU Neg implementation attached to one validated logical CPU snapshot.
///
/// The registering provider supplies the device identity and executor after validating its
/// own namespace and discovery generation. This wrapper checks that requests retain that
/// snapshot; it does not discover devices, establish physical interop or offer resident work.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuNegOffers {
    backend: PcuCpuCheckedNeg,
    device: PcuDeviceIdentity,
    executor: PcuExecutorId,
}

impl PcuCpuNegOffers {
    /// Binds a concrete, already admitted instruction choice to the provider's CPU snapshot.
    #[must_use]
    pub const fn new(
        backend: PcuCpuCheckedNeg,
        device: PcuDeviceIdentity,
        executor: PcuExecutorId,
    ) -> Self {
        Self {
            backend,
            device,
            executor,
        }
    }
}

impl PcuImplementationOffers<PcuDispatchKernelIr<'_>> for PcuCpuNegOffers {
    type Error = PcuCpuNegOfferError;

    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error> {
        if request.device != self.device {
            return Err(PcuCpuNegOfferError::DeviceMismatch);
        }
        if request.executor != self.executor {
            return Err(PcuCpuNegOfferError::ExecutorMismatch);
        }
        let prepared = match self.backend.prepare_host_kernel(request.operation) {
            Ok(prepared) => prepared,
            Err(PcuCpuPreparedNegError::UnsupportedProfile) => return Ok(0),
            Err(PcuCpuPreparedNegError::HeaderUnderflowMismatch) => {
                return Err(PcuCpuNegOfferError::UnderflowMismatch);
            }
            Err(error) => return Err(PcuCpuNegOfferError::Provider(error)),
        };
        if request.requirements != request.operation.numerical_requirements {
            return Ok(0);
        }
        if request.boundary != PcuCostBoundary::Host
            || request.requirements.range_policy != PcuRangePolicy::Reject
            || request.requirements.numerical_options.reproducibility
                == PcuReproducibility::PortableV1
        {
            return Ok(0);
        }
        if request.requirements.float_underflow != prepared.underflow_policy() {
            return Err(PcuCpuNegOfferError::UnderflowMismatch);
        }
        let local_id = prepared.local_id();
        if let Some(first) = output.first_mut() {
            *first = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: self.device,
                    executor: self.executor,
                    local_id,
                    revision: 1,
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

#[cfg(test)]
mod tests;
