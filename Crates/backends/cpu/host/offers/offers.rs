//! Unified cold offers for the statically selected CPU host profile.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuExecutorId,
    PcuHostKernelBackend,
    PcuImplementationCost,
    PcuImplementationId,
    PcuImplementationMechanism,
    PcuImplementationOffer,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuNumericalOptions,
    PcuRangePolicy,
};
#[rustfmt::skip]
use crate::{
    PCU_CPU_INTEGER_IMPLEMENTATION_REVISION,
    PcuCpuCheckedIntegerError,
    PcuCpuPreparedNegError,
    PcuCpuHostBackend,
    PcuCpuHostError,
    PcuCpuPreparedHost,
};

/// Snapshot or preparation failure; valid unsupported work alone yields zero offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuHostOfferError {
    DeviceMismatch,
    ExecutorMismatch,
    UnderflowMismatch,
    Provider(PcuCpuHostError),
}

/// Exact host implementation offers attached to a validated logical CPU snapshot.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuHostOffers {
    backend: PcuCpuHostBackend,
    device: PcuDeviceIdentity,
    executor: PcuExecutorId,
}
impl PcuCpuHostOffers {
    #[must_use]
    pub const fn new(
        backend: PcuCpuHostBackend,
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
impl PcuImplementationOffers<PcuDispatchKernelIr<'_>> for PcuCpuHostOffers {
    type Error = PcuCpuHostOfferError;
    fn implementation_offers(
        &self,
        request: &PcuImplementationRequest<'_, PcuDispatchKernelIr<'_>>,
        output: &mut [Option<PcuImplementationOffer>],
    ) -> Result<usize, Self::Error> {
        if request.device != self.device {
            return Err(PcuCpuHostOfferError::DeviceMismatch);
        }
        if request.executor != self.executor {
            return Err(PcuCpuHostOfferError::ExecutorMismatch);
        }
        let prepared = match self.backend.prepare_host_kernel(request.operation) {
            Ok(prepared) => prepared,
            Err(
                PcuCpuHostError::UnsupportedProfile
                | PcuCpuHostError::Neg(PcuCpuPreparedNegError::UnsupportedProfile)
                | PcuCpuHostError::Integer(PcuCpuCheckedIntegerError::UnsupportedProfile),
            ) => return Ok(0),
            Err(error) => return Err(PcuCpuHostOfferError::Provider(error)),
        };
        if request.boundary != PcuCostBoundary::Host
            || request.requirements.range_policy != PcuRangePolicy::Reject
            || request.requirements.numerical_options != PcuNumericalOptions::default()
        {
            return Ok(0);
        }
        let revision = if matches!(prepared, PcuCpuPreparedHost::Neg(_)) {
            1
        } else {
            PCU_CPU_INTEGER_IMPLEMENTATION_REVISION
        };
        let local_id = match prepared {
            PcuCpuPreparedHost::Neg(prepared) => {
                if request.requirements.float_underflow != prepared.underflow_policy() {
                    return Err(PcuCpuHostOfferError::UnderflowMismatch);
                }
                prepared.local_id()
            }
            PcuCpuPreparedHost::I8(prepared) => integer_id(0, prepared.operation()),
            PcuCpuPreparedHost::U8(prepared) => integer_id(1, prepared.operation()),
            PcuCpuPreparedHost::I16(prepared) => integer_id(2, prepared.operation()),
            PcuCpuPreparedHost::U16(prepared) => integer_id(3, prepared.operation()),
            PcuCpuPreparedHost::I32(prepared) => integer_id(4, prepared.operation()),
            PcuCpuPreparedHost::U32(prepared) => integer_id(5, prepared.operation()),
            PcuCpuPreparedHost::I64(prepared) => integer_id(6, prepared.operation()),
            PcuCpuPreparedHost::U64(prepared) => integer_id(7, prepared.operation()),
        };
        if let Some(slot) = output.first_mut() {
            *slot = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: self.device,
                    executor: self.executor,
                    local_id,
                    revision,
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
const fn integer_id(scalar: u32, operation: PcuDispatchIntegerBinaryOp) -> u32 {
    let operation = match operation {
        PcuDispatchIntegerBinaryOp::Add => 0,
        PcuDispatchIntegerBinaryOp::Sub => 1,
        PcuDispatchIntegerBinaryOp::Mul => 2,
    };
    4 + scalar * 3 + operation
}
