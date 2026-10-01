//! Cold scalar integer offers tied to an externally validated CPU snapshot.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCheckedInteger,
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
    PcuScalarType,
};
#[rustfmt::skip]
use super::{
    PCU_CPU_INTEGER_IMPLEMENTATION_REVISION,
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
/// It advertises host execution only. No discovery, resident memory or `PortableV1` proof is
/// established by this wrapper, and integer checking does not inherit Neg's SIMD capability.
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
        if request.boundary != PcuCostBoundary::Host
            || request.requirements.range_policy != PcuRangePolicy::Reject
            || request.requirements.numerical_options != PcuNumericalOptions::default()
        {
            return Ok(0);
        }
        let scalar = match T::TYPE {
            PcuScalarType::I8 => 0,
            PcuScalarType::U8 => 1,
            PcuScalarType::I16 => 2,
            PcuScalarType::U16 => 3,
            PcuScalarType::I32 => 4,
            PcuScalarType::U32 => 5,
            PcuScalarType::I64 => 6,
            PcuScalarType::U64 => 7,
            _ => unreachable!("sealed checked integer widths"),
        };
        let operation = match prepared.operation() {
            PcuDispatchIntegerBinaryOp::Add => 0,
            PcuDispatchIntegerBinaryOp::Sub => 1,
            PcuDispatchIntegerBinaryOp::Mul => 2,
        };
        if let Some(first) = output.first_mut() {
            *first = Some(PcuImplementationOffer {
                implementation: PcuImplementationId {
                    device: self.device,
                    executor: self.executor,
                    // IDs 0..=3 are reserved for the Neg instruction implementations.
                    local_id: 4 + scalar * 3 + operation,
                    revision: PCU_CPU_INTEGER_IMPLEMENTATION_REVISION,
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
