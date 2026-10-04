//! Unified cold offers for the statically selected CPU host profile.

#[rustfmt::skip]
use fusion_pcu::{
    PcuCostBoundary,
    PcuDeviceIdentity,
    PcuDispatchFloatBinaryOp,
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
    PCU_CPU_INTEGER_IMPLEMENTATION_REVISION,
    PCU_CPU_FLOAT_BINARY_IMPLEMENTATION_REVISION,
    PcuCpuPreparedBinaryError,
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
    #[allow(clippy::too_many_lines)] // Cold exact tuple checks and frozen implementation IDs remain adjacent.
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
            #[cfg(any(feature = "std", feature = "tensor"))]
            Err(PcuCpuHostError::Composed(crate::PcuCpuComposedMapError::UnsupportedProfile)) => {
                return Ok(0);
            }
            Err(
                PcuCpuHostError::UnsupportedProfile
                | PcuCpuHostError::Neg(PcuCpuPreparedNegError::UnsupportedProfile)
                | PcuCpuHostError::Integer(PcuCpuCheckedIntegerError::UnsupportedProfile)
                | PcuCpuHostError::Binary(PcuCpuPreparedBinaryError::UnsupportedProfile),
            ) => return Ok(0),
            Err(
                PcuCpuHostError::Neg(PcuCpuPreparedNegError::HeaderUnderflowMismatch)
                | PcuCpuHostError::HeaderUnderflowMismatch,
            ) => {
                return Err(PcuCpuHostOfferError::UnderflowMismatch);
            }
            Err(error) => return Err(PcuCpuHostOfferError::Provider(error)),
        };
        if request.requirements != request.operation.numerical_requirements {
            return Ok(0);
        }
        if request.boundary != PcuCostBoundary::Host
            || request.requirements.range_policy != prepared.range_policy()
        {
            return Ok(0);
        }
        let revision = if let PcuCpuPreparedHost::DivRem(plan) = &prepared {
            plan.implementation_revision()
        } else if matches!(
            prepared,
            PcuCpuPreparedHost::Unary(_)
                | PcuCpuPreparedHost::Neg(_)
                | PcuCpuPreparedHost::Identity(_)
                | PcuCpuPreparedHost::Conversion(_)
        ) {
            1
        } else if matches!(
            prepared,
            PcuCpuPreparedHost::F32Binary(_)
                | PcuCpuPreparedHost::F64Binary(_)
                | PcuCpuPreparedHost::F16Binary(_)
                | PcuCpuPreparedHost::BF16Binary(_)
                | PcuCpuPreparedHost::F8E4M3FNBinary(_)
                | PcuCpuPreparedHost::F8E5M2Binary(_)
        ) {
            PCU_CPU_FLOAT_BINARY_IMPLEMENTATION_REVISION
        } else if prepared.range_policy() == PcuRangePolicy::Clamp {
            1
        } else {
            PCU_CPU_INTEGER_IMPLEMENTATION_REVISION
        };
        macro_rules! binary {
            ($prepared:ident, $reject:expr, $clamp:expr, $portable:expr) => {{
                if request.requirements.float_underflow != $prepared.underflow_policy() {
                    return Err(PcuCpuHostOfferError::UnderflowMismatch);
                }
                float_id(
                    match $prepared.range_policy() {
                        PcuRangePolicy::Reject => {
                            if request.requirements.numerical_options.reproducibility
                                == PcuReproducibility::PortableV1
                            {
                                if $portable == 0 {
                                    return Ok(0);
                                }
                                $portable
                            } else {
                                $reject
                            }
                        }
                        PcuRangePolicy::Clamp => $clamp,
                    },
                    $prepared.operation(),
                )
            }};
        }
        let workspace_bytes = match &prepared {
            #[cfg(any(feature = "std", feature = "tensor"))]
            PcuCpuPreparedHost::Composed(plan) => plan.workspace_bytes(),
            #[cfg(any(feature = "std", feature = "tensor"))]
            PcuCpuPreparedHost::Transport(plan) => plan.workspace_bytes(),
            _ => 0,
        };
        let workspace_bytes = u64::try_from(workspace_bytes)
            .map_err(|_| PcuCpuHostOfferError::Provider(PcuCpuHostError::UnsupportedProfile))?;
        let local_id = match &prepared {
            #[cfg(any(feature = "std", feature = "tensor"))]
            PcuCpuPreparedHost::Composed(plan) => plan.local_id(),
            #[cfg(any(feature = "std", feature = "tensor"))]
            PcuCpuPreparedHost::Transport(plan) => plan.local_id(),
            PcuCpuPreparedHost::Unary(prepared) => {
                if request.requirements.float_underflow != prepared.underflow_policy() {
                    return Err(PcuCpuHostOfferError::UnderflowMismatch);
                }
                prepared.local_id()
            }
            #[cfg(any(feature = "std", feature = "tensor"))]
            PcuCpuPreparedHost::UnaryRoles(prepared) => {
                if request.requirements.float_underflow != prepared.underflow_policy() {
                    return Err(PcuCpuHostOfferError::UnderflowMismatch);
                }
                prepared.local_id()
            }
            PcuCpuPreparedHost::DivRem(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::Identity(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::Conversion(prepared) => {
                if request.requirements.float_underflow != prepared.underflow_policy() {
                    return Err(PcuCpuHostOfferError::UnderflowMismatch);
                }
                prepared.local_id()
            }
            PcuCpuPreparedHost::Neg(prepared) => {
                if request.requirements.float_underflow != prepared.underflow_policy() {
                    return Err(PcuCpuHostOfferError::UnderflowMismatch);
                }
                prepared.local_id()
            }
            PcuCpuPreparedHost::F32Binary(prepared) => binary!(prepared, 32, 192, 0),
            PcuCpuPreparedHost::F64Binary(prepared) => binary!(prepared, 36, 196, 0),
            PcuCpuPreparedHost::F16Binary(prepared) => binary!(prepared, 160, 200, 224),
            PcuCpuPreparedHost::BF16Binary(prepared) => binary!(prepared, 164, 204, 228),
            PcuCpuPreparedHost::F8E4M3FNBinary(prepared) => binary!(prepared, 168, 208, 232),
            PcuCpuPreparedHost::F8E5M2Binary(prepared) => binary!(prepared, 172, 212, 236),
            PcuCpuPreparedHost::I8(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U8(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::I16(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U16(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::I32(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U32(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::I64(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U64(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::I128(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U128(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::I256(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U256(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::I512(prepared) => prepared.local_id(),
            PcuCpuPreparedHost::U512(prepared) => prepared.local_id(),
        };
        let revision = if (1024..=1151).contains(&local_id)
            || (2048..=2335).contains(&local_id)
            || (2432..=2463).contains(&local_id)
            || (2560..=2591).contains(&local_id)
            || (4096..=4383).contains(&local_id)
            || (4480..=4511).contains(&local_id)
            || (4608..=4639).contains(&local_id)
            || (17408..=17413).contains(&local_id)
            || (17920..=17943).contains(&local_id)
            || (18688..=18693).contains(&local_id)
            || (19456..=19480).contains(&local_id)
            || (19712..=19725).contains(&local_id)
            || (20992..=21005).contains(&local_id)
        {
            1
        } else {
            revision
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
                workspace_bytes: Some(workspace_bytes),
                cost: PcuImplementationCost::unknown(PcuCostBoundary::Host),
            });
        }
        Ok(1)
    }
}
const fn float_id(base: u32, operation: PcuDispatchFloatBinaryOp) -> u32 {
    let operation = match operation {
        PcuDispatchFloatBinaryOp::Add => 0,
        PcuDispatchFloatBinaryOp::Sub => 1,
        PcuDispatchFloatBinaryOp::Mul => 2,
        PcuDispatchFloatBinaryOp::Div => 3,
    };
    base + operation
}
