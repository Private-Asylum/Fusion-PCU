//! Separate representation-preserving ordered loads/stores, detached from arithmetic profiles.
use alloc::vec::Vec;
#[path = "compile/compile.rs"]
mod compile;
#[path = "execution/execution.rs"]
mod execution;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuScalarTransportError,
    PcuScalarType,
};
use crate::PcuCpuHostArgumentError;

// These bounds belong to this detached realization, not the neutral transport contract.
const RESOURCES: usize = 16;
const STEPS: usize = 64;
const REGISTERS: usize = 256;
const CARRIER_BYTES: usize = 64;

/// Cold admission, argument or internal transaction failure; transport has no numeric faults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuScalarTransportError {
    UnsupportedProfile,
    InvalidTransport(PcuScalarTransportError),
    InvalidArguments(PcuCpuHostArgumentError),
    AllocationFailed,
    ExtentOverflow,
    PhysicalAlias,
    InvalidProgram,
}
#[derive(Debug, Clone, Copy)]
struct Resource {
    declaration: usize,
    read_bytes: usize,
    write_bytes: usize,
    shadow: Option<usize>,
}
#[derive(Debug, Clone, Copy)]
enum Step {
    Load {
        result: usize,
        resource: usize,
        zero: bool,
    },
    Store {
        value: usize,
        resource: usize,
    },
}

/// Cold-owned all-carrier transport; every stored raw bit is preserved without floating arithmetic.
///
/// The first realization supports 22 explicit sealed host carriers, Unspecified reproducibility,
/// at most 16 actual resources/64 ordered steps/256 SSA slots, and arbitrary unused declarations.
/// Mutable element-zero dependencies across multiple lanes and physical mutable overlap are refused.
/// Private writable shadows and scalar registers are allocated once during preparation. Cloning is
/// cold metadata/storage work; warm execution uses neither IR inspection nor heap allocation.
#[derive(Debug, Clone)]
pub struct PcuCpuPreparedScalarTransport {
    scalar: PcuScalarType,
    requirements: PcuImplementationRequirements,
    size: usize,
    extent: usize,
    local_id: u32,
    schema: Vec<(PcuBindingRef, usize, PcuBindingAccess)>,
    resources: Vec<Resource>,
    steps: Vec<Step>,
    scratch: Vec<u8>,
    registers: Vec<[u8; CARRIER_BYTES]>,
}
impl PcuCpuPreparedScalarTransport {
    pub(crate) fn prepare(
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self, PcuCpuScalarTransportError> {
        compile::prepare(kernel)
    }
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        self.local_id
    }
    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        self.requirements.range_policy
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.schema.len()
    }
    /// Actual private payload storage, excluding cold program/schema metadata.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.scratch.len() + self.registers.len() * CARRIER_BYTES
    }
}
impl PcuPreparedHostKernel for PcuCpuPreparedScalarTransport {
    type Error = PcuCpuScalarTransportError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        crate::host::validate_arguments(arguments, &self.schema, self.scalar, self.size)
            .map_err(PcuCpuScalarTransportError::InvalidArguments)?;
        execution::execute(self, arguments)
    }
}
