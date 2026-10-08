//! Detached, transactional homogeneous checked maps with cold resource/SSA indexing.
use alloc::vec::Vec;
use core::marker::PhantomData;
#[path = "compile/compile.rs"]
mod compile;
#[path = "execution/execution.rs"]
mod execution;
#[path = "program/program.rs"]
mod program;
#[rustfmt::skip]
use fusion_pcu::{
    CheckedFloatMapValidationError,
    CheckedIntegerMapValidationError,
    CheckedScalarMapResourceError,
    PcuBindingRef,
    PcuCheckedFloat,
    PcuDispatchFloatBinaryOp,
    PcuDispatchFloatUnaryOp,
    PcuDispatchIntegerBinaryOp,
    PcuDispatchKernelIr,
    PcuExecutionFault,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuRangePolicy,
    PcuScalarType,
};
use crate::PcuCpuHostArgumentError;

// Provider-private bounds, not universal PCU binding or program limits.
const BINDINGS: usize = 4;
const STEPS: usize = 64;
const REGISTERS: usize = 256;

/// Cold structural, argument or terminal failure of a composed checked map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuComposedMapError {
    UnsupportedProfile,
    InvalidResources(CheckedScalarMapResourceError<CheckedFloatMapValidationError>),
    InvalidIntegerResources(CheckedScalarMapResourceError<CheckedIntegerMapValidationError>),
    InvalidArguments(PcuCpuHostArgumentError),
    AllocationFailed,
    ExtentOverflow,
    /// Distinct actual resources overlap and at least one performs a write.
    PhysicalAlias,
    /// An internal detached program invariant failed; no output is published.
    InvalidProgram,
    Fault(PcuExecutionFault),
}
impl PcuCpuComposedMapError {
    #[must_use]
    pub const fn fault(self) -> Option<PcuExecutionFault> {
        match self {
            Self::Fault(fault) => Some(fault),
            _ => None,
        }
    }
}

/// Explicit six-format float preparation, available with `std` or `tensor` allocation.
///
/// This is an Unspecified-reproducibility checked scalar profile. It supports two to
/// four declarations and one or more checked operations, at most 64 detached steps.
/// Scalar SSA identifiers must be below 256 in this provider realization.
/// Structurally admitted primitive maps retain their existing executors. Other checked
/// programs use cold detached admission with separate one-effect identities. No SIMD claim.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuCheckedComposedMap<T: PcuCheckedFloat> {
    marker: PhantomData<T>,
}
impl<T: PcuCheckedFloat> PcuCpuCheckedComposedMap<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}
impl<T: PcuCheckedFloat> Default for PcuCpuCheckedComposedMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy)]
struct Resource {
    binding: PcuBindingRef,
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
    Constant {
        result: usize,
        bytes: [u8; 8],
    },
    IntegerConstant {
        result: usize,
        bytes: [u8; 16],
    },
    IntegerBinary {
        result: usize,
        left: usize,
        right: usize,
        operation: PcuDispatchIntegerBinaryOp,
        range: PcuRangePolicy,
    },
    Binary {
        result: usize,
        left: usize,
        right: usize,
        operation: PcuDispatchFloatBinaryOp,
        range: PcuRangePolicy,
        underflow: PcuFloatUnderflowPolicy,
    },
    Unary {
        result: usize,
        value: usize,
        operation: PcuDispatchFloatUnaryOp,
        range: PcuRangePolicy,
        underflow: PcuFloatUnderflowPolicy,
    },
    Store {
        resource: usize,
        value: usize,
    },
}
type Executable = fn(
    &mut PcuCpuPreparedComposedMap,
    &mut [PcuHostArgument<'_>],
) -> Result<(), PcuCpuComposedMapError>;

/// Owned cold program and private output state, independent of the temporary dispatch IR.
///
/// Exact fourteen-width integer composition separately supports Unspecified and
/// `PortableV1` requests; portable instructions must match their header range policy
/// and have no cross-index read/write dependencies. Float composition retains its
/// separately admitted Unspecified profile. No portable float support is inferred.
///
/// Cloning duplicates the cold program/scratch allocation. Calls never clone, allocate,
/// lower IR or discover an ISA. Constants and readonly broadcasts are initialized anew
/// each call. Observable results and faults retain logical lane/program order. Cold-proven
/// independent integer work may execute in bounded operation-major lane blocks against
/// private outputs. Any exceptional result, including Clamp, replays its block in scalar
/// lane/program order to determine the exact fault and complete recovered outputs.
/// Programs with mutable loads and small extents use the scalar executor.
/// Every public output is committed only after the entire call is fatal-free; complete
/// recovered Clamp output is committed together with the earliest recovered error.
#[derive(Debug, Clone)]
pub struct PcuCpuPreparedComposedMap {
    requirements: PcuImplementationRequirements,
    local_id: u32,
    program: program::ValidatedProgram,
    scratch: Vec<u8>,
}
impl PcuCpuPreparedComposedMap {
    #[must_use]
    pub const fn range_policy(&self) -> PcuRangePolicy {
        self.requirements.range_policy
    }
    #[must_use]
    pub const fn requirements(&self) -> PcuImplementationRequirements {
        self.requirements
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.program.scalar()
    }
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        self.local_id
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.program.schema().len()
    }
    /// Retained private byte workspace, excluding cold metadata allocation.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.scratch.len()
    }
}

// Public within this private module; no additional crate-root API is exported.
pub fn prepare_erased(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<PcuCpuPreparedComposedMap, PcuCpuComposedMapError> {
    // A typed homogeneous resource assessor checks every declaration/instruction below.
    let scalar = kernel
        .bindings
        .first()
        .and_then(|binding| match binding.binding_type {
            fusion_pcu::PcuBindingType::Value(fusion_pcu::PcuValueType::Scalar(scalar)) => {
                Some(scalar)
            }
            _ => None,
        })
        .ok_or(PcuCpuComposedMapError::UnsupportedProfile)?;
    match scalar {
        PcuScalarType::F16 => compile::prepare::<fusion_pcu::PcuF16Bits>(kernel),
        PcuScalarType::BF16 => compile::prepare::<fusion_pcu::PcuBf16Bits>(kernel),
        PcuScalarType::F8E4M3FN => compile::prepare::<fusion_pcu::PcuF8E4M3FnBits>(kernel),
        PcuScalarType::F8E5M2 => compile::prepare::<fusion_pcu::PcuF8E5M2Bits>(kernel),
        PcuScalarType::F32 => compile::prepare::<f32>(kernel),
        PcuScalarType::F64 => compile::prepare::<f64>(kernel),
        PcuScalarType::I8 => compile::integer::prepare::<i8>(kernel),
        PcuScalarType::U8 => compile::integer::prepare::<u8>(kernel),
        PcuScalarType::I16 => compile::integer::prepare::<i16>(kernel),
        PcuScalarType::U16 => compile::integer::prepare::<u16>(kernel),
        PcuScalarType::I32 => compile::integer::prepare::<i32>(kernel),
        PcuScalarType::U32 => compile::integer::prepare::<u32>(kernel),
        PcuScalarType::I64 => compile::integer::prepare::<i64>(kernel),
        PcuScalarType::U64 => compile::integer::prepare::<u64>(kernel),
        PcuScalarType::I128 => compile::integer::prepare::<i128>(kernel),
        PcuScalarType::U128 => compile::integer::prepare::<u128>(kernel),
        PcuScalarType::I256 => compile::integer::prepare::<fusion_pcu::PcuI256>(kernel),
        PcuScalarType::U256 => compile::integer::prepare::<fusion_pcu::PcuU256>(kernel),
        PcuScalarType::I512 => compile::integer::prepare::<fusion_pcu::PcuI512>(kernel),
        PcuScalarType::U512 => compile::integer::prepare::<fusion_pcu::PcuU512>(kernel),
        _ => Err(PcuCpuComposedMapError::UnsupportedProfile),
    }
}
impl<T: PcuCheckedFloat> PcuHostKernelBackend for PcuCpuCheckedComposedMap<T> {
    type Prepared = PcuCpuPreparedComposedMap;
    type Error = PcuCpuComposedMapError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        compile::prepare::<T>(kernel)
    }
}
impl PcuPreparedHostKernel for PcuCpuPreparedComposedMap {
    type Error = PcuCpuComposedMapError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        (self.program.executor())(self, arguments)
    }
}

impl PcuCpuPreparedComposedMap {
    fn validate_arguments(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[usize; BINDINGS], PcuCpuComposedMapError> {
        let mut indices = [0; BINDINGS];
        crate::host::validate_arguments_with_indices(
            arguments,
            self.program.schema(),
            self.program.scalar(),
            self.program.size(),
            |declaration, position| indices[declaration] = position,
        )
        .map_err(PcuCpuComposedMapError::InvalidArguments)?;
        Ok(indices)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
