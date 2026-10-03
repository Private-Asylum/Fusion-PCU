//! Frozen exact-width quotient/remainder maps with transactional two-output publication.

use core::marker::PhantomData;
#[path = "execution/bytes/bytes.rs"]
mod bytes;
#[path = "schema/schema.rs"]
mod schema;
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuCheckedIntegerDivision,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarType,
    PcuValueType,
};
#[rustfmt::skip]
use crate::{
    PcuCpuCheckedIntegerError,
    PcuCpuHostError,
};

/// Revision of the frozen checked `DivRem` executor, independent of prior integer binary IDs.
pub const PCU_CPU_DIV_REM_IMPLEMENTATION_REVISION: u64 = 2;

/// Explicit scalar CPU backend; the fourteen sealed checked integer carriers are admitted.
#[derive(Debug, Clone, Copy, Default)]
pub struct PcuCpuCheckedDivRem<T: PcuCheckedIntegerDivision>(PhantomData<T>);
impl<T: PcuCheckedIntegerDivision> PcuCpuCheckedDivRem<T> {
    #[must_use]
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

/// Detached schema and monomorphized executable; retains no IR or caller storage.
#[derive(Debug, Clone, Copy)]
pub struct PcuCpuPreparedDivRem {
    schema: [(PcuBindingRef, usize, PcuBindingAccess); 4],
    argument_count: usize,
    inputs: [PcuBindingRef; 2],
    outputs: [PcuBindingRef; 2],
    operands: [usize; 2],
    extent: usize,
    scalar: PcuScalarType,
    local_id: u32,
    size: usize,
    execute: bytes::Executable,
}
impl PcuCpuPreparedDivRem {
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.scalar
    }
    #[must_use]
    pub const fn local_id(&self) -> u32 {
        self.local_id
    }
    pub(super) fn prepare(
        kernel: &PcuDispatchKernelIr<'_>,
        value_type: PcuValueType,
    ) -> Result<Self, PcuCpuHostError> {
        macro_rules! width {
            ($ty:ty) => {
                PcuCpuCheckedDivRem::<$ty>::new()
                    .prepare_host_kernel(kernel)
                    .map_err(PcuCpuHostError::Integer)
            };
        }
        match value_type {
            PcuValueType::Scalar(PcuScalarType::I8) => width!(i8),
            PcuValueType::Scalar(PcuScalarType::U8) => width!(u8),
            PcuValueType::Scalar(PcuScalarType::I16) => width!(i16),
            PcuValueType::Scalar(PcuScalarType::U16) => width!(u16),
            PcuValueType::Scalar(PcuScalarType::I32) => width!(i32),
            PcuValueType::Scalar(PcuScalarType::U32) => width!(u32),
            PcuValueType::Scalar(PcuScalarType::I64) => width!(i64),
            PcuValueType::Scalar(PcuScalarType::U64) => width!(u64),
            PcuValueType::Scalar(PcuScalarType::I128) => width!(i128),
            PcuValueType::Scalar(PcuScalarType::U128) => width!(u128),
            PcuValueType::Scalar(PcuScalarType::I256) => width!(fusion_pcu::PcuI256),
            PcuValueType::Scalar(PcuScalarType::U256) => width!(fusion_pcu::PcuU256),
            PcuValueType::Scalar(PcuScalarType::I512) => width!(fusion_pcu::PcuI512),
            PcuValueType::Scalar(PcuScalarType::U512) => width!(fusion_pcu::PcuU512),
            _ => Err(PcuCpuHostError::UnsupportedProfile),
        }
    }
    pub(super) const fn host_schema(&self) -> [(PcuBindingRef, usize, PcuBindingAccess); 4] {
        self.schema
    }
    /// Original typed declarations, including schema-proved unused read-only spans.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.argument_count
    }
    /// Legacy indexed plans retain revision two; operand-role and exact Portable plans use revision one.
    #[must_use]
    pub const fn implementation_revision(&self) -> u64 {
        if self.local_id >= 8192 {
            1
        } else {
            PCU_CPU_DIV_REM_IMPLEMENTATION_REVISION
        }
    }
}
impl<T: PcuCheckedIntegerDivision> PcuHostKernelBackend for PcuCpuCheckedDivRem<T> {
    type Prepared = PcuCpuPreparedDivRem;
    type Error = PcuCpuCheckedIntegerError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        schema::prepare::<T>(kernel)
    }
}
impl PcuPreparedHostKernel for PcuCpuPreparedDivRem {
    type Error = PcuCpuHostError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        crate::host::validate_arguments(
            arguments,
            &self.host_schema()[..self.argument_count],
            self.scalar,
            self.size,
        )
        .map_err(PcuCpuHostError::Arguments)?;
        // Every original declaration is validated. Distinct read resources are borrowed once;
        // repeated SSA operands share that immutable borrow, and unread declarations are skipped.
        let mut inputs = [None, None];
        let mut outputs = [None, None];
        for argument in arguments {
            if let Some(slot) = self
                .inputs
                .iter()
                .position(|binding| *binding == argument.target())
            {
                inputs[slot] = Some(argument.bytes());
            } else if let Some(slot) = self
                .outputs
                .iter()
                .position(|binding| *binding == argument.target())
            {
                outputs[slot] = Some(argument.bytes_mut().expect("writable output validated"));
            }
        }
        let [Some(quotient), Some(remainder)] = outputs else {
            unreachable!("complete outputs")
        };
        (self.execute)(
            inputs[self.operands[0]].expect("complete input"),
            inputs[self.operands[1]].expect("complete input"),
            quotient,
            remainder,
            self.extent,
        )
        .map_err(PcuCpuHostError::Integer)
    }
}

const fn integer_id(scalar: PcuScalarType) -> Option<u32> {
    match scalar {
        PcuScalarType::I8 => Some(128),
        PcuScalarType::U8 => Some(129),
        PcuScalarType::I16 => Some(130),
        PcuScalarType::U16 => Some(131),
        PcuScalarType::I32 => Some(132),
        PcuScalarType::U32 => Some(133),
        PcuScalarType::I64 => Some(134),
        PcuScalarType::U64 => Some(135),
        PcuScalarType::I128 => Some(512),
        PcuScalarType::U128 => Some(513),
        PcuScalarType::I256 => Some(514),
        PcuScalarType::U256 => Some(515),
        PcuScalarType::I512 => Some(516),
        PcuScalarType::U512 => Some(517),
        _ => None,
    }
}
