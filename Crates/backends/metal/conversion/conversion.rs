//! Bounded checked mixed-width source adaptation; no aggregate offer is inferred.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,PcuBindingRef,PcuDispatchCheckedFloatConversion,
    PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchIndex,PcuDispatchKernelIr,
    PcuDispatchOp,PcuHostArgument,PcuHostKernelBackend,PcuImplementationRequirements,
    PcuPreparedHostKernel,PcuScalarType,PcuValueTypeCaps,
    validate_checked_float_conversion_map_kernel,validate_typed_dispatch_value_flow,
};
use crate::{MetalError, MetalPreparedFloatConversion, MetalSession};

/// Exact one-load/conversion/store schema retaining original numerical requirements.
#[derive(Clone, Copy, Debug)]
pub struct MetalCheckedConversionPlan {
    input: PcuBindingRef,
    output: PcuBindingRef,
    conversion: PcuDispatchCheckedFloatConversion,
    count: usize,
    broadcast: bool,
    requirements: PcuImplementationRequirements,
}
impl MetalCheckedConversionPlan {
    /// Exact conversion policy and addressing identity within this provider's source-host offers.
    pub(crate) fn implementation_local_id(self) -> u32 {
        let policy = match self.requirements.float_underflow {
            fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            fusion_pcu::PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            fusion_pcu::PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        };
        0x4000
            + u32::from(self.conversion == PcuDispatchCheckedFloatConversion::F64ToF32)
            + 2 * policy
            + 8 * u32::from(self.requirements.range_policy == fusion_pcu::PcuRangePolicy::Clamp)
            + 16 * u32::from(self.broadcast)
    }

    /// Validate mixed widths, typed SSA, launch/index layout and original permissions cold.
    /// # Errors
    /// Refuses Portable, other operations, altered access roles or mismatched policies.
    #[allow(clippy::too_many_lines)] // Closed three-operation profile retains all cold obligations together.
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MetalError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        crate::admission::require_scalar_numerics(kernel)?;
        validate_checked_float_conversion_map_kernel(
            kernel,
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        )
        .map_err(|_| MetalError::Unsupported)?;
        validate_typed_dispatch_value_flow(kernel).map_err(|_| MetalError::Unsupported)?;
        let (body, extent, index) = match kernel.ops {
            [
                PcuDispatchOp::GridStrideLoop { extent, body },
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (*body, *extent, PcuDispatchIndex::GridStrideId),
            [
                body @ ..,
                PcuDispatchOp::Control(PcuDispatchControlOp::Return),
            ] => (
                body,
                kernel.entry.logical_shape[0],
                PcuDispatchIndex::InvocationId,
            ),
            _ => return Err(MetalError::Unsupported),
        };
        let [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: loaded,
                binding: input,
                index: source_index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                result,
                value,
                conversion,
                range_policy,
                underflow_policy,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: output,
                index: output_index,
                value: stored,
            }),
        ] = body
        else {
            return Err(MetalError::Unsupported);
        };
        if loaded != value
            || result != stored
            || input == output
            || kernel.bindings.len() != 2
            || extent == 0
            || extent > 0x7fff_ffff
            || kernel.entry.logical_shape[1..] != [1, 1]
            || *output_index != index
            || (*source_index != index && *source_index != PcuDispatchIndex::BindingElementZero)
            || kernel.numerical_requirements.range_policy != *range_policy
            || kernel.numerical_requirements.float_underflow != *underflow_policy
        {
            return Err(MetalError::Unsupported);
        }
        for (reference, access) in [
            (*input, PcuBindingAccess::ReadOnly),
            (*output, PcuBindingAccess::ReadWrite),
        ] {
            if kernel
                .bindings
                .iter()
                .find(|binding| binding.reference() == reference)
                .is_none_or(|binding| binding.access != access)
            {
                return Err(MetalError::Unsupported);
            }
        }
        Ok(Self {
            input: *input,
            output: *output,
            conversion: *conversion,
            count: usize::try_from(extent).map_err(|_| MetalError::InvalidExtent)?,
            broadcast: *source_index == PcuDispatchIndex::BindingElementZero,
            requirements: kernel.numerical_requirements,
        })
    }
    #[must_use]
    pub const fn requirements(self) -> PcuImplementationRequirements {
        self.requirements
    }
    #[must_use]
    pub const fn conversion(self) -> PcuDispatchCheckedFloatConversion {
        self.conversion
    }
    #[must_use]
    pub const fn logical_extent(self) -> usize {
        self.count
    }
    #[must_use]
    pub const fn input_binding(self) -> PcuBindingRef {
        self.input
    }
    #[must_use]
    pub const fn output_binding(self) -> PcuBindingRef {
        self.output
    }
    const fn source_scalar(self) -> PcuScalarType {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => PcuScalarType::F32,
            PcuDispatchCheckedFloatConversion::F64ToF32 => PcuScalarType::F64,
        }
    }
    const fn output_scalar(self) -> PcuScalarType {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => PcuScalarType::F64,
            PcuDispatchCheckedFloatConversion::F64ToF32 => PcuScalarType::F32,
        }
    }
    fn bytes(self, input: bool) -> Result<usize, MetalError> {
        let scalar = if input {
            self.source_scalar()
        } else {
            self.output_scalar()
        };
        let count = if input && self.broadcast {
            1
        } else {
            self.count
        };
        count
            .checked_mul(usize::from(scalar.bit_width() / 8))
            .ok_or(MetalError::InvalidExtent)
    }
}
/// Explicit conversion source provider; ordinary global routing remains separately qualified.
pub struct MetalConversionHostBackend {
    session: MetalSession,
}
impl MetalConversionHostBackend {
    #[must_use]
    pub const fn new(session: MetalSession) -> Self {
        Self { session }
    }
}
/// Frozen schema/pipeline; caller borrows are staged into owned native resources per call.
pub struct MetalPreparedConversionHostKernel {
    plan: MetalCheckedConversionPlan,
    map: MetalPreparedFloatConversion,
    session: MetalSession,
    may_have_written: bool,
}
impl PcuHostKernelBackend for MetalConversionHostBackend {
    type Error = MetalError;
    type Prepared = MetalPreparedConversionHostKernel;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        let plan = MetalCheckedConversionPlan::assess(kernel)?;
        let map = self.session.prepare_float_conversion(
            plan.conversion,
            plan.requirements.float_underflow,
            plan.requirements.range_policy,
            plan.broadcast,
        )?;
        Ok(MetalPreparedConversionHostKernel {
            plan,
            map,
            session: self.session.clone(),
            may_have_written: false,
        })
    }
}
impl PcuPreparedHostKernel for MetalPreparedConversionHostKernel {
    type Error = MetalError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        if arguments.len() != 2 {
            return Err(MetalError::InvalidExtent);
        }
        let input = arguments
            .iter()
            .position(|arg| arg.target() == self.plan.input)
            .ok_or(MetalError::InvalidExtent)?;
        let output = arguments
            .iter()
            .position(|arg| arg.target() == self.plan.output)
            .ok_or(MetalError::InvalidExtent)?;
        let input_bytes = self.plan.bytes(true)?;
        let output_bytes = self.plan.bytes(false)?;
        if input == output
            || arguments[input].scalar() != self.plan.source_scalar()
            || arguments[output].scalar() != self.plan.output_scalar()
            || arguments[input].access() != PcuBindingAccess::ReadOnly
            || arguments[output].access() != PcuBindingAccess::ReadWrite
            || arguments[input].bytes().len() < input_bytes
            || arguments[output].bytes().len() < output_bytes
        {
            return Err(MetalError::InvalidExtent);
        }
        let resident = self
            .session
            .upload_bytes(&arguments[input].bytes()[..input_bytes])?;
        let (completed, fault) = self.map.execute_completed(&resident, self.plan.count)?;
        completed.read_into_bytes(
            &mut arguments[output]
                .bytes_mut()
                .ok_or(MetalError::InvalidExtent)?[..output_bytes],
        )?;
        self.may_have_written = true;
        fault.map_or(Ok(()), |fault| Err(MetalError::Arithmetic(fault)))
    }
}

impl MetalPreparedConversionHostKernel {
    #[must_use]
    pub const fn plan(&self) -> MetalCheckedConversionPlan {
        self.plan
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.session.ensure_quiescent().is_err()
    }
}
#[path = "mixed/mixed.rs"]
mod mixed;

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
