//! Bounded checked mixed-width source adaptation; no aggregate offer is inferred.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,PcuBindingRef,PcuDispatchCheckedFloatConversion,
    PcuDispatchControlOp,PcuDispatchDataOp,PcuDispatchIndex,PcuDispatchKernelIr,
    PcuDispatchOp,PcuHostArgument,PcuHostKernelBackend,PcuImplementationRequirements,
    PcuPreparedHostKernel,PcuScalarType,PcuValueTypeCaps,
    validate_checked_float_conversion_map_kernel,validate_typed_dispatch_value_flow,
};
use crate::{MlxError, MlxPreparedFloatConversion, MlxSession};

#[path = "publication/publication.rs"]
mod publication;

/// Exact one-load/conversion/store schema retaining original numerical requirements.
#[derive(Clone, Copy, Debug)]
pub struct MlxCheckedConversionPlan {
    input: PcuBindingRef,
    output: PcuBindingRef,
    conversion: PcuDispatchCheckedFloatConversion,
    count: usize,
    broadcast: bool,
    requirements: PcuImplementationRequirements,
}
impl MlxCheckedConversionPlan {
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
    pub fn assess(kernel: &PcuDispatchKernelIr<'_>) -> Result<Self, MlxError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            != fusion_pcu::PcuReproducibility::Unspecified
        {
            return Err(unsupported());
        }
        validate_checked_float_conversion_map_kernel(
            kernel,
            PcuValueTypeCaps::FLOAT32 | PcuValueTypeCaps::FLOAT64,
        )
        .map_err(|_| unsupported())?;
        validate_typed_dispatch_value_flow(kernel).map_err(|_| unsupported())?;
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
            _ => return Err(unsupported()),
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
            return Err(unsupported());
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
            return Err(unsupported());
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
                return Err(unsupported());
            }
        }
        Ok(Self {
            input: *input,
            output: *output,
            conversion: *conversion,
            count: usize::try_from(extent).map_err(|_| MlxError::InvalidExtent)?,
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
    /// Logical initial read span; scalar broadcasts retain one source element.
    #[must_use]
    pub const fn input_element_count(self) -> usize {
        if self.broadcast { 1 } else { self.count }
    }
    #[must_use]
    pub const fn input_binding(self) -> PcuBindingRef {
        self.input
    }
    #[must_use]
    pub const fn output_binding(self) -> PcuBindingRef {
        self.output
    }
    #[must_use]
    pub const fn source_scalar(self) -> PcuScalarType {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => PcuScalarType::F32,
            PcuDispatchCheckedFloatConversion::F64ToF32 => PcuScalarType::F64,
        }
    }
    #[must_use]
    pub const fn output_scalar(self) -> PcuScalarType {
        match self.conversion {
            PcuDispatchCheckedFloatConversion::F32ToF64 => PcuScalarType::F64,
            PcuDispatchCheckedFloatConversion::F64ToF32 => PcuScalarType::F32,
        }
    }
    fn bytes(self, input: bool) -> Result<usize, MlxError> {
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
            .ok_or(MlxError::InvalidExtent)
    }
}
/// Explicit conversion source provider; ordinary global routing remains separately qualified.
pub struct MlxConversionHostBackend {
    session: MlxSession,
}
impl MlxConversionHostBackend {
    #[must_use]
    pub const fn new(session: MlxSession) -> Self {
        Self { session }
    }
}
impl MlxConversionHostBackend {
    /// Cold specialization for an actual larger encoded input allocation.
    /// Logical indexing and numerical permissions remain those of the original kernel.
    /// # Errors
    /// Rejects unsupported source or insufficient/overflowing input capacity.
    pub fn prepare_host_kernel_with_input_extent(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_count: usize,
    ) -> Result<MlxPreparedConversionHostKernel, MlxError> {
        self.prepare(kernel, Some(input_count))
    }
    fn prepare(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
        input_count: Option<usize>,
    ) -> Result<MlxPreparedConversionHostKernel, MlxError> {
        crate::dispatch_shape::require_non_nested(kernel)?;
        let plan = MlxCheckedConversionPlan::assess(kernel)?;
        let input_count = input_count.unwrap_or_else(|| plan.input_element_count());
        if input_count < plan.input_element_count() {
            return Err(MlxError::InvalidExtent);
        }
        let input_bytes = input_count
            .checked_mul(usize::from(plan.source_scalar().bit_width() / 8))
            .ok_or(MlxError::InvalidExtent)?;
        let map = self.session.prepare_float_conversion_with_input_extent(
            plan.conversion,
            plan.requirements.float_underflow,
            plan.requirements.range_policy,
            plan.count,
            plan.broadcast,
            input_count,
        )?;
        Ok(MlxPreparedConversionHostKernel {
            plan,
            map,
            session: self.session.clone(),
            readback: vec![0; plan.bytes(false)?],
            input_bytes,
            input_count,
            may_have_written: false,
        })
    }
}

/// Frozen schema/pipeline; caller borrows are staged into owned MLX integer-carrier resources per call.
pub struct MlxPreparedConversionHostKernel {
    plan: MlxCheckedConversionPlan,
    map: MlxPreparedFloatConversion,
    session: MlxSession,
    readback: Vec<u8>,
    input_bytes: usize,
    input_count: usize,
    may_have_written: bool,
}
impl PcuHostKernelBackend for MlxConversionHostBackend {
    type Error = MlxError;
    type Prepared = MlxPreparedConversionHostKernel;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.prepare(kernel, None)
    }
}
impl PcuPreparedHostKernel for MlxPreparedConversionHostKernel {
    type Error = MlxError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        if arguments.len() != 2 {
            return Err(MlxError::InvalidExtent);
        }
        let input = arguments
            .iter()
            .position(|arg| arg.target() == self.plan.input)
            .ok_or(MlxError::InvalidExtent)?;
        let output = arguments
            .iter()
            .position(|arg| arg.target() == self.plan.output)
            .ok_or(MlxError::InvalidExtent)?;
        let input_bytes = self.input_bytes;
        let output_bytes = self.plan.bytes(false)?;
        if input == output
            || arguments[input].scalar() != self.plan.source_scalar()
            || arguments[output].scalar() != self.plan.output_scalar()
            || arguments[input].access() != PcuBindingAccess::ReadOnly
            || arguments[output].access() != PcuBindingAccess::ReadWrite
            || arguments[input].bytes().len() < input_bytes
            || arguments[output].bytes().len() < output_bytes
        {
            return Err(MlxError::InvalidExtent);
        }
        let input_count = self.input_count;
        let resident = self.session.upload_encoded_bytes(
            self.plan.source_scalar(),
            input_count,
            &arguments[input].bytes()[..input_bytes],
        )?;
        let result = self.map.execute_resident(&resident);
        resident.release()?;
        let destination = arguments[output]
            .bytes_mut()
            .ok_or(MlxError::InvalidExtent)?;
        self.publish_host_completion(result?, destination)
    }
}
impl MlxPreparedConversionHostKernel {
    #[must_use]
    pub const fn plan(&self) -> MlxCheckedConversionPlan {
        self.plan
    }
    #[must_use]
    pub const fn input_bindings(&self) -> &[PcuBindingRef] {
        std::slice::from_ref(&self.plan.input)
    }
    #[must_use]
    pub const fn output_binding(&self) -> PcuBindingRef {
        self.plan.output
    }
    #[must_use]
    pub const fn output_byte_len(&self) -> usize {
        self.readback.len()
    }
    /// Frozen native input capacity; the plan retains its smaller logical read span.
    #[must_use]
    pub const fn input_element_count(&self) -> usize {
        self.input_count
    }
    #[must_use]
    pub const fn input_byte_len(&self) -> usize {
        self.input_bytes
    }
    #[must_use]
    pub const fn scalar_type(&self) -> PcuScalarType {
        self.plan.output_scalar()
    }
    #[must_use]
    pub const fn source_scalar_type(&self) -> PcuScalarType {
        self.plan.source_scalar()
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.session.validate_access_available().is_err()
    }
    /// Borrow one actual typed input into a fresh completed converted owner. Resident
    /// input shapes equal the capacity frozen during cold preparation.
    /// # Errors
    /// Refuses binding/type/extent/session mismatch before work or returns checked/native faults.
    pub fn execute_inputs(
        &mut self,
        inputs: &[crate::MlxBinaryInput<'_>],
    ) -> Result<crate::MlxEncodedCompletion, MlxError> {
        self.may_have_written = false;
        self.session.validate_access_available()?;
        let [input] = inputs else {
            return Err(MlxError::InvalidExtent);
        };
        match input {
            crate::MlxBinaryInput::HostBytes {
                target,
                scalar,
                bytes,
            } => {
                if *target != self.plan.input
                    || *scalar != self.plan.source_scalar()
                    || bytes.len() < self.input_bytes
                {
                    return Err(MlxError::InvalidExtent);
                }
                let count = self.input_count;
                let resident = self.session.upload_encoded_bytes(
                    *scalar,
                    count,
                    &bytes[..self.input_bytes],
                )?;
                let result = self.map.execute_resident(&resident);
                resident.release()?;
                result
            }
            crate::MlxBinaryInput::Resident { target, array } => {
                if *target != self.plan.input {
                    return Err(MlxError::InvalidExtent);
                }
                self.map.execute_resident(array)
            }
        }
    }
}

fn unsupported() -> MlxError {
    MlxError::InvalidRequest("unsupported checked conversion source profile".into())
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
