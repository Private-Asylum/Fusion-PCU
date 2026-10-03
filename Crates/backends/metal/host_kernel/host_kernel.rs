//! Typed host authoring bridge over the owned checked Metal map.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarType,
};
#[rustfmt::skip]
use crate::{
    MetalError,
    MetalPreparedIntegerKernel,
    MetalPreparedCarrierKernel,
    MetalPreparedFloatKernel,
    MetalPreparedFloatBinaryKernel,
    MetalBuffer,
    MetalSession,
};

/// Structured binding/admission failure or Metal operational/numerical error.
pub type MetalHostKernelError = PcuHostDispatchError<MetalError>;

/// Reusable owned executable for fourteen-width checked integer maps and six-format encoding unary and integer-synthesized binary host calls.
///
/// Every call stages current inputs, allocates fresh device output/status and waits for terminal
/// completion. Host output publication follows checked success; untouched host tails survive.
pub struct MetalPreparedSingleHostKernel {
    session: MetalSession,
    kernel: Program,
    scalar: PcuScalarType,
    schema: Vec<PcuBindingRef>,
    required_bytes: usize,
    binding_bytes: Vec<usize>,
    may_have_written: bool,
}
enum Program {
    Carrier(MetalPreparedCarrierKernel),
    Integer(MetalPreparedIntegerKernel),
    Float(MetalPreparedFloatKernel),
    FloatBinary(MetalPreparedFloatBinaryKernel),
}
impl Program {
    const fn scalar(&self) -> PcuScalarType {
        match self {
            Self::Carrier(kernel) => kernel.scalar_type(),
            Self::Integer(kernel) => kernel.scalar_type(),
            Self::Float(kernel) => kernel.scalar_type(),
            Self::FloatBinary(kernel) => kernel.scalar_type(),
        }
    }
    const fn element_count(&self) -> usize {
        match self {
            Self::Carrier(kernel) => kernel.element_count(),
            Self::Integer(kernel) => kernel.element_count(),
            Self::Float(kernel) => kernel.element_count(),
            Self::FloatBinary(kernel) => kernel.element_count(),
        }
    }
    fn binding_bytes(&self, binding: PcuBindingRef) -> usize {
        match self {
            Self::FloatBinary(kernel) => kernel.binding_bytes(binding),
            Self::Float(kernel) => kernel.binding_bytes(binding),
            Self::Carrier(kernel) => kernel.binding_bytes(binding),
            Self::Integer(kernel) => kernel.binding_bytes(binding),
        }
    }
    const fn input_bindings(&self) -> [PcuBindingRef; 2] {
        match self {
            Self::Carrier(kernel) => [kernel.input_binding(); 2],
            Self::Integer(kernel) => kernel.actual_input_pair(),
            Self::Float(kernel) => [kernel.input_binding(); 2],
            Self::FloatBinary(kernel) => kernel.actual_input_pair(),
        }
    }
    const fn output_binding(&self) -> PcuBindingRef {
        match self {
            Self::Carrier(kernel) => kernel.output_binding(),
            Self::Integer(kernel) => kernel.output_binding(),
            Self::Float(kernel) => kernel.output_binding(),
            Self::FloatBinary(kernel) => kernel.output_binding(),
        }
    }
    fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        match self {
            Self::Carrier(kernel) => kernel.execute_into(inputs[0], output),
            Self::Integer(kernel) => kernel.execute_into(inputs, output),
            Self::Float(kernel) => kernel.execute_into(inputs[0], output),
            Self::FloatBinary(kernel) => kernel.execute_reads_into(inputs, output),
        }
    }
    fn execute_completed(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        match self {
            Self::Carrier(kernel) => kernel.execute(inputs[0]).map(|output| (output, None)),
            Self::Integer(kernel) => kernel.execute_completed(inputs),
            Self::Float(kernel) => kernel.execute_completed(inputs[0]),
            Self::FloatBinary(kernel) => kernel.execute_completed(inputs),
        }
    }
}
impl MetalSession {
    fn prepare_single_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedSingleHostKernel, MetalHostKernelError> {
        if cfg!(target_endian = "big") {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        }
        let prepared = if crate::admission::carrier::is_carrier_kernel(kernel) {
            Program::Carrier(self.prepare_carrier_kernel(kernel).map_err(PcuHostDispatchError::Backend)?)
        } else if fusion_pcu::describe_checked_float_unary_map(kernel).is_ok() {
            Program::Float(self.prepare_float_unary_kernel(kernel).map_err(PcuHostDispatchError::Backend)?)
        } else if kernel.bindings.first().is_some_and(|binding| {
            matches!(binding.binding_type, fusion_pcu::PcuBindingType::Value(value) if matches!(value, fusion_pcu::PcuValueType::Scalar(fusion_pcu::PcuScalarType::F32 | fusion_pcu::PcuScalarType::F64 | fusion_pcu::PcuScalarType::F16 | fusion_pcu::PcuScalarType::BF16 | fusion_pcu::PcuScalarType::F8E4M3FN | fusion_pcu::PcuScalarType::F8E5M2)))
        }) {
            Program::FloatBinary(self.prepare_float_binary_kernel(kernel).map_err(PcuHostDispatchError::Backend)?)
        } else {
            Program::Integer(
                self.prepare_integer_kernel(kernel)
                    .map_err(PcuHostDispatchError::Backend)?,
            )
        };
        let required_bytes = prepared
            .element_count()
            .checked_mul(usize::from(prepared.scalar().bit_width()) / 8)
            .ok_or(PcuHostDispatchError::Backend(MetalError::InvalidExtent))?;
        // Actual output role is last in the detached validation table, independent
        // of caller declaration order. Unread readonly declarations have zero span.
        let output = prepared.output_binding();
        let schema: Vec<_> = kernel
            .bindings
            .iter()
            .copied()
            .map(fusion_pcu::PcuBinding::reference)
            .filter(|binding| *binding != output)
            .chain(core::iter::once(output))
            .collect();
        let binding_bytes = schema
            .iter()
            .map(|binding| prepared.binding_bytes(*binding))
            .collect();
        Ok(MetalPreparedSingleHostKernel {
            session: self.clone(),
            scalar: prepared.scalar(),
            kernel: prepared,
            schema,
            required_bytes,
            binding_bytes,
            may_have_written: false,
        })
    }
}
impl PcuPreparedHostKernel for MetalPreparedSingleHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        validate(&self.schema, self.scalar, &self.binding_bytes, arguments)?;
        let references = self.kernel.input_bindings();
        let input = |reference| {
            let position = self
                .schema
                .iter()
                .position(|&target| target == reference)
                .ok_or(PcuHostDispatchError::Missing(reference))?;
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == reference)
                .ok_or(PcuHostDispatchError::Missing(reference))?;
            self.session
                .upload_bytes(&argument.bytes()[..self.binding_bytes[position]])
                .map_err(PcuHostDispatchError::Backend)
        };
        let left = input(references[0])?;
        let right = if references[0] == references[1] {
            None
        } else {
            Some(input(references[1])?)
        };
        self.may_have_written = true;
        let (output, recovered) = self
            .kernel
            .execute_completed([&left, right.as_ref().unwrap_or(&left)])
            .map_err(PcuHostDispatchError::Backend)?;
        let output_ref = self.kernel.output_binding();
        let destination = arguments
            .iter_mut()
            .find(|argument| argument.target() == output_ref)
            .ok_or(PcuHostDispatchError::Missing(output_ref))?
            .bytes_mut()
            .ok_or(PcuHostDispatchError::AccessMismatch(output_ref))?;
        // Publication occurs only after the full terminal numerical gate succeeds.
        output
            .read_into_bytes(&mut destination[..self.required_bytes])
            .map_err(PcuHostDispatchError::Backend)?;
        recovered.map_or(Ok(()), |fault| {
            Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
        })
    }
}
fn validate(
    schema: &[PcuBindingRef],
    scalar: PcuScalarType,
    binding_bytes: &[usize],
    arguments: &[PcuHostArgument<'_>],
) -> Result<(), MetalHostKernelError> {
    for (index, argument) in arguments.iter().enumerate() {
        let target = argument.target();
        if arguments[..index]
            .iter()
            .any(|prior| prior.target() == target)
        {
            return Err(PcuHostDispatchError::Duplicate(target));
        }
        let Some(position) = schema.iter().position(|&declared| declared == target) else {
            return Err(PcuHostDispatchError::Unexpected(target));
        };
        if argument.scalar() != scalar {
            return Err(PcuHostDispatchError::TypeMismatch(target));
        }
        let access = if position == schema.len() - 1 {
            PcuBindingAccess::ReadWrite
        } else {
            PcuBindingAccess::ReadOnly
        };
        if argument.access() != access {
            return Err(PcuHostDispatchError::AccessMismatch(target));
        }
        if argument.bytes().len() < binding_bytes[position] {
            return Err(PcuHostDispatchError::BufferTooSmall(target));
        }
    }
    for &target in schema {
        if !arguments.iter().any(|argument| argument.target() == target) {
            return Err(PcuHostDispatchError::Missing(target));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn host_admission_rejects_wrong_type_duplicate_access_extent_and_coverage() {
        let refs = [
            PcuBindingRef::new(0, 0),
            PcuBindingRef::new(0, 1),
            PcuBindingRef::new(0, 2),
        ];
        let mut output = [0_u32; 2];
        let good = [
            PcuHostArgument::read(refs[0], &[1_u32, 2]),
            PcuHostArgument::read(refs[1], &[3_u32, 4]),
            PcuHostArgument::read_write(refs[2], &mut output),
        ];
        assert_eq!(validate(&refs, PcuScalarType::U32, &[8; 3], &good), Ok(()));
        assert_eq!(
            validate(&refs, PcuScalarType::U32, &[9; 3], &good),
            Err(PcuHostDispatchError::BufferTooSmall(refs[0]))
        );
        assert_eq!(
            validate(&refs, PcuScalarType::U32, &[8; 3], &good[..2]),
            Err(PcuHostDispatchError::Missing(refs[2]))
        );
        let wrong_type = [PcuHostArgument::read(refs[0], &[1_i32])];
        assert_eq!(
            validate(&refs, PcuScalarType::U32, &[4; 3], &wrong_type),
            Err(PcuHostDispatchError::TypeMismatch(refs[0]))
        );
        let duplicate = [
            PcuHostArgument::read(refs[0], &[1_u32]),
            PcuHostArgument::read(refs[0], &[2_u32]),
        ];
        assert_eq!(
            validate(&refs, PcuScalarType::U32, &[4; 3], &duplicate),
            Err(PcuHostDispatchError::Duplicate(refs[0]))
        );
        let wrong_access = [PcuHostArgument::read(refs[2], &[1_u32])];
        assert_eq!(
            validate(&refs, PcuScalarType::U32, &[4; 3], &wrong_access),
            Err(PcuHostDispatchError::AccessMismatch(refs[2]))
        );
    }
}

#[path = "mixed/mixed.rs"]
mod mixed;
pub use mixed::MetalMixedHostArgument;

/// Static admitted executor with truthful one- or two-output binding arity.
pub enum MetalPreparedHostKernel {
    Single(MetalPreparedSingleHostKernel),
    DivRem(crate::MetalPreparedDivRemHostKernel),
    DivRemRoles(crate::MetalPreparedDivRemRoleHostKernel),
    Transport(crate::MetalPreparedTransportHostKernel),
    Composed(crate::MetalPreparedCheckedMapHostKernel),
}
impl PcuHostKernelBackend for MetalSession {
    type Prepared = MetalPreparedHostKernel;
    type Error = MetalHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let body = match kernel.ops {
            [fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. }, _] => *body,
            ops => ops,
        };
        if body.iter().any(|op| {
            matches!(
                op,
                fusion_pcu::PcuDispatchOp::Data(
                    fusion_pcu::PcuDispatchDataOp::CheckedDivRem { .. }
                )
            )
        }) {
            self.prepare_joint_host_kernel(kernel)
        } else if !crate::admission::carrier::is_carrier_kernel(kernel)
            && crate::MetalTransportPlan::assess_kernel(kernel).is_ok()
        {
            self.transport_host_backend()
                .prepare_host_kernel(kernel)
                .map(MetalPreparedHostKernel::Transport)
        } else if let Some(plan) = composed_plan(kernel) {
            self.composed_host_backend()
                .prepare_plan(plan)
                .map(MetalPreparedHostKernel::Composed)
        } else {
            self.prepare_single_host_kernel(kernel)
                .map(MetalPreparedHostKernel::Single)
        }
    }
}
pub fn composed_plan(kernel: &PcuDispatchKernelIr<'_>) -> Option<crate::MetalCheckedMapPlan> {
    if crate::admission::carrier::is_carrier_kernel(kernel)
        || fusion_pcu::describe_checked_float_unary_map(kernel).is_ok()
        || crate::admission::binary::is_checked_binary_kernel(kernel)
        || crate::admission::is_checked_integer_kernel(kernel)
    {
        return None;
    }
    let fusion_pcu::PcuBindingType::Value(fusion_pcu::PcuValueType::Scalar(scalar)) =
        kernel.bindings.first()?.binding_type
    else {
        return None;
    };
    crate::MetalCheckedMapPlan::assess(kernel, scalar).ok()
}
impl PcuPreparedHostKernel for MetalPreparedHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        match self {
            Self::Single(kernel) => kernel.call(arguments),
            Self::DivRem(kernel) => kernel.call(arguments),
            Self::DivRemRoles(kernel) => kernel.call(arguments),
            Self::Transport(kernel) => kernel.call(arguments),
            Self::Composed(kernel) => kernel.call(arguments),
        }
    }
}
impl MetalPreparedHostKernel {
    /// Number of execution arguments; additive role profiles may also accept unread declarations.
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        match self {
            Self::Single(kernel) => kernel.schema.len(),
            Self::DivRem(kernel) => kernel.argument_count(),
            Self::DivRemRoles(kernel) => kernel.argument_count(),
            Self::Transport(kernel) => kernel.argument_count(),
            Self::Composed(kernel) => kernel.argument_count(),
        }
    }
    /// Whether this call may have changed an existing caller destination.
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        match self {
            Self::Single(kernel) => kernel.last_call_may_have_written(),
            Self::DivRem(kernel) => kernel.last_call_may_have_written(),
            Self::DivRemRoles(kernel) => kernel.last_call_may_have_written(),
            Self::Transport(kernel) => kernel.last_call_may_have_written(),
            Self::Composed(kernel) => kernel.last_call_may_have_written(),
        }
    }
    /// Whether actual native completion is quarantined.
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        match self {
            Self::Single(kernel) => kernel.last_call_completion_uncertain(),
            Self::DivRem(kernel) => kernel.last_call_completion_uncertain(),
            Self::DivRemRoles(kernel) => kernel.last_call_completion_uncertain(),
            Self::Transport(kernel) => kernel.last_call_completion_uncertain(),
            Self::Composed(kernel) => kernel.last_call_completion_uncertain(),
        }
    }
    /// Runs the retained schema with explicit host or exact-session resident borrows.
    /// # Errors
    /// Returns binding, affinity, arithmetic or native completion errors.
    pub fn call_mixed(
        &mut self,
        arguments: &mut [MetalMixedHostArgument<'_>],
    ) -> Result<(), MetalHostKernelError> {
        match self {
            Self::Single(kernel) => kernel.call_mixed(arguments),
            Self::DivRem(kernel) => kernel.call_mixed(arguments),
            Self::DivRemRoles(kernel) => kernel.call_mixed(arguments),
            Self::Transport(kernel) => kernel.call_mixed(arguments),
            Self::Composed(kernel) => kernel.call_mixed(arguments),
        }
    }
}

impl MetalSession {
    pub(crate) fn prepare_joint_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<MetalPreparedHostKernel, MetalHostKernelError> {
        let plan =
            crate::MetalDivRemRolePlan::assess(kernel).map_err(PcuHostDispatchError::Backend)?;
        if kernel
            .numerical_requirements
            .numerical_options
            .reproducibility
            == fusion_pcu::PcuReproducibility::Unspecified
            && fusion_pcu::validate_integer_checked_div_rem_kernel(
                kernel,
                fusion_pcu::PcuValueType::Scalar(plan.scalar_type()),
                fusion_pcu::PcuValueTypeCaps::for_scalar(plan.scalar_type()),
            )
            .is_ok()
        {
            self.checked_div_rem_backend()
                .prepare_host_kernel(kernel)
                .map(MetalPreparedHostKernel::DivRem)
        } else {
            self.checked_div_rem_role_backend()
                .prepare_host_kernel(kernel)
                .map(MetalPreparedHostKernel::DivRemRoles)
        }
    }
}
