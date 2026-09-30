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
    MetalPreparedU32Kernel,
    MetalPreparedF32Kernel,
    MetalBuffer,
    MetalSession,
};

/// Structured binding/admission failure or Metal operational/numerical error.
pub type MetalHostKernelError = PcuHostDispatchError<MetalError>;

/// Reusable owned executable for typed U32 maps and encoding-only F32 unary host calls.
///
/// Every call stages current inputs, allocates fresh device output/status and waits for terminal
/// completion. Host output publication follows checked success; untouched host tails survive.
pub struct MetalPreparedHostKernel {
    session: MetalSession,
    kernel: Program,
    scalar: PcuScalarType,
    schema: Vec<PcuBindingRef>,
    required_bytes: usize,
}
enum Program {
    U32(MetalPreparedU32Kernel),
    F32(MetalPreparedF32Kernel),
}
impl Program {
    const fn scalar(&self) -> PcuScalarType {
        match self {
            Self::U32(_) => PcuScalarType::U32,
            Self::F32(_) => PcuScalarType::F32,
        }
    }
    const fn element_count(&self) -> usize {
        match self {
            Self::U32(kernel) => kernel.element_count(),
            Self::F32(kernel) => kernel.element_count(),
        }
    }
    const fn input_bindings(&self) -> [PcuBindingRef; 2] {
        match self {
            Self::U32(kernel) => kernel.input_bindings(),
            Self::F32(kernel) => [kernel.input_binding(); 2],
        }
    }
    const fn output_binding(&self) -> PcuBindingRef {
        match self {
            Self::U32(kernel) => kernel.output_binding(),
            Self::F32(kernel) => kernel.output_binding(),
        }
    }
    fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        match self {
            Self::U32(kernel) => kernel.execute_into(inputs, output),
            Self::F32(kernel) => kernel.execute_into(inputs[0], output),
        }
    }
    fn execute_prefix(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        match self {
            Self::U32(kernel) => kernel.execute_prefix(inputs),
            Self::F32(kernel) => kernel.execute_prefix(inputs[0]),
        }
    }
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        match self {
            Self::U32(kernel) => kernel.execute(inputs),
            Self::F32(kernel) => kernel.execute(inputs[0]),
        }
    }
}
impl PcuHostKernelBackend for MetalSession {
    type Prepared = MetalPreparedHostKernel;
    type Error = MetalHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        if cfg!(target_endian = "big") {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        }
        let prepared = if kernel.bindings.first().is_some_and(|binding| {
            binding.binding_type
                == fusion_pcu::PcuBindingType::Value(fusion_pcu::PcuValueType::f32())
        }) {
            Program::F32(
                self.prepare_f32_unary_kernel(kernel)
                    .map_err(PcuHostDispatchError::Backend)?,
            )
        } else {
            Program::U32(
                self.prepare_u32_kernel(kernel)
                    .map_err(PcuHostDispatchError::Backend)?,
            )
        };
        let required_bytes = prepared
            .element_count()
            .checked_mul(4)
            .ok_or(PcuHostDispatchError::Backend(MetalError::InvalidExtent))?;
        Ok(MetalPreparedHostKernel {
            session: self.clone(),
            scalar: prepared.scalar(),
            kernel: prepared,
            schema: kernel
                .bindings
                .iter()
                .copied()
                .map(fusion_pcu::PcuBinding::reference)
                .collect(),
            required_bytes,
        })
    }
}
impl PcuPreparedHostKernel for MetalPreparedHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        validate(&self.schema, self.scalar, self.required_bytes, arguments)?;
        let references = self.kernel.input_bindings();
        let input = |reference| {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == reference)
                .ok_or(PcuHostDispatchError::Missing(reference))?;
            self.session
                .upload_bytes(&argument.bytes()[..self.required_bytes])
                .map_err(PcuHostDispatchError::Backend)
        };
        let left = input(references[0])?;
        let right = if references[0] == references[1] {
            None
        } else {
            Some(input(references[1])?)
        };
        let output = self
            .kernel
            .execute([&left, right.as_ref().unwrap_or(&left)])
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
            .map_err(PcuHostDispatchError::Backend)
    }
}
fn validate(
    schema: &[PcuBindingRef],
    scalar: PcuScalarType,
    required_bytes: usize,
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
        if argument.bytes().len() < required_bytes {
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
        assert_eq!(validate(&refs, PcuScalarType::U32, 8, &good), Ok(()));
        assert_eq!(
            validate(&refs, PcuScalarType::U32, 9, &good),
            Err(PcuHostDispatchError::BufferTooSmall(refs[0]))
        );
        assert_eq!(
            validate(&refs, PcuScalarType::U32, 8, &good[..2]),
            Err(PcuHostDispatchError::Missing(refs[2]))
        );
        let wrong_type = [PcuHostArgument::read(refs[0], &[1_i32])];
        assert_eq!(
            validate(&refs, PcuScalarType::U32, 4, &wrong_type),
            Err(PcuHostDispatchError::TypeMismatch(refs[0]))
        );
        let duplicate = [
            PcuHostArgument::read(refs[0], &[1_u32]),
            PcuHostArgument::read(refs[0], &[2_u32]),
        ];
        assert_eq!(
            validate(&refs, PcuScalarType::U32, 4, &duplicate),
            Err(PcuHostDispatchError::Duplicate(refs[0]))
        );
        let wrong_access = [PcuHostArgument::read(refs[2], &[1_u32])];
        assert_eq!(
            validate(&refs, PcuScalarType::U32, 4, &wrong_access),
            Err(PcuHostDispatchError::AccessMismatch(refs[2]))
        );
    }
}

#[path = "mixed/mixed.rs"]
mod mixed;
pub use mixed::MetalMixedHostArgument;
