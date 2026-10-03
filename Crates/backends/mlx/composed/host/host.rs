//! Original host declarations and terminal private siblings before any public copy.
#[rustfmt::skip]
use fusion_pcu::{
    CheckedScalarMapResource,
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuValueType,
};
#[rustfmt::skip]
use crate::{
    MlxCheckedMapPlan,
    MlxEncodedArray,
    MlxError,
    MlxHostKernelError,
    MlxPreparedCheckedMapKernel,
    MlxSession,
};

#[path = "mixed/mixed.rs"]
mod mixed;
pub use mixed::MlxCheckedMapInput;

/// Separate bounded checked host factory; no aggregate or ordinary admission grant.
pub struct MlxComposedHostBackend {
    session: MlxSession,
}
/// Original schema, retained primitive and reusable cold sibling readback storage.
pub struct MlxPreparedCheckedMapHostKernel {
    kernel: MlxPreparedCheckedMapKernel,
    readback: [Vec<u8>; 2],
    may_have_written: bool,
    output_layout: [Option<(fusion_pcu::PcuBindingRef, usize)>; 2],
}
impl MlxSession {
    /// Borrow this exact native session for a checked private host-map factory.
    #[must_use]
    pub fn composed_host_backend(&self) -> MlxComposedHostBackend {
        MlxComposedHostBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MlxComposedHostBackend {
    type Prepared = MlxPreparedCheckedMapHostKernel;
    type Error = MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        source: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            source.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
        };
        let plan =
            MlxCheckedMapPlan::assess(source, scalar).map_err(PcuHostDispatchError::Backend)?;
        let width = usize::from(scalar.bit_width()) / 8;
        let kernel = self
            .session
            .prepare_checked_map_plan(plan)
            .map_err(PcuHostDispatchError::Backend)?;
        let output_layout = std::array::from_fn(|slot| {
            kernel.output_bindings().get(slot).map(|&binding| {
                let role = kernel
                    .plan()
                    .resources()
                    .iter()
                    .find(|resource| resource.binding == binding)
                    .unwrap();
                (
                    binding,
                    usize::try_from(role.minimum_write_elements).unwrap_or(0) * width,
                )
            })
        });
        let readback = std::array::from_fn(|slot| {
            let bytes = kernel
                .output_bindings()
                .get(slot)
                .and_then(|binding| {
                    kernel
                        .plan()
                        .resources()
                        .iter()
                        .find(|resource| resource.binding == *binding)
                })
                .map_or(0, |resource| {
                    usize::try_from(resource.minimum_write_elements).unwrap_or(0) * width
                });
            vec![0; bytes]
        });
        Ok(MlxPreparedCheckedMapHostKernel {
            kernel,
            readback,
            may_have_written: false,
            output_layout,
        })
    }
}
impl MlxPreparedCheckedMapHostKernel {
    #[must_use]
    pub const fn plan(&self) -> &MlxCheckedMapPlan {
        self.kernel.plan()
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        self.kernel.session()
    }
    /// Exact actual writer cardinality and logical bytes, retained cold.
    #[must_use]
    pub const fn output_layout(&self) -> [Option<(fusion_pcu::PcuBindingRef, usize)>; 2] {
        self.output_layout
    }
    /// Actual first-access bindings requiring original input contents, not all declarations.
    #[must_use]
    pub const fn input_bindings(&self) -> &[fusion_pcu::PcuBindingRef] {
        self.kernel.input_bindings()
    }
    /// Exact native physical shapes retained during cold priming.
    #[must_use]
    pub fn prepared_input_element_counts(&self) -> &[usize] {
        self.kernel.prepared_input_element_counts()
    }
    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.plan().declared_bindings().len()
    }
    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }
    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.session().validate_access_available().is_err()
    }
    fn validate(&self, arguments: &[PcuHostArgument<'_>]) -> Result<(), MlxHostKernelError> {
        self.session()
            .validate_access_available()
            .map_err(PcuHostDispatchError::Backend)?;
        let plan = self.plan();
        let width = usize::from(plan.value_type().scalar_type().bit_width()) / 8;
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let declaration = plan
                .declared_bindings()
                .iter()
                .find(|binding| binding.0 == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if argument.scalar() != plan.value_type().scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if argument.access() != declaration.1 {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let count = plan
                .resources()
                .iter()
                .find(|resource| resource.binding == target)
                .copied()
                .map_or(0, CheckedScalarMapResource::minimum_elements);
            if argument.bytes().len()
                < usize::try_from(count)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(width)
            {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        for (target, _) in plan.declared_bindings() {
            if !arguments
                .iter()
                .any(|argument| argument.target() == *target)
            {
                return Err(PcuHostDispatchError::Missing(*target));
            }
        }
        Ok(())
    }
    fn stage(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[Option<MlxEncodedArray>; 4], MlxHostKernelError> {
        let mut staged = std::array::from_fn(|_| None);
        let scalar = self.plan().value_type().scalar_type();
        let width = usize::from(scalar.bit_width()) / 8;
        for (slot, (&binding, &count)) in self
            .kernel
            .input_bindings()
            .iter()
            .zip(self.kernel.prepared_input_element_counts())
            .enumerate()
        {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == binding)
                .ok_or(PcuHostDispatchError::Missing(binding))?;
            staged[slot] = Some(
                self.session()
                    .upload_transport_bytes(scalar, count, &argument.bytes()[..count * width])
                    .map_err(PcuHostDispatchError::Backend)?,
            );
        }
        Ok(staged)
    }
}
impl PcuPreparedHostKernel for MlxPreparedCheckedMapHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        self.validate(arguments)?;
        let staged = self.stage(arguments)?;
        let first = staged[0]
            .as_ref()
            .ok_or(PcuHostDispatchError::Backend(MlxError::InvalidExtent))?;
        let mut inputs = [first; 4];
        for (slot, input) in staged.iter().enumerate() {
            if let Some(input) = input {
                inputs[slot] = input;
            }
        }
        let completed = self
            .kernel
            .execute(&inputs[..self.kernel.input_bindings().len()])
            .map_err(PcuHostDispatchError::Backend)?;
        let (outputs, notice) = completed.into_outputs();
        for (slot, output) in outputs.iter().enumerate() {
            if let Some(output) = output {
                output
                    .read_bytes_into(&mut self.readback[slot])
                    .map_err(PcuHostDispatchError::Backend)?;
            }
        }
        // Both private reads and every explicit native release precede either public copy.
        for output in outputs.into_iter().flatten() {
            output.release().map_err(PcuHostDispatchError::Backend)?;
        }
        for input in staged.into_iter().flatten() {
            input.release().map_err(PcuHostDispatchError::Backend)?;
        }
        self.may_have_written = true;
        for (slot, &binding) in self.kernel.output_bindings().iter().enumerate() {
            let output = arguments
                .iter_mut()
                .find(|argument| argument.target() == binding)
                .unwrap();
            output.bytes_mut().unwrap()[..self.readback[slot].len()]
                .copy_from_slice(&self.readback[slot]);
        }
        notice.map_or(Ok(()), |fault| {
            Err(PcuHostDispatchError::Backend(MlxError::Arithmetic(fault)))
        })
    }
}
