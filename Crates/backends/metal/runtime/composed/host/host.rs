//! Private ordered checked banks and atomic sibling host publication.

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuValueType,
    CheckedScalarMapResource,
};
#[rustfmt::skip]
use super::{
    MetalBuffer,
    MetalError,
    MetalPreparedCheckedMapKernel,
    MetalSession,
    MetalCheckedMapPlan,
};
use crate::MetalHostKernelError;

#[path = "mixed/mixed.rs"]
mod mixed;

/// Separate bounded ordered-map factory; no aggregate capability is inferred.
pub struct MetalComposedHostBackend {
    session: MetalSession,
}

/// Original declarations, ordered checked effects and cold sibling-readback storage.
pub struct MetalPreparedCheckedMapHostKernel {
    kernel: MetalPreparedCheckedMapKernel,
    readback: [Vec<u8>; 4],
    may_have_written: bool,
}
impl MetalSession {
    /// Borrow this actual native session for a bounded private host-map factory.
    /// The factory preserves the original request and each checked effect.
    #[must_use]
    pub fn composed_host_backend(&self) -> MetalComposedHostBackend {
        MetalComposedHostBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MetalComposedHostBackend {
    type Prepared = MetalPreparedCheckedMapHostKernel;
    type Error = MetalHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(PcuHostDispatchError::Backend(MetalError::Unsupported));
        };
        let plan =
            MetalCheckedMapPlan::assess(kernel, scalar).map_err(PcuHostDispatchError::Backend)?;
        self.prepare_plan(plan)
    }
}
impl MetalComposedHostBackend {
    /// Prepare an already validated exact detached plan; no native failure falls back.
    /// # Errors
    /// Returns exact emission, native compilation/allocation or quarantine failures.
    pub fn prepare_plan(
        &self,
        plan: MetalCheckedMapPlan,
    ) -> Result<MetalPreparedCheckedMapHostKernel, MetalHostKernelError> {
        let readback = core::array::from_fn(|slot| {
            let count = plan
                .resources()
                .get(slot)
                .map_or(0, |resource| resource.minimum_write_elements);
            vec![0; usize::try_from(count).unwrap_or(0) * scalar_width(&plan)]
        });
        Ok(MetalPreparedCheckedMapHostKernel {
            kernel: self
                .session
                .prepare_checked_map_plan(plan)
                .map_err(PcuHostDispatchError::Backend)?,
            readback,
            may_have_written: false,
        })
    }
}
impl MetalPreparedCheckedMapHostKernel {
    #[must_use]
    pub const fn plan(&self) -> &MetalCheckedMapPlan {
        self.kernel.plan()
    }

    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.kernel.plan().declared_bindings().len()
    }

    #[must_use]
    pub const fn last_call_may_have_written(&self) -> bool {
        self.may_have_written
    }

    #[must_use]
    pub fn last_call_completion_uncertain(&self) -> bool {
        self.kernel.session.ensure_quiescent().is_err()
    }

    fn validate(&self, arguments: &[PcuHostArgument<'_>]) -> Result<(), MetalHostKernelError> {
        let plan = self.kernel.plan();
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
            let required = plan
                .resources()
                .iter()
                .find(|resource| resource.binding == target)
                .copied()
                .map_or(0, CheckedScalarMapResource::minimum_elements);
            if argument.bytes().len()
                < usize::try_from(required)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(scalar_width(plan))
            {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        for declaration in plan.declared_bindings() {
            if !arguments
                .iter()
                .any(|argument| argument.target() == declaration.0)
            {
                return Err(PcuHostDispatchError::Missing(declaration.0));
            }
        }
        Ok(())
    }

    fn stage(
        &self,
        arguments: &[PcuHostArgument<'_>],
    ) -> Result<[Option<MetalBuffer>; 4], MetalHostKernelError> {
        let mut staged = core::array::from_fn(|_| None);
        for (slot, resource) in self.kernel.plan().resources().iter().enumerate() {
            let initial = usize::try_from(resource.minimum_initial_read_elements)
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::InvalidExtent))?
                * scalar_width(self.kernel.plan());
            let extent = usize::try_from(resource.minimum_elements())
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::InvalidExtent))?
                * scalar_width(self.kernel.plan());
            let mut buffer = self
                .kernel
                .session
                .allocate_zeroed_bytes(extent)
                .map_err(PcuHostDispatchError::Backend)?;
            if initial != 0 {
                let argument = argument(arguments, resource.binding)?;
                buffer
                    .write_bytes(0, &argument.bytes()[..initial])
                    .map_err(PcuHostDispatchError::Backend)?;
            }
            staged[slot] = Some(buffer);
        }
        Ok(staged)
    }
}
impl PcuPreparedHostKernel for MetalPreparedCheckedMapHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        self.validate(arguments)?;
        let staged = self.stage(arguments)?;
        let count = self.kernel.plan().resources().len();
        let references: [&MetalBuffer; 4] =
            core::array::from_fn(|slot| staged[slot.min(count - 1)].as_ref().unwrap());
        let notice = match self.kernel.execute_into(&references[..count]) {
            Ok(()) => None,
            Err(MetalError::Arithmetic(fault)) if fault.recovered => Some(fault),
            Err(error) => return Err(PcuHostDispatchError::Backend(error)),
        };
        // Every fallible sibling read finishes before any caller byte changes.
        for (slot, readback) in self.readback.iter_mut().enumerate() {
            if !readback.is_empty() {
                staged[slot]
                    .as_ref()
                    .unwrap()
                    .read_bytes(0, readback)
                    .map_err(PcuHostDispatchError::Backend)?;
            }
        }
        self.may_have_written = true;
        for (slot, resource) in self.kernel.plan().resources().iter().enumerate() {
            if resource.minimum_write_elements != 0 {
                let output = arguments
                    .iter_mut()
                    .find(|argument| argument.target() == resource.binding)
                    .unwrap();
                let bytes = output.bytes_mut().unwrap();
                bytes[..self.readback[slot].len()].copy_from_slice(&self.readback[slot]);
            }
        }
        notice.map_or(Ok(()), |fault| {
            Err(PcuHostDispatchError::Backend(MetalError::Arithmetic(fault)))
        })
    }
}
const fn scalar_width(plan: &MetalCheckedMapPlan) -> usize {
    plan.value_type().scalar_type().bit_width() as usize / 8
}
fn argument<'argument, 'data>(
    arguments: &'argument [PcuHostArgument<'data>],
    target: PcuBindingRef,
) -> Result<&'argument PcuHostArgument<'data>, MetalHostKernelError> {
    arguments
        .iter()
        .find(|argument| argument.target() == target)
        .ok_or(PcuHostDispatchError::Missing(target))
}
