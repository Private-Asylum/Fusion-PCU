//! Fresh private transport banks and atomic semantic host publication.

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
    PcuScalarTransportResource,
};
#[rustfmt::skip]
use super::{
    MetalBuffer,
    MetalError,
    MetalPreparedTransportKernel,
    MetalSession,
    MetalTransportPlan,
};
use crate::MetalHostKernelError;

#[path = "mixed/mixed.rs"]
mod mixed;

/// Separate exact transport factory; no arithmetic capability is inferred.
pub struct MetalTransportHostBackend {
    session: MetalSession,
}

/// Original declared schema, byte-SSA pipeline and cold sibling-readback storage.
pub struct MetalPreparedTransportHostKernel {
    kernel: MetalPreparedTransportKernel,
    readback: [Vec<u8>; 4],
    may_have_written: bool,
}
impl MetalSession {
    #[must_use]
    pub fn transport_host_backend(&self) -> MetalTransportHostBackend {
        MetalTransportHostBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MetalTransportHostBackend {
    type Prepared = MetalPreparedTransportHostKernel;
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
            MetalTransportPlan::assess(kernel, scalar).map_err(PcuHostDispatchError::Backend)?;
        let readback = core::array::from_fn(|slot| {
            let count = plan
                .resources()
                .get(slot)
                .map_or(0, |resource| resource.minimum_write_elements);
            vec![0; usize::try_from(count).unwrap_or(0) * plan.width]
        });
        Ok(MetalPreparedTransportHostKernel {
            kernel: self
                .session
                .prepare_transport_plan(plan)
                .map_err(PcuHostDispatchError::Backend)?,
            readback,
            may_have_written: false,
        })
    }
}
impl MetalPreparedTransportHostKernel {
    #[must_use]
    pub const fn plan(&self) -> &MetalTransportPlan {
        self.kernel.plan()
    }

    #[must_use]
    pub const fn argument_count(&self) -> usize {
        self.kernel.plan.declarations.len()
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
        let plan = &self.kernel.plan;
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|prior| prior.target() == target)
            {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let declaration = plan
                .declarations
                .iter()
                .find(|binding| binding.0 == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if argument.scalar() != plan.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if argument.access() != declaration.1 {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let required = plan
                .description
                .resource(target)
                .map_or(0, PcuScalarTransportResource::minimum_elements);
            if argument.bytes().len()
                < usize::try_from(required)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(plan.width)
            {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        for declaration in &plan.declarations {
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
        for (slot, resource) in self.kernel.plan.resources().iter().enumerate() {
            let initial = usize::try_from(resource.minimum_initial_read_elements)
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::InvalidExtent))?
                * self.kernel.plan.width;
            let extent = usize::try_from(resource.minimum_elements())
                .map_err(|_| PcuHostDispatchError::Backend(MetalError::InvalidExtent))?
                * self.kernel.plan.width;
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
impl PcuPreparedHostKernel for MetalPreparedTransportHostKernel {
    type Error = MetalHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        self.validate(arguments)?;
        let staged = self.stage(arguments)?;
        let count = self.kernel.plan.resources().len();
        let references: [&MetalBuffer; 4] =
            core::array::from_fn(|slot| staged[slot.min(count - 1)].as_ref().unwrap());
        self.kernel
            .execute_into(&references[..count])
            .map_err(PcuHostDispatchError::Backend)?;
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
        for (slot, resource) in self.kernel.plan.resources().iter().enumerate() {
            if resource.minimum_write_elements != 0 {
                let output = arguments
                    .iter_mut()
                    .find(|argument| argument.target() == resource.binding)
                    .unwrap();
                let bytes = output.bytes_mut().unwrap();
                bytes[..self.readback[slot].len()].copy_from_slice(&self.readback[slot]);
            }
        }
        Ok(())
    }
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
