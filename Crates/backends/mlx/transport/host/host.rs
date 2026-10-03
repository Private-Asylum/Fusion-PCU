//! Exact declared host schema, minimum initial uploads and joint private readback publication.
#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingType,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostDispatchError,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuScalarTransportResource,
    PcuValueType,
};
#[rustfmt::skip]
use crate::{
    MlxEncodedArray,
    MlxError,
    MlxHostKernelError,
    MlxPreparedTransportKernel,
    MlxSession,
    MlxTransportPlan,
};
#[path = "mixed/mixed.rs"]
mod mixed;
pub use mixed::MlxTransportInput;

/// Explicit transport factory; scalar arithmetic and aggregate offers remain separate.
pub struct MlxTransportHostBackend {
    session: MlxSession,
}
/// Cold exact byte-SSA primitive plus reusable private sibling readback storage.
pub struct MlxPreparedTransportHostKernel {
    kernel: MlxPreparedTransportKernel,
    readback: [Vec<u8>; 2],
    may_have_written: bool,
    output_layout: [Option<(fusion_pcu::PcuBindingRef, usize)>; 2],
}
impl MlxSession {
    #[must_use]
    pub fn transport_host_backend(&self) -> MlxTransportHostBackend {
        MlxTransportHostBackend {
            session: self.clone(),
        }
    }
}
impl PcuHostKernelBackend for MlxTransportHostBackend {
    type Prepared = MlxPreparedTransportHostKernel;
    type Error = MlxHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            kernel.bindings.first().map(|b| b.binding_type)
        else {
            return Err(PcuHostDispatchError::Backend(MlxError::InvalidRequest(
                "unsupported transport scalar schema".into(),
            )));
        };
        let plan =
            MlxTransportPlan::assess(kernel, scalar).map_err(PcuHostDispatchError::Backend)?;
        self.prepare_plan(plan, None)
    }
}
impl MlxTransportHostBackend {
    /// Freezes actual resident capacities without widening any logical initial read span.
    /// Host slots must retain their minimum size; runtime argument kind participates in the
    /// caller's cold specialization. Cold native priming uploads full synthetic capacities.
    /// # Errors
    /// Rejects wrong arity, short/overflowing shapes or unproved source before SDK work.
    pub fn prepare_host_kernel_with_input_extents(
        &self,
        source: &PcuDispatchKernelIr<'_>,
        extents: &[usize],
    ) -> Result<MlxPreparedTransportHostKernel, MlxHostKernelError> {
        let Some(PcuBindingType::Value(PcuValueType::Scalar(scalar))) =
            source.bindings.first().map(|binding| binding.binding_type)
        else {
            return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
        };
        let plan =
            MlxTransportPlan::assess(source, scalar).map_err(PcuHostDispatchError::Backend)?;
        let full = plan
            .assess_input_extents(extents)
            .map_err(PcuHostDispatchError::Backend)?;
        self.prepare_plan(plan, Some(full))
    }
    fn prepare_plan(
        &self,
        plan: MlxTransportPlan,
        full: Option<[usize; 4]>,
    ) -> Result<MlxPreparedTransportHostKernel, MlxHostKernelError> {
        let input_count = plan.inputs().len();
        let width = usize::from(plan.scalar_type().bit_width()) / 8;
        let readback = std::array::from_fn(|slot| {
            vec![
                0;
                plan.outputs()
                    .get(slot)
                    .map_or(0, |r| usize::try_from(r.minimum_write_elements).unwrap())
                    * width
            ]
        });
        let output_layout = std::array::from_fn(|slot| {
            plan.outputs().get(slot).map(|resource| {
                (
                    resource.binding,
                    usize::try_from(resource.minimum_write_elements).unwrap() * width,
                )
            })
        });
        Ok(MlxPreparedTransportHostKernel {
            kernel: match full {
                Some(extents) => self
                    .session
                    .prepare_transport_plan_with_input_extents(plan, &extents[..input_count]),
                None => self.session.prepare_transport_plan(plan),
            }
            .map_err(PcuHostDispatchError::Backend)?,
            readback,
            may_have_written: false,
            output_layout,
        })
    }
}
impl MlxPreparedTransportHostKernel {
    #[must_use]
    pub const fn plan(&self) -> &MlxTransportPlan {
        self.kernel.plan()
    }
    #[must_use]
    pub const fn session(&self) -> &MlxSession {
        self.kernel.session()
    }
    /// Exact frozen physical counts, independently of logical original-byte minima.
    #[must_use]
    pub fn prepared_input_element_counts(&self) -> &[usize] {
        self.kernel.prepared_input_element_counts()
    }
    /// Exact writer slots; absent entries do not represent extra native outputs.
    #[must_use]
    pub const fn output_layout(&self) -> [Option<(fusion_pcu::PcuBindingRef, usize)>; 2] {
        self.output_layout
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
        let width = usize::from(plan.scalar_type().bit_width()) / 8;
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index].iter().any(|a| a.target() == target) {
                return Err(PcuHostDispatchError::Duplicate(target));
            }
            let declaration = plan
                .declared_bindings()
                .iter()
                .find(|d| d.0 == target)
                .ok_or(PcuHostDispatchError::Unexpected(target))?;
            if argument.scalar() != plan.scalar_type() {
                return Err(PcuHostDispatchError::TypeMismatch(target));
            }
            if argument.access() != declaration.1 {
                return Err(PcuHostDispatchError::AccessMismatch(target));
            }
            let count = plan
                .resources()
                .iter()
                .find(|r| r.binding == target)
                .copied()
                .map_or(0, PcuScalarTransportResource::minimum_elements);
            if argument.bytes().len()
                < usize::try_from(count)
                    .unwrap_or(usize::MAX)
                    .saturating_mul(width)
            {
                return Err(PcuHostDispatchError::BufferTooSmall(target));
            }
        }
        for (target, _) in plan.declared_bindings() {
            if !arguments.iter().any(|a| a.target() == *target) {
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
        let width = usize::from(self.plan().scalar_type().bit_width()) / 8;
        for (slot, resource) in self.plan().inputs().iter().enumerate() {
            let count = usize::try_from(resource.minimum_initial_read_elements)
                .map_err(|_| PcuHostDispatchError::Backend(MlxError::InvalidExtent))?;
            let argument = arguments
                .iter()
                .find(|a| a.target() == resource.binding)
                .ok_or(PcuHostDispatchError::Missing(resource.binding))?;
            staged[slot] = Some(
                self.session()
                    .upload_transport_bytes(
                        self.plan().scalar_type(),
                        count,
                        &argument.bytes()[..count * width],
                    )
                    .map_err(PcuHostDispatchError::Backend)?,
            );
        }
        Ok(staged)
    }
}
impl PcuPreparedHostKernel for MlxPreparedTransportHostKernel {
    type Error = MlxHostKernelError;
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.may_have_written = false;
        self.validate(arguments)?;
        if self.prepared_input_element_counts()
            != &self.plan().input_element_counts()[..self.plan().inputs().len()]
        {
            return Err(PcuHostDispatchError::Backend(MlxError::InvalidExtent));
        }
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
            .execute(&inputs[..self.plan().inputs().len()])
            .map_err(PcuHostDispatchError::Backend)?;
        for (slot, output) in completed.outputs().iter().enumerate() {
            if let Some(output) = output {
                output
                    .read_bytes_into(&mut self.readback[slot])
                    .map_err(PcuHostDispatchError::Backend)?;
            }
        }
        // Every fallible sibling read AND explicit native cleanup precedes both public copies.
        for output in completed.into_outputs().into_iter().flatten() {
            output.release().map_err(PcuHostDispatchError::Backend)?;
        }
        for input in staged.into_iter().flatten() {
            input.release().map_err(PcuHostDispatchError::Backend)?;
        }
        self.may_have_written = true;
        for (slot, resource) in self.kernel.plan().outputs().iter().enumerate() {
            let output = arguments
                .iter_mut()
                .find(|a| a.target() == resource.binding)
                .unwrap();
            output.bytes_mut().unwrap()[..self.readback[slot].len()]
                .copy_from_slice(&self.readback[slot]);
        }
        Ok(())
    }
}
