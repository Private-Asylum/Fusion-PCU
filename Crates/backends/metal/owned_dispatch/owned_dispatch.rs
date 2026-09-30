//! Synchronous neutral owned dispatch over bounded checked Metal maps.

#[rustfmt::skip]
use std::{
    cell::RefCell,
    rc::Rc,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuBaseContract,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuCompletionState,
    PcuDeviceIdentity,
    PcuDispatchSubmission,
    PcuExecutorDescriptor,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuKernelIrContract,
    PcuMemoryAccess,
    PcuMemoryPoolId,
    PcuMemoryResource,
    PcuOwnedBinding,
    PcuOwnedBindingRequirement,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchBindingError,
    PcuOwnedDispatchBindingSchemaContract,
    PcuOwnedDispatchMemorySession,
    PcuPreparedOwnedDispatch,
    PcuSupport,
    PcuValueType,
    validate_dispatch_submission,
    validate_parameters,
};
#[rustfmt::skip]
use crate::{
    MetalBuffer,
    MetalError,
    MetalMemoryProvider,
    MetalMemoryResource,
    MetalPreparedF32Kernel,
    MetalPreparedU32Kernel,
    MetalSession,
};
#[path = "caps/caps.rs"]
mod caps;
pub const fn discovery_support() -> PcuSupport {
    caps::support()
}
pub const fn executor_descriptors() -> &'static [PcuExecutorDescriptor] {
    &caps::EXECUTORS
}
#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;

/// Selected device session with synchronous owned dispatch and neutral shared memory services.
///
/// Obtain this through `MetalDiscovery::open_owned_device`; identities are discovery bound.
pub struct MetalOwnedDispatchBackend {
    session: MetalSession,
    device: PcuDeviceIdentity,
}
impl MetalOwnedDispatchBackend {
    pub(crate) const fn new(session: MetalSession, device: PcuDeviceIdentity) -> Self {
        Self { session, device }
    }
    #[must_use]
    pub const fn session(&self) -> &MetalSession {
        &self.session
    }
}

/// Private allocation lease. No native pointers or unchecked mutable access escape.
pub struct MetalOwnedResource {
    buffer: Rc<RefCell<MetalBuffer>>,
    access: PcuBindingAccess,
    binding_type: PcuBindingType,
}

/// Common admission, owned binding, or native operational failure.
#[derive(Debug)]
pub enum MetalOwnedDispatchError {
    Admission(fusion_pcu::PcuError),
    Binding(PcuOwnedDispatchBindingError),
    Metal(MetalError),
    Memory(fusion_pcu::PcuMemoryProviderError),
}
impl std::fmt::Display for MetalOwnedDispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}
impl std::error::Error for MetalOwnedDispatchError {}
impl From<MetalError> for MetalOwnedDispatchError {
    fn from(error: MetalError) -> Self {
        Self::Metal(error)
    }
}

enum Program {
    U32(MetalPreparedU32Kernel),
    F32(MetalPreparedF32Kernel),
}
impl Program {
    const fn inputs(&self) -> [PcuBindingRef; 2] {
        match self {
            Self::U32(kernel) => kernel.input_bindings(),
            Self::F32(kernel) => [kernel.input_binding(); 2],
        }
    }
    const fn output(&self) -> PcuBindingRef {
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
    fn execute(&self, inputs: [&MetalBuffer; 2]) -> Result<MetalBuffer, MetalError> {
        match self {
            Self::U32(kernel) => kernel.execute_prefix(inputs),
            Self::F32(kernel) => kernel.execute_prefix(inputs[0]),
        }
    }
}

/// Prepared checked map owning its executable and binding schema independently of source IR.
pub struct MetalPreparedDispatch {
    session: MetalSession,
    device: PcuDeviceIdentity,
    shape: PcuInvocationShape,
    requirements: Vec<PcuOwnedBindingRequirement>,
    program: Program,
}
impl MetalPreparedDispatch {
    fn validate_resources(
        &self,
        bindings: &[PcuOwnedBinding<MetalOwnedResource>],
    ) -> Result<(), MetalOwnedDispatchError> {
        self.requirements
            .as_slice()
            .validate(self.device, bindings)
            .map_err(MetalOwnedDispatchError::Binding)?;
        for binding in bindings {
            let buffer = binding.resource.buffer.borrow();
            if !self.session.same_session(buffer.session()) {
                return Err(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::WrongDevice(binding.target),
                ));
            }
            if binding.byte_len != buffer.len() as u64 * 4 {
                return Err(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::BufferTooSmall {
                        binding: binding.target,
                        required: binding.byte_len,
                        available: buffer.len() as u64 * 4,
                    },
                ));
            }
            if binding.access != binding.resource.access {
                return Err(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::AccessMismatch(binding.target),
                ));
            }
            if binding.binding_type != binding.resource.binding_type {
                return Err(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::TypeMismatch(binding.target),
                ));
            }
        }
        self.session.ensure_quiescent()?;
        Ok(())
    }
    fn execute(
        &self,
        bindings: &[PcuOwnedBinding<MetalOwnedResource>],
    ) -> Result<PcuCompletionOutcome, MetalOwnedDispatchError> {
        self.validate_resources(bindings)?;
        let find = |target| {
            bindings
                .iter()
                .find(|binding| binding.target == target)
                .ok_or(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::Missing(target),
                ))
        };
        let inputs = self.program.inputs();
        let left = find(inputs[0])?.resource.buffer.borrow();
        let right = find(inputs[1])?.resource.buffer.borrow();
        // Inputs remain borrowed through terminal completion. The fresh output shields aliases.
        let result = self.program.execute([&left, &right]);
        drop(right);
        drop(left);
        let output = match result {
            Ok(output) => output,
            Err(MetalError::Arithmetic(fault)) => return Ok(PcuCompletionOutcome::Fault(fault)),
            Err(error) => return Err(error.into()),
        };
        let words = output.download_u32()?;
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
        find(self.program.output())?
            .resource
            .buffer
            .borrow_mut()
            .write_bytes(0, &bytes)?;
        Ok(PcuCompletionOutcome::Succeeded)
    }
}

/// Already terminal synchronous completion, retaining its submitted binding leases.
pub struct MetalOwnedCompletion {
    outcome: PcuCompletionOutcome,
    _bindings: Vec<PcuOwnedBinding<MetalOwnedResource>>,
}
impl PcuOwnedCompletion for MetalOwnedCompletion {
    type Error = MetalError;
    fn state(&self) -> Result<PcuCompletionState, MetalError> {
        Ok(match self.outcome {
            PcuCompletionOutcome::Succeeded => PcuCompletionState::Succeeded,
            PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_) => {
                PcuCompletionState::Failed
            }
        })
    }
    fn wait(&mut self) -> Result<PcuCompletionOutcome, MetalError> {
        Ok(self.outcome)
    }
}
impl PcuBaseContract for MetalOwnedDispatchBackend {
    fn support(&self) -> PcuSupport {
        caps::support()
    }
    fn executors(&self) -> &'static [PcuExecutorDescriptor] {
        &caps::EXECUTORS
    }
}
impl fusion_pcu::PcuHostKernelBackend for MetalOwnedDispatchBackend {
    type Prepared = crate::MetalPreparedHostKernel;
    type Error = crate::MetalHostKernelError;
    fn prepare_host_kernel(
        &self,
        kernel: &fusion_pcu::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        self.session.prepare_host_kernel(kernel)
    }
}
impl PcuOwnedDispatchBackend for MetalOwnedDispatchBackend {
    type Resource = MetalOwnedResource;
    type Bindings = Vec<PcuOwnedBinding<MetalOwnedResource>>;
    type Completion = MetalOwnedCompletion;
    type Error = MetalOwnedDispatchError;
    type Prepared<'kernel, 'parameters>
        = MetalPreparedDispatch
    where
        Self: 'kernel;
    fn device_identity(&self) -> PcuDeviceIdentity {
        self.device
    }
    fn submit_dispatch_owned_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: Self::Bindings,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::Completion, Self::Error> {
        self.prepare_dispatch_owned_direct(submission, parameters)?
            .submit_owned_direct(bindings)
    }
    fn prepare_dispatch_owned_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<MetalPreparedDispatch, Self::Error> {
        validate_dispatch_submission(submission).map_err(MetalOwnedDispatchError::Admission)?;
        validate_parameters(submission.kernel.signature(), parameters)
            .map_err(MetalOwnedDispatchError::Admission)?;
        if !parameters.is_empty()
            || !submission.kernel.ports.is_empty()
            || cfg!(target_endian = "big")
        {
            return Err(MetalError::Unsupported.into());
        }
        let program = if submission
            .kernel
            .bindings
            .iter()
            .all(|binding| binding.binding_type == PcuBindingType::Value(PcuValueType::u32()))
        {
            Program::U32(self.session.prepare_u32_kernel(submission.kernel)?)
        } else {
            Program::F32(self.session.prepare_f32_unary_kernel(submission.kernel)?)
        };
        let requirements = submission
            .kernel
            .bindings
            .iter()
            .map(|binding| {
                PcuOwnedBindingRequirement::from_verified_binding(
                    submission.kernel,
                    binding.reference(),
                    submission.shape,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(MetalOwnedDispatchError::Binding)?;
        Ok(MetalPreparedDispatch {
            session: self.session.clone(),
            device: self.device,
            shape: submission.shape,
            requirements,
            program,
        })
    }
}
impl PcuPreparedOwnedDispatch for MetalPreparedDispatch {
    type Resource = MetalOwnedResource;
    type Bindings = Vec<PcuOwnedBinding<MetalOwnedResource>>;
    type BindingSchema = [PcuOwnedBindingRequirement];
    type Completion = MetalOwnedCompletion;
    type Error = MetalOwnedDispatchError;
    fn binding_schema(&self) -> &Self::BindingSchema {
        &self.requirements
    }
    fn shape(&self) -> PcuInvocationShape {
        self.shape
    }
    fn device_identity(&self) -> PcuDeviceIdentity {
        self.device
    }
    fn submit_owned_direct(
        &self,
        bindings: Self::Bindings,
    ) -> Result<Self::Completion, Self::Error> {
        let outcome = self.execute(&bindings)?;
        Ok(MetalOwnedCompletion {
            outcome,
            _bindings: bindings,
        })
    }
}
impl PcuOwnedDispatchMemorySession for MetalOwnedDispatchBackend {
    type MemoryProvider = MetalMemoryProvider;
    fn memory_provider(&self, pool: PcuMemoryPoolId) -> MetalMemoryProvider {
        self.session.memory_provider(pool)
    }
    fn bind(
        &self,
        target: PcuBindingRef,
        access: PcuBindingAccess,
        binding_type: PcuBindingType,
        resource: &MetalMemoryResource,
    ) -> Result<PcuOwnedBinding<MetalOwnedResource>, Self::Error> {
        if !matches!(binding_type, PcuBindingType::Value(value) if value == PcuValueType::u32() || value == PcuValueType::f32())
        {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::TypeMismatch(target),
            ));
        }
        let allowed = match access {
            PcuBindingAccess::ReadOnly => resource.access() != PcuMemoryAccess::WriteOnly,
            PcuBindingAccess::WriteOnly => resource.access() != PcuMemoryAccess::ReadOnly,
            PcuBindingAccess::ReadWrite => resource.access() == PcuMemoryAccess::ReadWrite,
        };
        if !allowed {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::AccessMismatch(target),
            ));
        }
        let buffer = resource.lease();
        if !self.session.same_session(buffer.borrow().session()) {
            return Err(MetalOwnedDispatchError::Binding(
                PcuOwnedDispatchBindingError::WrongDevice(target),
            ));
        }
        self.session.ensure_quiescent()?;
        Ok(PcuOwnedBinding::new(
            target,
            self.device,
            resource.size_bytes(),
            access,
            binding_type,
            MetalOwnedResource {
                buffer,
                access,
                binding_type,
            },
        ))
    }
}

#[path = "device/device.rs"]
mod device;
pub use device::MetalPreparedDeviceKernel;
