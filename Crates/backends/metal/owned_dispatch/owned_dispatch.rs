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
    MetalPreparedFloatKernel,
    MetalPreparedFloatBinaryKernel,
    MetalPreparedIntegerKernel,
    MetalPreparedCarrierKernel,
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
/// Cloning retains the exact session and identity; it never opens or rebinds a device. Clones
/// share completion/quarantine and resource affinity, without a Send/Sync promise.
#[derive(Clone)]
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
    /// Proves both the retained device identity and the exact native session are shared.
    /// Independently opened sessions reject even when they target the same physical device.
    #[must_use]
    pub fn shares_session(&self, other: &Self) -> bool {
        self.device == other.device && self.session.same_session(&other.session)
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
    DivRem(crate::MetalPreparedDivRemHostKernel),
    DivRemRoles(crate::MetalPreparedDivRemRoleHostKernel),
    Carrier(MetalPreparedCarrierKernel),
    Integer(MetalPreparedIntegerKernel),
    Float(MetalPreparedFloatKernel),
    FloatBinary(MetalPreparedFloatBinaryKernel),
    Transport(crate::MetalPreparedTransportKernel),
    Composed(RefCell<crate::MetalPreparedCheckedMapKernel>),
}
impl Program {
    const fn inputs(&self) -> Option<[PcuBindingRef; 2]> {
        match self {
            Self::DivRem(_) | Self::DivRemRoles(_) | Self::Transport(_) | Self::Composed(_) => None,
            Self::Carrier(kernel) => Some([kernel.input_binding(); 2]),
            Self::Integer(kernel) => Some(kernel.actual_input_pair()),
            Self::Float(kernel) => Some([kernel.input_binding(); 2]),
            Self::FloatBinary(kernel) => Some(kernel.actual_input_pair()),
        }
    }
    const fn output(&self) -> Option<PcuBindingRef> {
        match self {
            Self::DivRem(_) | Self::DivRemRoles(_) | Self::Transport(_) | Self::Composed(_) => None,
            Self::Carrier(kernel) => Some(kernel.output_binding()),
            Self::Integer(kernel) => Some(kernel.output_binding()),
            Self::Float(kernel) => Some(kernel.output_binding()),
            Self::FloatBinary(kernel) => Some(kernel.output_binding()),
        }
    }
    fn execute_into(
        &self,
        inputs: [&MetalBuffer; 2],
        output: &MetalBuffer,
    ) -> Result<(), MetalError> {
        match self {
            Self::DivRem(_) | Self::DivRemRoles(_) | Self::Transport(_) | Self::Composed(_) => {
                Err(MetalError::Unsupported)
            }
            Self::Carrier(kernel) => kernel.execute_into(inputs[0], output),
            Self::Integer(kernel) => kernel.execute_into(inputs, output),
            Self::Float(kernel) => kernel.execute_into(inputs[0], output),
            Self::FloatBinary(kernel) => kernel.execute_reads_into(inputs, output),
        }
    }
    fn execute(
        &self,
        inputs: [&MetalBuffer; 2],
    ) -> Result<(MetalBuffer, Option<crate::MetalFault>), MetalError> {
        match self {
            Self::DivRem(_) | Self::DivRemRoles(_) | Self::Transport(_) | Self::Composed(_) => {
                Err(MetalError::Unsupported)
            }
            Self::Carrier(kernel) => kernel.execute(inputs[0]).map(|output| (output, None)),
            Self::Integer(kernel) => kernel.execute_completed(inputs),
            Self::Float(kernel) => kernel.execute_completed(inputs[0]),
            Self::FloatBinary(kernel) => kernel.execute_completed(inputs),
        }
    }
    fn is_unread_declaration(&self, target: PcuBindingRef) -> bool {
        match self {
            Self::DivRemRoles(kernel) => kernel.is_unread_declaration(target),
            Self::Float(kernel) => kernel.is_unread_declaration(target),
            Self::FloatBinary(kernel) => kernel.is_unread_declaration(target),
            Self::Integer(kernel) => kernel.is_unread_declaration(target),
            Self::Composed(kernel) => {
                let kernel = kernel.borrow();
                kernel
                    .plan()
                    .declared_bindings()
                    .iter()
                    .any(|binding| binding.0 == target)
                    && !kernel
                        .plan()
                        .resources()
                        .iter()
                        .any(|resource| resource.binding == target)
            }
            Self::Transport(kernel) => {
                kernel
                    .plan()
                    .declared_bindings()
                    .iter()
                    .any(|binding| binding.0 == target)
                    && !kernel
                        .plan()
                        .resources()
                        .iter()
                        .any(|resource| resource.binding == target)
            }
            _ => false,
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
            if binding.byte_len != buffer.byte_len() as u64 {
                return Err(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::BufferTooSmall {
                        binding: binding.target,
                        required: binding.byte_len,
                        available: buffer.byte_len() as u64,
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
        if let Program::DivRem(kernel) = &self.program {
            return Self::execute_div_rem(kernel, bindings);
        }
        if let Program::DivRemRoles(kernel) = &self.program {
            return Self::execute_div_rem_roles(kernel, bindings);
        }
        if let Program::Composed(kernel) = &self.program {
            return Self::execute_composed(kernel, bindings);
        }
        if let Program::Transport(kernel) = &self.program {
            return Self::execute_transport(kernel, bindings);
        }
        let find = |target| {
            bindings
                .iter()
                .find(|binding| binding.target == target)
                .ok_or(MetalOwnedDispatchError::Binding(
                    PcuOwnedDispatchBindingError::Missing(target),
                ))
        };
        let inputs = self.program.inputs().ok_or(MetalError::Unsupported)?;
        let left = find(inputs[0])?.resource.buffer.borrow();
        let right = find(inputs[1])?.resource.buffer.borrow();
        // Inputs remain borrowed through terminal completion. The fresh output shields aliases.
        let result = self.program.execute([&left, &right]);
        drop(right);
        drop(left);
        let (output, recovered) = match result {
            Ok(completed) => completed,
            Err(MetalError::Arithmetic(fault)) => return Ok(PcuCompletionOutcome::Fault(fault)),
            Err(error) => return Err(error.into()),
        };
        let mut bytes = vec![0_u8; output.byte_len()];
        output.read_into_bytes(&mut bytes)?;
        find(self.program.output().ok_or(MetalError::Unsupported)?)?
            .resource
            .buffer
            .borrow_mut()
            .write_bytes(0, &bytes)?;
        Ok(recovered.map_or(PcuCompletionOutcome::Succeeded, PcuCompletionOutcome::Fault))
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
        let body = match submission.kernel.ops {
            [fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. }, _] => *body,
            ops => ops,
        };
        let program = if body.iter().any(|op|matches!(op,fusion_pcu::PcuDispatchOp::Data(fusion_pcu::PcuDispatchDataOp::CheckedDivRem{..}))) {
            match self.session.prepare_joint_host_kernel(submission.kernel).map_err(|error|match error {fusion_pcu::PcuHostDispatchError::Backend(error)=>MetalOwnedDispatchError::Metal(error),_=>MetalOwnedDispatchError::Metal(MetalError::Unsupported)})? {
                crate::MetalPreparedHostKernel::DivRem(kernel) => Program::DivRem(kernel),
                crate::MetalPreparedHostKernel::DivRemRoles(kernel) => Program::DivRemRoles(kernel),
                crate::MetalPreparedHostKernel::Single(_) | crate::MetalPreparedHostKernel::Transport(_) | crate::MetalPreparedHostKernel::Composed(_) => return Err(MetalError::Unsupported.into()),
            }
        } else if fusion_pcu::describe_checked_float_unary_map(submission.kernel).is_ok() {
            Program::Float(self.session.prepare_float_unary_kernel(submission.kernel)?)
        } else if crate::admission::carrier::is_carrier_kernel(submission.kernel) {
            Program::Carrier(self.session.prepare_carrier_kernel(submission.kernel)?)
        } else if let Ok(plan) = crate::MetalTransportPlan::assess_kernel(submission.kernel) {
            Program::Transport(self.session.prepare_transport_plan(plan)?)
        } else if let Some(plan) = crate::host_kernel::composed_plan(submission.kernel) {
            Program::Composed(RefCell::new(self.session.prepare_checked_map_plan(plan)?))
        } else if submission
            .kernel
            .bindings
            .iter()
            .all(|binding| matches!(binding.binding_type, PcuBindingType::Value(value) if matches!(value, PcuValueType::Scalar(fusion_pcu::PcuScalarType::I8 | fusion_pcu::PcuScalarType::U8 | fusion_pcu::PcuScalarType::I16 | fusion_pcu::PcuScalarType::U16 | fusion_pcu::PcuScalarType::I32 | fusion_pcu::PcuScalarType::U32 | fusion_pcu::PcuScalarType::I64 | fusion_pcu::PcuScalarType::U64 | fusion_pcu::PcuScalarType::I128 | fusion_pcu::PcuScalarType::U128 | fusion_pcu::PcuScalarType::I256 | fusion_pcu::PcuScalarType::U256 | fusion_pcu::PcuScalarType::I512 | fusion_pcu::PcuScalarType::U512))))
        {
            Program::Integer(self.session.prepare_integer_kernel(submission.kernel)?)
        } else {
            Program::FloatBinary(self.session.prepare_float_binary_kernel(submission.kernel)?)
        };
        let requirements = if let Program::Composed(kernel) = &program {
            composed::requirements(&kernel.borrow())
        } else if let Program::Transport(kernel) = &program {
            transport::requirements(kernel)
        } else {
            submission
                .kernel
                .bindings
                .iter()
                .filter(|binding| !program.is_unread_declaration(binding.reference()))
                .map(|binding| {
                    PcuOwnedBindingRequirement::from_verified_binding(
                        submission.kernel,
                        binding.reference(),
                        submission.shape,
                    )
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(MetalOwnedDispatchError::Binding)?
        };
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
        if !matches!(binding_type,PcuBindingType::Value(PcuValueType::Scalar(scalar)) if scalar.bit_width()>=8)
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

#[path = "div_rem/div_rem.rs"]
mod div_rem;

#[path = "transport/transport.rs"]
mod transport;

#[path = "composed/composed.rs"]
mod composed;
