//! Synchronous typed host-call adapter over `ROCm`'s owned Dispatch path.

#[rustfmt::skip]
use core::{
    fmt,
    num::NonZeroU32,
};
use std::error::Error;

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuInvocationShape,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuOwnedBindingRequirement,
    PcuOwnedDispatchBackend,
    PcuPreparedHostKernel,
    PcuPreparedOwnedDispatch,
    PcuScalarType,
    PcuValueType,
};

#[rustfmt::skip]
use crate::{
    DeviceBuffer,
    RocmMemoryProvider,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmOwnedDispatchError,
    RocmPreparedDispatch,
};

/// Failure from preparing or synchronously calling a typed `ROCm` host kernel.
#[derive(Debug)]
pub enum RocmHostKernelError {
    Dispatch(RocmOwnedDispatchError),
    Memory(fusion_pcu::PcuMemoryProviderError),
    InvalidInvocationShape([u32; 3]),
    BigEndianHostUnsupported,
    PoisonedAfterUncertainCompletion,
    ArgumentCount {
        expected: usize,
        actual: usize,
    },
    DuplicateArgument(PcuBindingRef),
    MissingArgument(PcuBindingRef),
    UnexpectedArgument(PcuBindingRef),
    ScalarMismatch {
        binding: PcuBindingRef,
        expected: PcuScalarType,
        actual: PcuScalarType,
    },
    UnsupportedBindingType(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    BufferTooSmall {
        binding: PcuBindingRef,
        required: u64,
        actual: usize,
    },
    CheckedExecutionFault(fusion_pcu::PcuExecutionFault),
    CheckedExecutionFailed,
    BufferSizeOverflow(PcuBindingRef),
}

impl fmt::Display for RocmHostKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dispatch(error) => error.fmt(f),
            Self::Memory(error) => write!(f, "ROCm host-call memory operation failed: {error:?}"),
            Self::InvalidInvocationShape(shape) => write!(
                f,
                "ROCm host calls require a nonzero one-dimensional invocation shape, got {shape:?}"
            ),
            Self::BigEndianHostUnsupported => {
                f.write_str("ROCm typed host calls currently require a little-endian host")
            }
            Self::PoisonedAfterUncertainCompletion => f.write_str(
                "ROCm host-call executable is unavailable after an uncertain completion; prepare it again",
            ),
            Self::ArgumentCount { expected, actual } => write!(
                f,
                "ROCm host call expected {expected} resource arguments, got {actual}"
            ),
            Self::DuplicateArgument(binding) => {
                write!(f, "ROCm host call contains duplicate binding {binding:?}")
            }
            Self::MissingArgument(binding) => {
                write!(f, "ROCm host call is missing binding {binding:?}")
            }
            Self::UnexpectedArgument(binding) => {
                write!(f, "ROCm host call contains unexpected binding {binding:?}")
            }
            Self::ScalarMismatch { binding, expected, actual } => write!(
                f,
                "ROCm host binding {binding:?} has scalar {actual:?}, expected {expected:?}"
            ),
            Self::UnsupportedBindingType(binding) => write!(
                f,
                "ROCm typed host calls do not support non-scalar binding {binding:?}"
            ),
            Self::AccessMismatch(binding) => {
                write!(f, "ROCm host binding {binding:?} does not allow the declared access")
            }
            Self::BufferTooSmall { binding, required, actual } => write!(
                f,
                "ROCm host binding {binding:?} has {actual} bytes, but the kernel requires {required}"
            ),
            Self::CheckedExecutionFault(fault) => write!(
                f,
                "ROCm host kernel reported {:?} at invocation {}",
                fault.kind, fault.invocation_id
            ),
            Self::CheckedExecutionFailed => f.write_str("ROCm host kernel completed with failure"),
            Self::BufferSizeOverflow(binding) => {
                write!(f, "ROCm host binding {binding:?} size does not fit in u64")
            }
        }
    }
}

impl Error for RocmHostKernelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dispatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<RocmOwnedDispatchError> for RocmHostKernelError {
    fn from(value: RocmOwnedDispatchError) -> Self {
        Self::Dispatch(value)
    }
}

/// Reusable `ROCm` executable and owned staging resources for typed host calls.
pub struct RocmPreparedHostKernel {
    dispatch: RocmPreparedDispatch,
    memory: RocmMemoryProvider,
    pool: PcuMemoryPoolId,
    fault_word: Option<DeviceBuffer>,
    resources: Vec<Option<RocmMemoryResource>>,
    bindings: Vec<PcuOwnedBinding<DeviceBuffer>>,
    poisoned: bool,
}

impl PcuHostKernelBackend for RocmOwnedDispatchBackend {
    type Prepared = RocmPreparedHostKernel;
    type Error = RocmHostKernelError;

    fn prepare_host_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let logical_shape = kernel.entry.logical_shape;
        let Some(invocations) = NonZeroU32::new(logical_shape[0]) else {
            return Err(RocmHostKernelError::InvalidInvocationShape(logical_shape));
        };
        if logical_shape[1] != 1 || logical_shape[2] != 1 {
            return Err(RocmHostKernelError::InvalidInvocationShape(logical_shape));
        }
        let shape = PcuInvocationShape::invocations(invocations);
        let dispatch =
            self.prepare_dispatch(fusion_pcu::PcuDispatchSubmission { kernel, shape })?;
        let slots = dispatch.binding_schema().len();
        // The pool identity is stable for this discovery's device domain: discovery assigns the
        // pool id from the selected device id. Keeping the provider scoped to this prepared value
        // avoids inventing a separate host-call allocation namespace.
        let pool = PcuMemoryPoolId(self.device_identity().device_id());
        let fault_word = if dispatch.requires_checked_fault_word() {
            Some(
                self.allocate(core::mem::size_of::<u64>())
                    .map_err(RocmOwnedDispatchError::Hip)?,
            )
        } else {
            None
        };
        Ok(RocmPreparedHostKernel {
            dispatch,
            memory: self.memory_provider(pool),
            pool,
            fault_word,
            resources: core::iter::repeat_with(|| None).take(slots).collect(),
            bindings: Vec::with_capacity(slots),
            poisoned: false,
        })
    }
}

impl PcuPreparedHostKernel for RocmPreparedHostKernel {
    type Error = RocmHostKernelError;

    #[allow(clippy::too_many_lines)]
    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        if self.poisoned {
            return Err(RocmHostKernelError::PoisonedAfterUncertainCompletion);
        }
        if cfg!(target_endian = "big") {
            return Err(RocmHostKernelError::BigEndianHostUnsupported);
        }

        let requirements = self.dispatch.binding_schema();
        // Validate the entire host surface before allocating, transferring, or launching.
        validate_host_arguments(requirements, arguments)?;

        // Grow all changed slots first. If any provider allocation fails, previous reusable
        // resources remain intact and no command has been submitted.
        let mut replacements: Vec<(usize, RocmMemoryResource)> = Vec::new();
        for (slot, requirement) in requirements.iter().enumerate() {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == requirement.target)
                .expect("coverage validated above");
            let size = argument.bytes().len();
            let required_size = u64::try_from(size)
                .map_err(|_| RocmHostKernelError::BufferSizeOverflow(requirement.target))?;
            let needs_growth = self.resources[slot]
                .as_ref()
                .is_none_or(|resource| resource.size_bytes() < required_size);
            if needs_growth {
                let resource = self
                    .memory
                    .allocate(PcuMemoryAllocationRequest {
                        pool: self.pool,
                        size_bytes: required_size,
                        alignment_bytes: 16,
                        access: PcuMemoryAccess::ReadWrite,
                        host_access: PcuMemoryHostAccess::TransferOnly,
                        require_device_local: false,
                    })
                    .map_err(RocmHostKernelError::Memory)?;
                replacements.push((slot, resource));
            }
        }
        for (slot, resource) in replacements {
            self.resources[slot] = Some(resource);
        }

        self.bindings.clear();
        for (slot, requirement) in requirements.iter().enumerate() {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == requirement.target)
                .expect("coverage validated above");
            let resource = self.resources[slot].as_ref().expect("slot allocated above");
            let byte_len = resource.size_bytes();
            self.bindings.push(PcuOwnedBinding::new(
                requirement.target,
                self.dispatch.device_identity(),
                byte_len,
                argument.access(),
                requirement.binding_type,
                resource.device_buffer().clone(),
            ));
        }

        // Transfer-only provider operations preserve input values (including mutable arguments)
        // without exposing a HIP pointer to any borrowed host memory.
        for (slot, requirement) in requirements.iter().enumerate() {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == requirement.target)
                .expect("coverage validated above");
            self.memory
                .transfer_to(
                    self.resources[slot].as_mut().expect("slot allocated above"),
                    0,
                    argument.bytes(),
                )
                .map_err(RocmHostKernelError::Memory)?;
        }

        let submission = if let Some(fault_word) = self.fault_word.as_mut() {
            self.dispatch
                .submit_with_fault_word(&self.bindings, fault_word)
        } else {
            self.dispatch.submit(&self.bindings)
        };
        let mut completion = match submission {
            Ok(completion) => completion,
            Err(error) => {
                if is_certain_prelaunch_error(&error) {
                    self.bindings.clear();
                    return Err(error.into());
                }
                self.poisoned = true;
                return Err(error.into());
            }
        };
        let outcome = match completion.wait() {
            Ok(outcome) => outcome,
            Err(error) => {
                self.poisoned = true;
                return Err(RocmOwnedDispatchError::Hip(error).into());
            }
        };
        // The wait returned a terminal quiescent outcome; the completion no longer needs these
        // binding leases and the prepared staging resources remain the sole owners.
        self.bindings.clear();
        match outcome {
            PcuCompletionOutcome::Succeeded => {}
            PcuCompletionOutcome::Fault(fault) => {
                // The checked completion is quiescent. Keep the executable retryable, but do not
                // expose outputs from a faulting invocation.
                return Err(RocmHostKernelError::CheckedExecutionFault(fault));
            }
            PcuCompletionOutcome::Failed => {
                return Err(RocmHostKernelError::CheckedExecutionFailed);
            }
        }

        // Only expose mutable result bytes after the completion token proves quiescence.
        for (slot, requirement) in requirements.iter().enumerate() {
            let Some(argument) = arguments
                .iter_mut()
                .find(|argument| argument.target() == requirement.target)
            else {
                unreachable!("coverage validated above")
            };
            let Some(bytes) = argument.bytes_mut() else {
                continue;
            };
            self.memory
                .transfer_from(
                    self.resources[slot].as_ref().expect("slot allocated above"),
                    0,
                    bytes,
                )
                .map_err(RocmHostKernelError::Memory)?;
        }
        Ok(())
    }
}

fn validate_host_arguments(
    requirements: &[PcuOwnedBindingRequirement],
    arguments: &[PcuHostArgument<'_>],
) -> Result<(), RocmHostKernelError> {
    for (index, argument) in arguments.iter().enumerate() {
        let target = argument.target();
        if arguments[..index]
            .iter()
            .any(|other| other.target() == target)
        {
            return Err(RocmHostKernelError::DuplicateArgument(target));
        }
        let Some(requirement) = requirements
            .iter()
            .find(|required| required.target == target)
        else {
            return Err(RocmHostKernelError::UnexpectedArgument(target));
        };
        let PcuBindingType::Value(PcuValueType::Scalar(expected)) = requirement.binding_type else {
            return Err(RocmHostKernelError::UnsupportedBindingType(target));
        };
        if argument.scalar() != expected {
            return Err(RocmHostKernelError::ScalarMismatch {
                binding: target,
                expected,
                actual: argument.scalar(),
            });
        }
        if argument.access() == PcuBindingAccess::ReadOnly
            && requirement.access != PcuBindingAccess::ReadOnly
        {
            return Err(RocmHostKernelError::AccessMismatch(target));
        }
        let actual = argument.bytes().len();
        if u64::try_from(actual).map_or(true, |actual| actual < requirement.min_required_bytes) {
            return Err(RocmHostKernelError::BufferTooSmall {
                binding: target,
                required: requirement.min_required_bytes,
                actual,
            });
        }
    }
    for required in requirements {
        if !arguments
            .iter()
            .any(|argument| argument.target() == required.target)
        {
            return Err(RocmHostKernelError::MissingArgument(required.target));
        }
    }
    Ok(())
}

const fn is_certain_prelaunch_error(error: &RocmOwnedDispatchError) -> bool {
    matches!(
        error,
        RocmOwnedDispatchError::Binding(_)
            | RocmOwnedDispatchError::BufferSizeMismatch { .. }
            | RocmOwnedDispatchError::BufferTooSmall { .. }
            | RocmOwnedDispatchError::DifferentRuntime(_)
            | RocmOwnedDispatchError::MemoryAccessMismatch(_)
    )
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use fusion_pcu::{
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingType,
        PcuHostArgument,
        PcuOwnedBindingRequirement,
        PcuScalarType,
        PcuValueType,
    };

    #[rustfmt::skip]
    use super::{
        validate_host_arguments,
        RocmHostKernelError,
    };

    fn requirement(
        target: PcuBindingRef,
        scalar: PcuScalarType,
        access: PcuBindingAccess,
        min_required_bytes: u64,
    ) -> PcuOwnedBindingRequirement {
        PcuOwnedBindingRequirement {
            target,
            access,
            binding_type: PcuBindingType::Value(PcuValueType::Scalar(scalar)),
            min_required_bytes,
        }
    }

    #[test]
    fn validates_coverage_scalar_access_and_minimum_extent_without_hardware() {
        let input = PcuBindingRef::new(0, 0);
        let output = PcuBindingRef::new(0, 1);
        let requirements = [
            requirement(input, PcuScalarType::U32, PcuBindingAccess::ReadOnly, 8),
            requirement(output, PcuScalarType::U32, PcuBindingAccess::WriteOnly, 8),
        ];
        let source = [1_u32, 2, 3];
        let mut destination = [0_u32; 4];
        let valid = [
            PcuHostArgument::read(input, &source),
            PcuHostArgument::read_write(output, &mut destination),
        ];
        assert!(validate_host_arguments(&requirements, &valid).is_ok());

        let missing = [PcuHostArgument::read(input, &source)];
        assert!(matches!(
            validate_host_arguments(&requirements, &missing),
            Err(RocmHostKernelError::MissingArgument(target)) if target == output
        ));

        let mut wrong_scalar_output = [0_u32; 2];
        let wrong_scalar = [
            PcuHostArgument::read(input, &[1_u64]),
            PcuHostArgument::read_write(output, &mut wrong_scalar_output),
        ];
        assert!(matches!(
            validate_host_arguments(&requirements, &wrong_scalar),
            Err(RocmHostKernelError::ScalarMismatch { binding, .. }) if binding == input
        ));

        let mut short_output = [0_u32; 2];
        let short = [
            PcuHostArgument::read(input, &[1_u32]),
            PcuHostArgument::read_write(output, &mut short_output),
        ];
        assert!(matches!(
            validate_host_arguments(&requirements, &short),
            Err(RocmHostKernelError::BufferTooSmall { binding, .. }) if binding == input
        ));

        let read_only_output = [
            PcuHostArgument::read(input, &source),
            PcuHostArgument::read(output, &source),
        ];
        assert!(matches!(
            validate_host_arguments(&requirements, &read_only_output),
            Err(RocmHostKernelError::AccessMismatch(target)) if target == output
        ));
    }

    #[test]
    fn rejects_duplicate_and_unexpected_resources_and_accepts_larger_slices() {
        let target = PcuBindingRef::new(2, 3);
        let requirements = [requirement(
            target,
            PcuScalarType::F64,
            PcuBindingAccess::ReadOnly,
            8,
        )];
        let values = [0.0_f64; 4];
        let valid = [PcuHostArgument::read(target, &values)];
        assert!(validate_host_arguments(&requirements, &valid).is_ok());

        let duplicate = [
            PcuHostArgument::read(target, &values),
            PcuHostArgument::read(target, &values),
        ];
        assert!(matches!(
            validate_host_arguments(&requirements, &duplicate),
            Err(RocmHostKernelError::DuplicateArgument(binding)) if binding == target
        ));

        let unexpected = [PcuHostArgument::read(PcuBindingRef::new(9, 9), &values)];
        assert!(matches!(
            validate_host_arguments(&requirements, &unexpected),
            Err(RocmHostKernelError::UnexpectedArgument(PcuBindingRef {
                set: 9,
                binding: 9
            }))
        ));
    }
}
