//! Synchronous typed host-call adapter over `ROCm`'s owned Dispatch path.

#[rustfmt::skip]
use core::{
    fmt,
    num::NonZeroU32,
};
use std::error::Error;
use crate::owned_dispatch::FaultWordState;

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceArgument,
    PcuDispatchKernelIr,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuInvocationShape,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryProviderOperation,
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
    ResidentArgumentOverlap(PcuBindingRef),
    ResidentResourceMismatch(PcuBindingRef),
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
            Self::ResidentArgumentOverlap(binding) => write!(
                f,
                "ROCm resident binding {binding:?} overlaps another argument that may write"
            ),
            Self::ResidentResourceMismatch(binding) => write!(
                f,
                "ROCm resident binding {binding:?} does not belong to the prepared runtime and pool"
            ),
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

struct HostBindingSlot {
    resource: Option<RocmMemoryResource>,
    fully_written_prefix: Option<usize>,
}

/// A single heterogeneous host or resident argument for the synchronous mixed-call adapter.
///
/// This is a low-level internal carrier for generated facade calls, not a second public function
/// entry point. Host bytes are staged; resident storage is validated and bound directly.
#[doc(hidden)]
pub enum RocmMixedHostArgument<'a> {
    Host(PcuHostArgument<'a>),
    Resident(PcuDeviceArgument<'a, RocmMemoryResource>),
}

trait HostKernelCallArgument {
    fn target(&self) -> PcuBindingRef;
    fn scalar(&self) -> PcuScalarType;
    fn access(&self) -> PcuBindingAccess;
    fn host_bytes(&self) -> Option<&[u8]>;
    fn host_bytes_mut(&mut self) -> Option<&mut [u8]>;
    fn resident_resource(&self) -> Option<&RocmMemoryResource>;
    fn resident_elements(&self) -> Option<usize>;
}

impl HostKernelCallArgument for PcuHostArgument<'_> {
    fn target(&self) -> PcuBindingRef {
        PcuHostArgument::target(self)
    }

    fn scalar(&self) -> PcuScalarType {
        PcuHostArgument::scalar(self)
    }

    fn access(&self) -> PcuBindingAccess {
        PcuHostArgument::access(self)
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        Some(PcuHostArgument::bytes(self))
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        PcuHostArgument::bytes_mut(self)
    }

    fn resident_resource(&self) -> Option<&RocmMemoryResource> {
        None
    }

    fn resident_elements(&self) -> Option<usize> {
        None
    }
}

impl HostKernelCallArgument for RocmMixedHostArgument<'_> {
    fn target(&self) -> PcuBindingRef {
        match self {
            Self::Host(argument) => argument.target(),
            Self::Resident(argument) => argument.target(),
        }
    }

    fn scalar(&self) -> PcuScalarType {
        match self {
            Self::Host(argument) => argument.scalar(),
            Self::Resident(argument) => argument.scalar(),
        }
    }

    fn access(&self) -> PcuBindingAccess {
        match self {
            Self::Host(argument) => argument.access(),
            Self::Resident(argument) => argument.access(),
        }
    }

    fn host_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Host(argument) => Some(argument.bytes()),
            Self::Resident(_) => None,
        }
    }

    fn host_bytes_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Self::Host(argument) => argument.bytes_mut(),
            Self::Resident(_) => None,
        }
    }

    fn resident_resource(&self) -> Option<&RocmMemoryResource> {
        match self {
            Self::Host(_) => None,
            Self::Resident(argument) => Some(argument.resource()),
        }
    }

    fn resident_elements(&self) -> Option<usize> {
        match self {
            Self::Host(_) => None,
            Self::Resident(argument) => Some(argument.elements()),
        }
    }
}

/// Reusable `ROCm` executable and owned staging resources for typed host calls.
pub struct RocmPreparedHostKernel {
    dispatch: RocmPreparedDispatch,
    memory: RocmMemoryProvider,
    pool: PcuMemoryPoolId,
    fault_word: Option<DeviceBuffer>,
    fault_word_state: FaultWordState,
    slots: Vec<HostBindingSlot>,
    bindings: Vec<PcuOwnedBinding<DeviceBuffer>>,
    poisoned: bool,
    last_call_completion_uncertain: bool,
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
        let binding_count = dispatch.binding_schema().len();
        // This follows backend SSA/type/geometry admission. Each staging slot has independent
        // backing, so a different binding cannot alias a proven output behind the analysis.
        let slots = dispatch
            .binding_schema()
            .iter()
            .map(|requirement| {
                let fully_written_prefix = kernel
                    .fully_written_binding_elements(requirement.target, invocations.get())
                    .map(|_| {
                        usize::try_from(requirement.min_required_bytes).map_err(|_| {
                            RocmHostKernelError::BufferSizeOverflow(requirement.target)
                        })
                    })
                    .transpose()?;
                Ok(HostBindingSlot {
                    resource: None,
                    fully_written_prefix,
                })
            })
            .collect::<Result<Vec<_>, RocmHostKernelError>>()?;
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
            fault_word_state: FaultWordState::NeedsReset,
            slots,
            bindings: Vec::with_capacity(binding_count),
            poisoned: false,
            last_call_completion_uncertain: false,
        })
    }
}

impl PcuPreparedHostKernel for RocmPreparedHostKernel {
    type Error = RocmHostKernelError;

    fn call(&mut self, arguments: &mut [PcuHostArgument<'_>]) -> Result<(), Self::Error> {
        self.call_arguments(arguments)
    }
}

impl RocmPreparedHostKernel {
    /// Executes a call with a mix of host-staged and directly resident arguments.
    #[doc(hidden)]
    pub fn call_mixed(
        &mut self,
        arguments: &mut [RocmMixedHostArgument<'_>],
    ) -> Result<(), RocmHostKernelError> {
        self.call_arguments(arguments)
    }

    /// Reports whether the most recent call may still be executing or failed to establish
    /// quiescence. A poisoned executable rejection is a certain prelaunch failure.
    #[doc(hidden)]
    #[must_use]
    pub const fn last_call_completion_uncertain(&self) -> bool {
        self.last_call_completion_uncertain
    }

    #[allow(clippy::too_many_lines)]
    fn call_arguments<A: HostKernelCallArgument>(
        &mut self,
        arguments: &mut [A],
    ) -> Result<(), RocmHostKernelError> {
        self.last_call_completion_uncertain = false;
        if self.poisoned {
            return Err(RocmHostKernelError::PoisonedAfterUncertainCompletion);
        }
        // A non-poisoned executable must not retain borrows/resources from an earlier call. The
        // uncertain path is poisoned above and deliberately keeps its submission leases alive.
        self.bindings.clear();
        if cfg!(target_endian = "big") {
            return Err(RocmHostKernelError::BigEndianHostUnsupported);
        }

        let requirements = self.dispatch.binding_schema();
        validate_call_arguments(requirements, arguments, Some(&self.memory))?;

        // Grow every host slot before any transfer. Resident resources are borrowed directly and
        // do not consume a staging allocation.
        let mut replacements = Vec::new();
        for (slot, requirement) in requirements.iter().enumerate() {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == requirement.target)
                .expect("coverage validated above");
            let Some(bytes) = argument.host_bytes() else {
                continue;
            };
            let required_size = u64::try_from(bytes.len())
                .map_err(|_| RocmHostKernelError::BufferSizeOverflow(requirement.target))?;
            let needs_growth = self.slots[slot]
                .resource
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
            self.slots[slot].resource = Some(resource);
        }

        // Upload only host arguments. A complete writer may elide the incoming copy exactly as
        // the host-only adapter did; resident buffers are already the device-side value. Finish
        // uploads before cloning resident handles into persistent submission storage, so any
        // transfer failure returns without retaining caller-owned resident allocations.
        for (slot, requirement) in requirements.iter().enumerate() {
            let Some(argument) = arguments
                .iter()
                .find(|argument| argument.target() == requirement.target)
            else {
                unreachable!("coverage validated above")
            };
            let Some(bytes) = argument.host_bytes() else {
                continue;
            };
            if self.slots[slot].fully_written_prefix.is_some() {
                continue;
            }
            let transfer = self.memory.transfer_to(
                self.slots[slot]
                    .resource
                    .as_mut()
                    .expect("host slot allocated above"),
                0,
                bytes,
            );
            if let Err(error) = transfer {
                // If the synchronous copy could not prove quiescence, dropping this owner leaves
                // the existing allocation lease quarantined. A later call allocates a fresh
                // staging slot instead of repeatedly hitting that quarantined gate.
                self.slots[slot].resource = None;
                return Err(RocmHostKernelError::Memory(error));
            }
        }

        for (slot, requirement) in requirements.iter().enumerate() {
            let argument = arguments
                .iter()
                .find(|argument| argument.target() == requirement.target)
                .expect("coverage validated above");
            let (buffer, byte_len) = if let Some(resource) = argument.resident_resource() {
                (resource.device_buffer().clone(), resource.size_bytes())
            } else {
                let resource = self.slots[slot]
                    .resource
                    .as_ref()
                    .expect("host slot allocated above");
                (resource.device_buffer().clone(), resource.size_bytes())
            };
            self.bindings.push(PcuOwnedBinding::new(
                requirement.target,
                self.dispatch.device_identity(),
                byte_len,
                argument.access(),
                requirement.binding_type,
                buffer,
            ));
        }

        let submission = if let Some(fault_word) = self.fault_word.as_mut() {
            let reset_fault_word = self.fault_word_state.begin_submission();
            self.dispatch
                .submit_with_fault_word_state(&self.bindings, fault_word, reset_fault_word)
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
                self.last_call_completion_uncertain = true;
                return Err(error.into());
            }
        };
        let outcome = match completion.wait() {
            Ok(outcome) => outcome,
            Err(error) => {
                self.poisoned = true;
                self.last_call_completion_uncertain = true;
                return Err(RocmOwnedDispatchError::Hip(error).into());
            }
        };
        self.bindings.clear();
        if self.fault_word.is_some() {
            self.fault_word_state = FaultWordState::after_terminal(outcome);
        }
        let recovered_fault = match outcome {
            PcuCompletionOutcome::Succeeded => None,
            PcuCompletionOutcome::Fault(fault) if fault.recovered => Some(fault),
            PcuCompletionOutcome::Fault(fault) => {
                return Err(RocmHostKernelError::CheckedExecutionFault(fault));
            }
            PcuCompletionOutcome::Failed => {
                return Err(RocmHostKernelError::CheckedExecutionFailed);
            }
        };

        for (slot, requirement) in requirements.iter().enumerate() {
            let Some(argument) = arguments
                .iter_mut()
                .find(|argument| argument.target() == requirement.target)
            else {
                unreachable!("coverage validated above")
            };
            let Some(bytes) = argument.host_bytes_mut() else {
                continue;
            };
            let bytes = if let Some(prefix) = self.slots[slot].fully_written_prefix {
                &mut bytes[..prefix]
            } else {
                bytes
            };
            let transfer = self.memory.transfer_from(
                self.slots[slot]
                    .resource
                    .as_ref()
                    .expect("host slot allocated above"),
                0,
                bytes,
            );
            if let Err(error) = transfer {
                self.slots[slot].resource = None;
                return Err(RocmHostKernelError::Memory(error));
            }
        }
        if let Some(fault) = recovered_fault {
            return Err(RocmHostKernelError::CheckedExecutionFault(fault));
        }
        Ok(())
    }
}

#[cfg(test)]
fn validate_host_arguments(
    requirements: &[PcuOwnedBindingRequirement],
    arguments: &[PcuHostArgument<'_>],
) -> Result<(), RocmHostKernelError> {
    validate_call_arguments(requirements, arguments, None)
}

fn validate_call_arguments<A: HostKernelCallArgument>(
    requirements: &[PcuOwnedBindingRequirement],
    arguments: &[A],
    memory: Option<&RocmMemoryProvider>,
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
        if !access_satisfies(argument.access(), requirement.access) {
            return Err(RocmHostKernelError::AccessMismatch(target));
        }
        let actual_bytes = if let Some(bytes) = argument.host_bytes() {
            bytes.len()
        } else if let (Some(resource), Some(elements)) =
            (argument.resident_resource(), argument.resident_elements())
        {
            if !resource_access_satisfies(resource.access(), requirement.access) {
                return Err(RocmHostKernelError::AccessMismatch(target));
            }
            memory
                .ok_or(RocmHostKernelError::ResidentResourceMismatch(target))?
                .validate_resource(resource, PcuMemoryProviderOperation::CopyResource)
                .map_err(RocmHostKernelError::Memory)?;
            resource
                .device_buffer()
                .validate_access_available()
                .map_err(|error| {
                    RocmHostKernelError::Dispatch(RocmOwnedDispatchError::Hip(error))
                })?;
            let element_bytes = u64::from(expected.bit_width().div_ceil(8));
            let required = u64::try_from(elements)
                .ok()
                .and_then(|count| count.checked_mul(element_bytes))
                .ok_or(RocmHostKernelError::BufferSizeOverflow(target))?;
            if resource.size_bytes() < required {
                return Err(RocmHostKernelError::BufferTooSmall {
                    binding: target,
                    required,
                    actual: usize::try_from(resource.size_bytes()).unwrap_or(usize::MAX),
                });
            }
            usize::try_from(required).unwrap_or(usize::MAX)
        } else {
            return Err(RocmHostKernelError::ResidentResourceMismatch(target));
        };
        if u64::try_from(actual_bytes)
            .map_or(true, |actual| actual < requirement.min_required_bytes)
        {
            return Err(RocmHostKernelError::BufferTooSmall {
                binding: target,
                required: requirement.min_required_bytes,
                actual: actual_bytes,
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
    for (index, argument) in arguments.iter().enumerate() {
        let Some(resource) = argument.resident_resource() else {
            continue;
        };
        for other in &arguments[index + 1..] {
            let Some(other_resource) = other.resident_resource() else {
                continue;
            };
            if (argument.access() != PcuBindingAccess::ReadOnly
                || other.access() != PcuBindingAccess::ReadOnly)
                && resource.may_overlap(other_resource)
            {
                return Err(RocmHostKernelError::ResidentArgumentOverlap(other.target()));
            }
        }
    }
    Ok(())
}

const fn access_satisfies(actual: PcuBindingAccess, required: PcuBindingAccess) -> bool {
    matches!(
        (actual, required),
        (PcuBindingAccess::ReadWrite, _)
            | (PcuBindingAccess::ReadOnly, PcuBindingAccess::ReadOnly)
            | (PcuBindingAccess::WriteOnly, PcuBindingAccess::WriteOnly)
    )
}

const fn resource_access_satisfies(actual: PcuMemoryAccess, required: PcuBindingAccess) -> bool {
    matches!(
        (actual, required),
        (PcuMemoryAccess::ReadWrite, _)
            | (PcuMemoryAccess::ReadOnly, PcuBindingAccess::ReadOnly)
            | (PcuMemoryAccess::WriteOnly, PcuBindingAccess::WriteOnly)
    )
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
        access_satisfies,
        validate_host_arguments,
        validate_call_arguments,
        RocmMixedHostArgument,
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
    fn mixed_host_variant_preserves_the_host_adapter_contract() {
        let input = PcuBindingRef::new(0, 0);
        let source = [3_u32, 5, 8];
        let argument = [RocmMixedHostArgument::Host(PcuHostArgument::read(
            input, &source,
        ))];
        let requirements = [requirement(
            input,
            PcuScalarType::U32,
            PcuBindingAccess::ReadOnly,
            12,
        )];
        assert!(validate_call_arguments(&requirements, &argument, None).is_ok());
    }

    #[test]
    fn binding_access_requires_exclusive_rw_for_readwrite_kernels() {
        assert!(access_satisfies(
            PcuBindingAccess::ReadWrite,
            PcuBindingAccess::ReadOnly
        ));
        assert!(!access_satisfies(
            PcuBindingAccess::ReadOnly,
            PcuBindingAccess::ReadWrite
        ));
        assert!(!access_satisfies(
            PcuBindingAccess::WriteOnly,
            PcuBindingAccess::ReadWrite
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
