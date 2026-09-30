//! Synchronous typed device-resident kernel execution and transfer helpers.

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
    PcuDeviceBuffer,
    PcuDeviceBufferAllocationError,
    PcuDeviceBufferAllocator,
    PcuDeviceKernelBackend,
    PcuDispatchKernelIr,
    PcuMemoryAccess,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryPoolId,
    PcuMemoryProvider,
    PcuMemoryResource,
    PcuHostArgument,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuPreparedDeviceKernel,
    PcuPreparedOwnedDispatch,
    PcuScalar,
    PcuScalarType,
    PcuValueType,
};

#[rustfmt::skip]
use crate::{
    CudaMemoryResource,
    CudaOwnedDispatchBackend,
    CudaOwnedDispatchError,
    CudaPreparedDispatch,
};

/// Typed failure from preparing or calling a resident `CUDA` kernel.
#[derive(Debug)]
pub enum CudaDeviceKernelError {
    Dispatch(CudaOwnedDispatchError),
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
    UnsupportedBindingType(PcuBindingRef),
    ScalarMismatch {
        binding: PcuBindingRef,
        expected: PcuScalarType,
        actual: PcuScalarType,
    },
    AccessMismatch(PcuBindingRef),
    BufferTooSmall {
        binding: PcuBindingRef,
        required: u64,
        actual: u64,
    },
    LengthMismatch {
        expected: usize,
        actual: usize,
    },
    ZeroSizedBuffer,
    SizeOverflow,
    CheckedExecutionFault(fusion_pcu::PcuExecutionFault),
    CheckedExecutionFailed,
}

impl fmt::Display for CudaDeviceKernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dispatch(error) => error.fmt(f),
            Self::Memory(error) => write!(f, "CUDA resident-buffer transfer failed: {error:?}"),
            Self::InvalidInvocationShape(shape) => write!(
                f,
                "CUDA resident calls require a nonzero one-dimensional shape, got {shape:?}"
            ),
            Self::BigEndianHostUnsupported => {
                f.write_str("CUDA typed buffer transfers currently require a little-endian host")
            }
            Self::PoisonedAfterUncertainCompletion => f.write_str(
                "CUDA resident executable cannot be reused after an uncertain completion; prepare it again",
            ),
            Self::ArgumentCount { expected, actual } => write!(
                f,
                "CUDA resident call expected {expected} resource arguments, got {actual}"
            ),
            Self::DuplicateArgument(binding) => write!(f, "duplicate binding {binding:?}"),
            Self::MissingArgument(binding) => write!(f, "missing binding {binding:?}"),
            Self::UnexpectedArgument(binding) => write!(f, "unexpected binding {binding:?}"),
            Self::UnsupportedBindingType(binding) => {
                write!(f, "unsupported non-scalar binding {binding:?}")
            }
            Self::ScalarMismatch { binding, expected, actual } => write!(
                f,
                "binding {binding:?} has scalar {actual:?}, expected {expected:?}"
            ),
            Self::AccessMismatch(binding) => write!(f, "incompatible access for binding {binding:?}"),
            Self::BufferTooSmall { binding, required, actual } => write!(
                f,
                "binding {binding:?} has {actual} device bytes, kernel requires {required}"
            ),
            Self::LengthMismatch { expected, actual } => {
                write!(f, "typed buffer length is {actual}, expected {expected}")
            }
            Self::ZeroSizedBuffer => f.write_str("CUDA provider does not allocate empty buffers"),
            Self::SizeOverflow => f.write_str("typed buffer byte size overflows the platform size"),
            Self::CheckedExecutionFault(fault) => write!(
                f,
                "CUDA resident kernel reported {:?} at invocation {}",
                fault.kind, fault.invocation_id
            ),
            Self::CheckedExecutionFailed => f.write_str("CUDA resident kernel completed with failure"),
        }
    }
}

impl Error for CudaDeviceKernelError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Dispatch(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CudaOwnedDispatchError> for CudaDeviceKernelError {
    fn from(value: CudaOwnedDispatchError) -> Self {
        Self::Dispatch(value)
    }
}

/// Compiled device-resident executable. No host bytes or staging buffers are retained here.
pub struct CudaPreparedDeviceKernel {
    dispatch: CudaPreparedDispatch,
    fault_word: Option<crate::DeviceBuffer>,
    fault_word_state: FaultWordState,
    bindings: Vec<PcuOwnedBinding<crate::DeviceBuffer>>,
    poisoned: bool,
}

impl PcuDeviceKernelBackend for CudaOwnedDispatchBackend {
    type Resource = CudaMemoryResource;
    type Prepared = CudaPreparedDeviceKernel;
    type Error = CudaDeviceKernelError;

    fn prepare_device_kernel(
        &self,
        kernel: &PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        let logical_shape = kernel.entry.logical_shape;
        let Some(invocations) = NonZeroU32::new(logical_shape[0]) else {
            return Err(CudaDeviceKernelError::InvalidInvocationShape(logical_shape));
        };
        if logical_shape[1] != 1 || logical_shape[2] != 1 {
            return Err(CudaDeviceKernelError::InvalidInvocationShape(logical_shape));
        }
        let shape = fusion_pcu::PcuInvocationShape::invocations(invocations);
        let dispatch =
            self.prepare_dispatch(fusion_pcu::PcuDispatchSubmission { kernel, shape })?;
        let fault_word = if dispatch.requires_checked_fault_word() {
            Some(
                self.allocate(core::mem::size_of::<u64>())
                    .map_err(CudaOwnedDispatchError::Cuda)?,
            )
        } else {
            None
        };
        Ok(CudaPreparedDeviceKernel {
            dispatch,
            fault_word,
            fault_word_state: FaultWordState::NeedsReset,
            bindings: Vec::with_capacity(kernel.bindings.len()),
            poisoned: false,
        })
    }
}

impl PcuPreparedDeviceKernel for CudaPreparedDeviceKernel {
    type Resource = CudaMemoryResource;
    type Error = CudaDeviceKernelError;

    #[allow(clippy::too_many_lines)]
    fn call(
        &mut self,
        arguments: &mut [PcuDeviceArgument<'_, Self::Resource>],
    ) -> Result<(), Self::Error> {
        if self.poisoned {
            return Err(CudaDeviceKernelError::PoisonedAfterUncertainCompletion);
        }
        let requirements = self.dispatch.binding_schema();
        if arguments.len() != requirements.len() {
            return Err(CudaDeviceKernelError::ArgumentCount {
                expected: requirements.len(),
                actual: arguments.len(),
            });
        }
        for (index, argument) in arguments.iter().enumerate() {
            let target = argument.target();
            if arguments[..index]
                .iter()
                .any(|other| other.target() == target)
            {
                return Err(CudaDeviceKernelError::DuplicateArgument(target));
            }
            let Some(requirement) = requirements.iter().find(|item| item.target == target) else {
                return Err(CudaDeviceKernelError::UnexpectedArgument(target));
            };
            let PcuBindingType::Value(PcuValueType::Scalar(expected)) = requirement.binding_type
            else {
                return Err(CudaDeviceKernelError::UnsupportedBindingType(target));
            };
            if argument.scalar() != expected {
                return Err(CudaDeviceKernelError::ScalarMismatch {
                    binding: target,
                    expected,
                    actual: argument.scalar(),
                });
            }
            if !binding_access_supports(argument.access(), requirement.access) {
                return Err(CudaDeviceKernelError::AccessMismatch(target));
            }
            let resource_access = argument.resource().access();
            if !memory_access_supports(resource_access, requirement.access)
                || !memory_access_supports(resource_access, argument.access())
            {
                return Err(CudaDeviceKernelError::AccessMismatch(target));
            }
            let Some(element_bytes) = argument
                .elements()
                .checked_mul(scalar_width(argument.scalar()))
            else {
                return Err(CudaDeviceKernelError::SizeOverflow);
            };
            let required =
                u64::try_from(element_bytes).map_err(|_| CudaDeviceKernelError::SizeOverflow)?;
            if required < requirement.min_required_bytes
                || argument.resource().size_bytes() < required
            {
                return Err(CudaDeviceKernelError::BufferTooSmall {
                    binding: target,
                    required: requirement.min_required_bytes.max(required),
                    actual: argument.resource().size_bytes().min(required),
                });
            }
        }
        for requirement in requirements {
            if !arguments
                .iter()
                .any(|argument| argument.target() == requirement.target)
            {
                return Err(CudaDeviceKernelError::MissingArgument(requirement.target));
            }
        }

        self.bindings.clear();
        for requirement in requirements {
            let argument = arguments
                .iter()
                .find(|item| item.target() == requirement.target)
                .expect("coverage validated above");
            let resource = argument.resource();
            self.bindings.push(PcuOwnedBinding::new(
                requirement.target,
                self.dispatch.device_identity(),
                resource.size_bytes(),
                argument.access(),
                requirement.binding_type,
                resource.device_buffer().clone(),
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
                } else {
                    self.poisoned = true;
                }
                return Err(error.into());
            }
        };
        let outcome = match completion.wait() {
            Ok(outcome) => outcome,
            Err(error) => {
                self.poisoned = true;
                return Err(CudaOwnedDispatchError::Cuda(error).into());
            }
        };
        self.bindings.clear();
        if self.fault_word.is_some() {
            self.fault_word_state = FaultWordState::after_terminal(outcome);
        }
        match outcome {
            PcuCompletionOutcome::Succeeded => Ok(()),
            PcuCompletionOutcome::Fault(fault) => {
                Err(CudaDeviceKernelError::CheckedExecutionFault(fault))
            }
            PcuCompletionOutcome::Failed => Err(CudaDeviceKernelError::CheckedExecutionFailed),
        }
    }
}

impl CudaOwnedDispatchBackend {
    /// Allocates typed, initially unspecified device storage in this backend's selected runtime
    /// and the pool named by `request`.
    ///
    /// This uses the same provider allocation contract as uploads, but performs no host transfer
    /// or initialization. The caller must not read output elements until a complete writer has
    /// finished successfully.
    ///
    /// # Errors
    ///
    /// Returns a typed extent, request, provider, or allocation-resource contract error.
    pub fn allocate_device_buffer<T: PcuScalar>(
        &self,
        request: fusion_pcu::PcuMemoryAllocationRequest,
        elements: usize,
    ) -> Result<PcuDeviceBuffer<T, CudaMemoryResource>, PcuDeviceBufferAllocationError> {
        let mut provider = self.memory_provider(request.pool);
        provider.allocate_device_buffer::<T>(request, elements)
    }

    /// Uploads typed host values into reusable provider-owned `CUDA` storage.
    ///
    /// # Errors
    /// Returns a typed error when allocation or upload fails, when `values` is empty, or on a
    /// big-endian host where the native byte view would not match `CUDA`'s little-endian encoding.
    pub fn upload_buffer<T: PcuScalar>(
        &self,
        pool: PcuMemoryPoolId,
        values: &[T],
    ) -> Result<PcuDeviceBuffer<T, CudaMemoryResource>, CudaDeviceKernelError> {
        if cfg!(target_endian = "big") {
            return Err(CudaDeviceKernelError::BigEndianHostUnsupported);
        }
        if values.is_empty() {
            return Err(CudaDeviceKernelError::ZeroSizedBuffer);
        }
        let host = PcuHostArgument::read(PcuBindingRef::new(0, 0), values);
        let size = host.bytes().len();
        let mut memory = self.memory_provider(pool);
        let mut resource = memory
            .allocate(PcuMemoryAllocationRequest {
                pool,
                size_bytes: u64::try_from(size).map_err(|_| CudaDeviceKernelError::SizeOverflow)?,
                alignment_bytes: 16,
                access: PcuMemoryAccess::ReadWrite,
                host_access: PcuMemoryHostAccess::TransferOnly,
                require_device_local: false,
            })
            .map_err(CudaDeviceKernelError::Memory)?;
        memory
            .transfer_to(&mut resource, 0, host.bytes())
            .map_err(CudaDeviceKernelError::Memory)?;
        Ok(PcuDeviceBuffer::new(resource, values.len()))
    }

    /// Replaces all contents in an existing typed device buffer through the memory provider.
    ///
    /// # Errors
    /// Returns a typed error when the input length differs, the provider rejects the resource, or
    /// the host byte order cannot be represented by the `CUDA` transfer ABI.
    pub fn refresh_buffer<T: PcuScalar>(
        &self,
        pool: PcuMemoryPoolId,
        buffer: &mut PcuDeviceBuffer<T, CudaMemoryResource>,
        values: &[T],
    ) -> Result<(), CudaDeviceKernelError> {
        if cfg!(target_endian = "big") {
            return Err(CudaDeviceKernelError::BigEndianHostUnsupported);
        }
        if values.len() != buffer.len() {
            return Err(CudaDeviceKernelError::LengthMismatch {
                expected: buffer.len(),
                actual: values.len(),
            });
        }
        let host = PcuHostArgument::read(PcuBindingRef::new(0, 0), values);
        let mut memory = self.memory_provider(pool);
        memory
            .transfer_to(buffer.resource_mut(), 0, host.bytes())
            .map_err(CudaDeviceKernelError::Memory)
    }

    /// Downloads and decodes all elements from a typed device buffer.
    ///
    /// # Errors
    /// Returns a typed error when the output length differs, the provider rejects the resource, or
    /// the host byte order cannot be represented by the `CUDA` transfer ABI.
    ///
    /// # Panics
    /// The core's sealed typed mutable byte view is always available.
    pub fn download_buffer<T: PcuScalar>(
        &self,
        pool: PcuMemoryPoolId,
        buffer: &PcuDeviceBuffer<T, CudaMemoryResource>,
        values: &mut [T],
    ) -> Result<(), CudaDeviceKernelError> {
        if cfg!(target_endian = "big") {
            return Err(CudaDeviceKernelError::BigEndianHostUnsupported);
        }
        if values.len() != buffer.len() {
            return Err(CudaDeviceKernelError::LengthMismatch {
                expected: buffer.len(),
                actual: values.len(),
            });
        }
        let mut host = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), values);
        let mut memory = self.memory_provider(pool);
        memory
            .transfer_from(
                buffer.resource(),
                0,
                host.bytes_mut().expect("read-write typed host view"),
            )
            .map_err(CudaDeviceKernelError::Memory)
    }
}

const fn is_certain_prelaunch_error(error: &CudaOwnedDispatchError) -> bool {
    matches!(
        error,
        CudaOwnedDispatchError::Binding(_)
            | CudaOwnedDispatchError::BufferSizeMismatch { .. }
            | CudaOwnedDispatchError::BufferTooSmall { .. }
            | CudaOwnedDispatchError::DifferentRuntime(_)
            | CudaOwnedDispatchError::MemoryAccessMismatch(_)
    )
}

const fn scalar_width(scalar: PcuScalarType) -> usize {
    match scalar {
        PcuScalarType::I8 | PcuScalarType::U8 => 1,
        PcuScalarType::I16 | PcuScalarType::U16 | PcuScalarType::F16 | PcuScalarType::BF16 => 2,
        PcuScalarType::I32 | PcuScalarType::U32 | PcuScalarType::F32 => 4,
        PcuScalarType::I64 | PcuScalarType::U64 | PcuScalarType::F64 => 8,
        PcuScalarType::Bool | PcuScalarType::I4 | PcuScalarType::U4 => 0,
    }
}

const fn memory_access_supports(memory: PcuMemoryAccess, required: PcuBindingAccess) -> bool {
    matches!(
        (memory, required),
        (PcuMemoryAccess::ReadWrite, _)
            | (PcuMemoryAccess::ReadOnly, PcuBindingAccess::ReadOnly)
            | (PcuMemoryAccess::WriteOnly, PcuBindingAccess::WriteOnly)
    )
}

const fn binding_access_supports(provided: PcuBindingAccess, required: PcuBindingAccess) -> bool {
    matches!(
        (provided, required),
        (PcuBindingAccess::ReadWrite, _)
            | (PcuBindingAccess::ReadOnly, PcuBindingAccess::ReadOnly)
            | (PcuBindingAccess::WriteOnly, PcuBindingAccess::WriteOnly)
    )
}

#[cfg(test)]
mod tests {
    use super::binding_access_supports;
    use fusion_pcu::PcuBindingAccess::{ReadOnly, ReadWrite, WriteOnly};

    #[test]
    fn device_argument_access_must_cover_every_kernel_access() {
        let cases = [
            (ReadOnly, ReadOnly, true),
            (ReadOnly, WriteOnly, false),
            (ReadOnly, ReadWrite, false),
            (WriteOnly, ReadOnly, false),
            (WriteOnly, WriteOnly, true),
            (WriteOnly, ReadWrite, false),
            (ReadWrite, ReadOnly, true),
            (ReadWrite, WriteOnly, true),
            (ReadWrite, ReadWrite, true),
        ];
        for (provided, required, expected) in cases {
            assert_eq!(
                binding_access_supports(provided, required),
                expected,
                "{provided:?} cannot satisfy {required:?}"
            );
        }
    }
}
