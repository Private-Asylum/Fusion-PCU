//! Scoped host-buffer execution for the bounded scalar Dispatch profile.
//!
//! The primary asynchronous route owns device resources. This route borrows host slices and
//! returns only after the implementation has established that no device access remains.

use crate::{
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchSubmission,
    PcuError,
    PcuInvocationParameters,
    PcuScalar,
    PcuValueType,
    validate_dispatch_submission,
};

/// A scalar host slice whose access permission follows its Rust reference.
#[derive(Debug)]
pub enum PcuHostScalarSlice<'a, T: PcuScalar> {
    Read(&'a [T]),
    ReadWrite(&'a mut [T]),
}

impl<T: PcuScalar> PcuHostScalarSlice<'_, T> {
    #[must_use]
    pub const fn access(&self) -> PcuBindingAccess {
        match self {
            Self::Read(_) => PcuBindingAccess::ReadOnly,
            Self::ReadWrite(_) => PcuBindingAccess::ReadWrite,
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        match self {
            Self::Read(slice) => slice.len(),
            Self::ReadWrite(slice) => slice.len(),
        }
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One host slice assigned to a kernel binding.
#[derive(Debug)]
pub struct PcuHostScalarBinding<'a, T: PcuScalar> {
    pub target: PcuBindingRef,
    pub slice: PcuHostScalarSlice<'a, T>,
}

/// Admission or execution failure for scoped host-slice Dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuHostDispatchError<E> {
    Admission(PcuError),
    Duplicate(PcuBindingRef),
    Missing(PcuBindingRef),
    Unexpected(PcuBindingRef),
    AccessMismatch(PcuBindingRef),
    TypeMismatch(PcuBindingRef),
    UnsupportedLayout(PcuBindingRef),
    BufferTooSmall(PcuBindingRef),
    Backend(E),
}

/// Backend that can finish all host-buffer access before returning.
///
/// # Safety
///
/// Implementations must establish device quiescence before returning from `run_host_direct`,
/// including error paths and uncertain launches. They must never retain a host-slice pointer or
/// access the slices after return. A completion timeout alone does not establish quiescence.
/// They must also reject unsupported operations and scalar types before executing them; common
/// admission checks only the bounded resource layout and reference-derived permissions.
pub unsafe trait PcuSynchronousHostDispatchBackend<T: PcuScalar> {
    type Error;

    /// Runs an admitted scalar kernel to completion against borrowed host slices.
    ///
    /// # Errors
    ///
    /// Returns a backend error only after host-buffer access has ended.
    fn run_host_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, T>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), Self::Error>;

    /// Validates the bounded scalar profile, then runs it synchronously.
    ///
    /// # Errors
    ///
    /// Returns structural, binding, or backend failure.
    fn run_host(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuHostScalarBinding<'_, T>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<(), PcuHostDispatchError<Self::Error>> {
        validate_host_scalar_bindings::<T, Self::Error>(submission, bindings)?;
        if !parameters.bindings.is_empty() {
            return Err(PcuHostDispatchError::Admission(PcuError::unsupported()));
        }
        self.run_host_direct(submission, bindings, parameters)
            .map_err(PcuHostDispatchError::Backend)
    }
}

/// Checks reference-derived permissions, scalar type, coverage and buffer extent.
///
/// # Errors
///
/// Returns the first structural or binding error.
pub fn validate_host_scalar_bindings<T: PcuScalar, E>(
    submission: PcuDispatchSubmission<'_>,
    bindings: &[PcuHostScalarBinding<'_, T>],
) -> Result<(), PcuHostDispatchError<E>> {
    validate_dispatch_submission(submission).map_err(PcuHostDispatchError::Admission)?;
    if !submission.kernel.ports.is_empty() || !submission.kernel.parameters.is_empty() {
        return Err(PcuHostDispatchError::Admission(PcuError::unsupported()));
    }
    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index]
            .iter()
            .any(|previous| previous.target == binding.target)
        {
            return Err(PcuHostDispatchError::Duplicate(binding.target));
        }
        let Some(declared) = submission
            .kernel
            .bindings
            .iter()
            .find(|declared| declared.reference() == binding.target)
        else {
            return Err(PcuHostDispatchError::Unexpected(binding.target));
        };
        if declared.binding_type != PcuBindingType::Value(PcuValueType::Scalar(T::TYPE)) {
            return Err(PcuHostDispatchError::TypeMismatch(binding.target));
        }
        if declared.storage != PcuBindingStorageClass::Storage || declared.builtin.is_some() {
            return Err(PcuHostDispatchError::UnsupportedLayout(binding.target));
        }
        if !matches!(
            (binding.slice.access(), declared.access),
            (PcuBindingAccess::ReadOnly, PcuBindingAccess::ReadOnly)
                | (PcuBindingAccess::ReadWrite, _)
        ) {
            return Err(PcuHostDispatchError::AccessMismatch(binding.target));
        }
        let required = submission
            .kernel
            .minimum_binding_elements_for(binding.target, submission.shape.invocation_count().get())
            as usize;
        if binding.slice.len() < required {
            return Err(PcuHostDispatchError::BufferTooSmall(binding.target));
        }
    }
    for declared in submission.kernel.bindings {
        let target = declared.reference();
        if !bindings.iter().any(|binding| binding.target == target) {
            return Err(PcuHostDispatchError::Missing(target));
        }
    }
    Ok(())
}
