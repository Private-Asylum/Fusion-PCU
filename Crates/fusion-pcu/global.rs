//! Overridable hosted execution policy and cold preparation for direct PCU calls.
//!
//! Backend selection is runtime policy. The core IR and dialects never depend on this module.
//! Prepared state is thread-owned; no backend receives an invented `Send` or `Sync` promise.

use core::any::TypeId;
use core::fmt;
#[cfg(feature = "rocm")]
use core::sync::atomic::AtomicUsize;

#[cfg(feature = "rocm")]
#[path = "global/hosted.rs"]
mod hosted;

#[cfg(feature = "rocm")]
mod session;

// Sealed source carriers preserve ordinary borrows across automatic host/device staging.
// The public logical owner hides backend resources and retains initialized device results.
mod arguments;

mod tensor;
pub use arguments::PcuTensor;
#[doc(hidden)]
#[rustfmt::skip]
pub use arguments::{
    FixedArrayShape,
    FixedMatrixShape,
    PcuArgumentError,
    PcuCallArgument,
    PcuReadStorage,
    PcuSourceShape,
    PcuWriteStorage,
    ScalarShape,
    SliceShape,
};
#[doc(hidden)]
#[rustfmt::skip]
pub use tensor::{
    DynamicResidentShape,
    PcuTensorInput,
    PcuTensorCallInput,
    PcuTensorSource,
    PcuTensorGraphCapture,
    PcuTensorGraphOwner,
    PcuTensorGraphValue,
    call_consumed_owners_tensor_capture,
    call_consumed_pair_tensor_capture,
    call_consumed_tensor_capture,
    call_owned_tensor_capture,
    call_mixed_consumed_tensor_capture,
};

/// Which compiled execution provider may satisfy a direct call.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PcuBackendChoice {
    /// Select among compatible compiled providers. CPU is never an implicit fallback.
    #[default]
    Automatic,
    /// Require `ROCm`, failing if it is disabled or no compatible device can be opened.
    Rocm,
}

/// Runtime preferences for later calls; an already executing call retains its selected state.
#[derive(Clone, Copy, Debug)]
pub struct PcuExecutionPolicy {
    pub backend: PcuBackendChoice,
    /// Runtime device ordinal. An explicit request never substitutes another device.
    pub device: Option<u32>,
    /// Maximum cached specializations per host thread.
    pub cache_capacity: usize,
    /// Backend launch block size, checked by device preparation.
    pub block_size: u32,
    /// Cold candidate scoring after explicit device filtering; higher scores rank first.
    pub score_device: fn(&crate::PcuDeviceDescriptor<'_>, u64) -> i128,
}

impl Default for PcuExecutionPolicy {
    fn default() -> Self {
        Self {
            backend: PcuBackendChoice::Automatic,
            device: None,
            cache_capacity: 64,
            block_size: 256,
            score_device: default_device_score,
        }
    }
}

/// Default cold scoring prefers more physical memory; ties use stable runtime ordinals.
#[must_use]
pub fn default_device_score(_device: &crate::PcuDeviceDescriptor<'_>, total_memory: u64) -> i128 {
    i128::from(total_memory)
}

/// Failures at the hosted boundary, separate from generic IR validation errors.
#[derive(Debug)]
pub enum PcuExecutionError {
    NoBackendEnabled,
    ReentrantCall,
    ThreadUnavailable,
    InvalidPolicy,
    ResidentPolicyConflict,
    PolicyUnavailable,
    KernelBuild,
    Argument(PcuArgumentError),
    TensorExecutionUnavailable,
    EmptyTensorInput,
    InvalidTensorSourcePlan,
    TensorSourceRankMismatch {
        expected: usize,
        actual: usize,
    },
    TensorSourceShapeMismatch {
        expected: PcuSourceShape,
        actual: alloc::vec::Vec<usize>,
    },
    #[cfg(feature = "tensor")]
    TensorBuild(crate::dialect::tensor::TensorError),
    RecursiveTensorSource,
    TensorSourceNestingLimit,
    #[cfg(feature = "rocm")]
    Memory(crate::PcuMemoryProviderError),
    #[cfg(feature = "rocm")]
    DeviceExecution(fusion_pcu_rocm::RocmDeviceKernelError),
    #[cfg(feature = "rocm")]
    TensorStorage(crate::PcuDeviceTensorError),
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    TensorInitialization(fusion_pcu_rocm::RocblasError),
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    TensorExecution(fusion_pcu_rocm::RocmTensorExecutionError),
    PreparationDidNotProduceKernel,
    #[cfg(feature = "rocm")]
    KernelBuildDetails(std::string::String),
    #[cfg(feature = "rocm")]
    NoCompatibleDevice(std::vec::Vec<(u32, Self)>),
    #[cfg(feature = "rocm")]
    Discovery(fusion_pcu_rocm::HipError),
    #[cfg(feature = "rocm")]
    Execution(fusion_pcu_rocm::RocmHostKernelError),
    #[cfg(feature = "rocm")]
    BackendInitialization(fusion_pcu_rocm::RocmOwnedDispatchError),
}

impl fmt::Display for PcuExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoBackendEnabled => f.write_str("no executable backend is enabled"),
            Self::ReentrantCall => {
                f.write_str("the thread's PCU execution environment is already in use")
            }
            Self::ThreadUnavailable => {
                f.write_str("the thread execution environment is being destroyed")
            }
            Self::ResidentPolicyConflict => f.write_str(
                "explicit execution policy conflicts with the resident value execution domain",
            ),
            Self::InvalidPolicy => f.write_str("PCU cache capacity and block size must be nonzero"),
            Self::PolicyUnavailable => {
                f.write_str("PCU policy is unavailable or its generation is exhausted")
            }
            Self::Argument(error) => write!(f, "PCU argument rejected: {error:?}"),
            Self::TensorExecutionUnavailable => {
                f.write_str("owned tensor execution requires a compiled tensor-capable backend")
            }
            Self::InvalidTensorSourcePlan => f.write_str("invalid captured tensor graph value"),
            Self::TensorSourceRankMismatch { expected, actual } => {
                write!(
                    f,
                    "tensor source rank mismatch: expected {expected}, found {actual}"
                )
            }
            Self::TensorSourceShapeMismatch { expected, actual } => write!(
                f,
                "tensor source shape mismatch: expected {expected:?}, found {actual:?}"
            ),
            #[cfg(feature = "tensor")]
            Self::TensorBuild(error) => write!(f, "PCU tensor graph construction failed: {error}"),
            Self::RecursiveTensorSource => {
                f.write_str("recursive PCU tensor source helper capture is unsupported")
            }
            Self::TensorSourceNestingLimit => {
                f.write_str("PCU tensor source helper nesting exceeds the limit of 64")
            }
            Self::EmptyTensorInput => f.write_str("owned tensor operations require nonempty input"),
            #[cfg(feature = "rocm")]
            Self::Memory(error) => write!(f, "PCU memory operation failed: {error:?}"),
            #[cfg(feature = "rocm")]
            Self::DeviceExecution(error) => write!(f, "PCU device operation failed: {error}"),
            #[cfg(feature = "rocm")]
            Self::TensorStorage(error) => error.fmt(f),
            #[cfg(all(feature = "rocm", feature = "tensor"))]
            Self::TensorInitialization(error) => {
                write!(f, "PCU tensor state initialization failed: {error}")
            }
            #[cfg(all(feature = "rocm", feature = "tensor"))]
            Self::TensorExecution(error) => write!(f, "PCU tensor execution failed: {error}"),
            Self::KernelBuild => f.write_str("PCU kernel construction failed"),
            Self::PreparationDidNotProduceKernel => {
                f.write_str("PCU cold preparation produced no executable")
            }
            #[cfg(feature = "rocm")]
            Self::KernelBuildDetails(error) => write!(f, "PCU kernel construction failed: {error}"),
            #[cfg(feature = "rocm")]
            Self::NoCompatibleDevice(errors) => {
                f.write_str("no compatible device")?;
                for (device, error) in errors {
                    write!(f, "; device {device}: {error}")?;
                }
                Ok(())
            }
            #[cfg(feature = "rocm")]
            Self::Discovery(error) => write!(f, "PCU discovery failed: {error}"),
            #[cfg(feature = "rocm")]
            Self::Execution(error) => write!(f, "PCU execution failed: {error}"),
            #[cfg(feature = "rocm")]
            Self::BackendInitialization(error) => {
                write!(f, "PCU backend initialization failed: {error}")
            }
        }
    }
}
impl core::error::Error for PcuExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        #[cfg(feature = "tensor")]
        if let Self::TensorBuild(error) = self {
            return Some(error);
        }
        #[cfg(feature = "rocm")]
        match self {
            Self::NoCompatibleDevice(errors) => {
                return errors.first().map(|(_, error)| error as _);
            }
            Self::BackendInitialization(error) => return Some(error),
            Self::Discovery(error) => return Some(error),
            Self::Execution(error) => return Some(error),
            Self::DeviceExecution(error) => return Some(error),
            Self::TensorStorage(error) => return Some(error),
            #[cfg(feature = "tensor")]
            Self::TensorInitialization(error) => return Some(error),
            #[cfg(feature = "tensor")]
            Self::TensorExecution(error) => return Some(error),
            _ => {}
        }
        None
    }
}

/// Per-source-function cache hint. A hint is never trusted without checking specialization identity.
#[doc(hidden)]
pub struct PcuHostCallSite {
    #[cfg(feature = "rocm")]
    slot: AtomicUsize,
}
impl PcuHostCallSite {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(feature = "rocm")]
            slot: AtomicUsize::new(usize::MAX),
        }
    }
}
impl Default for PcuHostCallSite {
    fn default() -> Self {
        Self::new()
    }
}

/// Cold preparation context used by generated code, never by a device invocation.
#[doc(hidden)]
pub struct PcuHostPreparation {
    #[cfg(feature = "rocm")]
    inner: hosted::Preparation,
}
impl PcuHostPreparation {
    /// Admit, select and compile the concrete kernel against runtime devices.
    ///
    /// # Errors
    /// Returns discovery, compatibility or compiler errors; never falls back to CPU.
    #[allow(clippy::missing_const_for_fn)] // Hosted implementations perform runtime IO/state mutation.
    pub fn prepare(
        &mut self,
        kernel: &crate::PcuDispatchKernelIr<'_>,
    ) -> Result<(), PcuExecutionError> {
        #[cfg(feature = "rocm")]
        {
            self.inner.prepare(kernel)
        }
        #[cfg(not(feature = "rocm"))]
        {
            let _ = kernel;
            Err(PcuExecutionError::NoBackendEnabled)
        }
    }
}

/// Replace runtime preferences for future calls. Warm entries invalidate lazily by generation.
///
/// # Errors
/// Rejects invalid preferences or a poisoned configuration lock.
#[allow(clippy::missing_const_for_fn)] // Hosted implementations perform runtime IO/state mutation.
pub fn configure(policy: PcuExecutionPolicy) -> Result<(), PcuExecutionError> {
    if policy.cache_capacity == 0 || policy.block_size == 0 {
        return Err(PcuExecutionError::InvalidPolicy);
    }
    #[cfg(feature = "rocm")]
    {
        hosted::configure(policy)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = policy;
        Err(PcuExecutionError::NoBackendEnabled)
    }
}

/// Restore documented automatic selection preferences.
///
/// # Errors
/// Returns configuration errors, or reports that no backend is enabled.
pub fn use_defaults() -> Result<(), PcuExecutionError> {
    configure(PcuExecutionPolicy::default())
}

/// Clear this thread's prepared entries, selected `ROCm` sessions and discovery snapshot.
///
/// Direct hosted calls complete synchronously, so no command or device-resource lease escapes
/// this cache today.
///
/// # Errors
/// Rejects clearing during a nested active call.
#[allow(clippy::missing_const_for_fn)] // Hosted implementations perform runtime IO/state mutation.
pub fn clear_thread_cache() -> Result<(), PcuExecutionError> {
    #[cfg(feature = "rocm")]
    {
        #[cfg(feature = "tensor")]
        tensor::clear_cache()?;
        hosted::clear_thread_cache()
    }
    #[cfg(not(feature = "rocm"))]
    {
        Ok(())
    }
}

/// Translate a cold builder error without adding work to successful warm calls.
#[doc(hidden)]
pub fn build_error(error: impl fmt::Debug) -> PcuExecutionError {
    #[cfg(feature = "rocm")]
    {
        PcuExecutionError::KernelBuildDetails(std::format!("{error:?}"))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = error;
        PcuExecutionError::KernelBuild
    }
}

/// Preserve structured core tensor graph construction failures at the hosted boundary.
#[doc(hidden)]
#[cfg(feature = "tensor")]
#[must_use]
pub const fn tensor_build_error(error: crate::dialect::tensor::TensorError) -> PcuExecutionError {
    PcuExecutionError::TensorBuild(error)
}

/// Preserve typed argument failures before selection or submission.
#[doc(hidden)]
#[must_use]
pub const fn argument_error(error: PcuArgumentError) -> PcuExecutionError {
    PcuExecutionError::Argument(error)
}

/// Generated ordinary source call with statically typed host/resident borrows.
///
/// # Errors
/// Returns argument, policy, preparation or device execution errors without a CPU fallback.
#[doc(hidden)]
pub fn call_arguments<const N: usize>(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: [PcuCallArgument<'_>; N],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    #[cfg(feature = "rocm")]
    {
        hosted::call_arguments(site, specialization, arguments, prepare)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (site, specialization, arguments, prepare);
        Err(PcuExecutionError::NoBackendEnabled)
    }
}

/// Generated direct entry: obtain a checked cache slot and invoke it with fresh host borrows.
///
/// # Errors
/// Returns preparation, runtime policy or execution errors.
#[doc(hidden)]
pub fn call_host(
    site: &PcuHostCallSite,
    specialization: TypeId,
    arguments: &mut [crate::PcuHostArgument<'_>],
    prepare: impl FnOnce(&mut PcuHostPreparation) -> Result<(), PcuExecutionError>,
) -> Result<(), PcuExecutionError> {
    #[cfg(feature = "rocm")]
    {
        hosted::call_host(site, specialization, arguments, prepare)
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = (site, specialization, arguments, prepare);
        Err(PcuExecutionError::NoBackendEnabled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_capacity_or_geometry_is_rejected_before_policy_changes() {
        for policy in [
            PcuExecutionPolicy {
                cache_capacity: 0,
                ..PcuExecutionPolicy::default()
            },
            PcuExecutionPolicy {
                block_size: 0,
                ..PcuExecutionPolicy::default()
            },
        ] {
            assert!(matches!(
                configure(policy),
                Err(PcuExecutionError::InvalidPolicy)
            ));
        }
    }

    #[cfg(all(feature = "rocm", feature = "tensor"))]
    #[test]
    fn device_rejections_retain_typed_causes_and_readable_device_order() {
        let error = PcuExecutionError::NoCompatibleDevice(alloc::vec![
            (
                3,
                PcuExecutionError::TensorExecution(
                    fusion_pcu_rocm::RocmTensorExecutionError::UnsupportedScalarType(
                        crate::PcuScalarType::U32,
                    ),
                ),
            ),
            (
                7,
                PcuExecutionError::BackendInitialization(
                    fusion_pcu_rocm::RocmOwnedDispatchError::InvalidBlockSize,
                ),
            ),
        ]);
        assert!(matches!(
            &error,
            PcuExecutionError::NoCompatibleDevice(rejections)
                if matches!(
                    &rejections[0].1,
                    PcuExecutionError::TensorExecution(
                        fusion_pcu_rocm::RocmTensorExecutionError::UnsupportedScalarType(
                            crate::PcuScalarType::U32
                        )
                    )
                )
        ));
        let message = error.to_string();
        assert!(
            message
                .find("device 3:")
                .zip(message.find("device 7:"))
                .is_some_and(|(first, second)| first < second)
        );
        assert!(message.contains("U32"));
        assert!(message.contains("block size"));
        assert!(core::error::Error::source(&error).is_some());
    }

    #[cfg(not(feature = "rocm"))]
    #[test]
    fn disabled_provider_does_not_build_or_fall_back_to_cpu() {
        let mut constructed = false;
        let mut output = [123_u32];
        let mut arguments = [crate::PcuHostArgument::read_write(
            crate::PcuBindingRef::new(0, 0),
            &mut output,
        )];
        let result = call_host(
            &PcuHostCallSite::new(),
            TypeId::of::<u32>(),
            &mut arguments,
            |_| {
                constructed = true;
                Ok(())
            },
        );
        assert!(matches!(result, Err(PcuExecutionError::NoBackendEnabled)));
        assert!(!constructed);
        assert_eq!(output, [123]);
    }
}
