//! Overridable hosted execution policy and cold preparation for direct PCU calls.
//!
//! Backend selection is runtime policy. The core IR and dialects never depend on this module.
//! Prepared state is thread-owned; no backend receives an invented `Send` or `Sync` promise.

use core::any::TypeId;
use core::fmt;
#[cfg(feature = "std")]
mod policy;
#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
use core::sync::atomic::AtomicUsize;

#[cfg(feature = "rocm")]
#[path = "global/hosted.rs"]
mod hosted;
#[cfg(any(feature = "cuda", feature = "metal"))]
#[path = "global/provider_hosted/provider_hosted.rs"]
mod provider_hosted;

#[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
#[path = "global/resident/resident.rs"]
mod resident;
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
    PcuResidentBufferOwner,
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
    /// Require `CUDA`, failing if it is disabled or no compatible device can be opened.
    #[cfg(feature = "cuda")]
    Cuda,
    /// Require Metal; disabled or incompatible devices never substitute another provider.
    #[cfg(feature = "metal")]
    Metal,
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
    /// Default for checked F32/F64 tensor arithmetic in unannotated owned source functions.
    /// Explicit function flags override this value without changing other helpers.
    pub float_underflow: crate::PcuFloatUnderflowPolicy,
    /// Default range handling for checked invocation floating arithmetic.
    /// Owned tensor graph calls currently require `Reject` and fail explicitly for `Clamp`.
    pub range_policy: crate::PcuRangePolicy,
    /// Default compound numerical contract; explicit owned helper flags override this value.
    pub numerical_mode: crate::PcuNumericalMode,
    /// Independent compound arithmetic, precision and reproducibility defaults.
    /// Function-local overrides inherit unrelated fields. Library/API failures remain errors
    /// even under explicitly backend-defined compound numerical arithmetic.
    pub numerical_options: crate::PcuNumericalOptions,
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
            float_underflow: crate::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: crate::PcuRangePolicy::Reject,
            numerical_mode: crate::PcuNumericalMode::Boundary,
            numerical_options: crate::PcuNumericalOptions::default(),
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
    /// A terminal arithmetic fault, independent of the selected backend.
    ArithmeticFault(crate::PcuExecutionFault),
    NoBackendEnabled,
    /// Cold source admission failures retain each selected physical provider reference.
    #[cfg(any(feature = "cuda", feature = "metal"))]
    NoCompatibleInvocationDevice {
        rejected: alloc::vec::Vec<(crate::PcuObjectRef, Self)>,
        discovery: alloc::vec::Vec<alloc::string::String>,
    },
    BackendFailure(alloc::string::String),
    ReentrantCall,
    ThreadUnavailable,
    InvalidPolicy,
    /// The selected clamped range mode is not supported by this execution profile.
    UnsupportedRangePolicy,
    /// No implementation of the requested numerical requirements is admitted for this call.
    UnsupportedNumericalOptions(crate::PcuNumericalOptions),
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
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    Memory(crate::PcuMemoryProviderError),
    #[cfg(feature = "rocm")]
    DeviceExecution(fusion_pcu_rocm::RocmDeviceKernelError),
    #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
    TensorStorage(crate::PcuDeviceTensorError),
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    TensorInitialization(fusion_pcu_rocm::RocblasError),
    #[cfg(all(feature = "rocm", feature = "tensor"))]
    TensorExecution(fusion_pcu_rocm::RocmTensorExecutionError),
    #[cfg(all(feature = "cuda", feature = "tensor"))]
    CudaTensorExecution(fusion_pcu_cuda::CudaTensorExecutionError),
    /// Cold candidate failures retain provider identity and structured native rejection reasons.
    #[cfg(all(feature = "cuda", feature = "tensor"))]
    NoCompatibleResidentDevice(alloc::vec::Vec<(crate::PcuObjectRef, Self)>),
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
            Self::ArithmeticFault(fault) => write!(
                f,
                "PCU {}arithmetic fault {:?} at logical invocation {}",
                if fault.recovered { "recovered " } else { "" },
                fault.kind,
                fault.invocation_id
            ),
            Self::NoBackendEnabled => f.write_str("no executable backend is enabled"),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            Self::NoCompatibleInvocationDevice {
                rejected,
                discovery,
            } => format_invocation_rejections(f, rejected, discovery),
            Self::BackendFailure(error) => f.write_str(error),
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
            Self::UnsupportedRangePolicy => {
                f.write_str("clamped range handling is unsupported by this PCU execution profile")
            }
            Self::UnsupportedNumericalOptions(options) => {
                write!(f, "PCU numerical requirements are unsupported: {options:?}")
            }
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
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            Self::Memory(error) => write!(f, "PCU memory operation failed: {error:?}"),
            #[cfg(feature = "rocm")]
            Self::DeviceExecution(error) => write!(f, "PCU device operation failed: {error}"),
            #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
            Self::TensorStorage(error) => error.fmt(f),
            #[cfg(all(feature = "rocm", feature = "tensor"))]
            Self::TensorInitialization(error) => {
                write!(f, "PCU tensor state initialization failed: {error}")
            }
            #[cfg(all(feature = "rocm", feature = "tensor"))]
            Self::TensorExecution(error) => write!(f, "PCU tensor execution failed: {error}"),
            #[cfg(all(feature = "cuda", feature = "tensor"))]
            Self::CudaTensorExecution(error) => {
                write!(f, "PCU CUDA tensor execution failed: {error}")
            }
            #[cfg(all(feature = "cuda", feature = "tensor"))]
            Self::NoCompatibleResidentDevice(errors) => format_resident_rejections(f, errors),
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

impl PcuExecutionError {
    /// Returns the first recovered range fault when an invocation completed with a clamped value.
    ///
    /// Fatal arithmetic faults and non-arithmetic execution errors return `None`.
    #[must_use]
    pub const fn recovered_range_fault(&self) -> Option<crate::PcuExecutionFault> {
        match self {
            Self::ArithmeticFault(fault) if fault.recovered => Some(*fault),
            _ => None,
        }
    }
}
impl core::error::Error for PcuExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        #[cfg(feature = "tensor")]
        if let Self::TensorBuild(error) = self {
            return Some(error);
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        if let Self::TensorStorage(error) = self {
            return Some(error);
        }
        #[cfg(all(feature = "cuda", feature = "tensor"))]
        match self {
            Self::CudaTensorExecution(error) => return Some(error),
            Self::NoCompatibleResidentDevice(errors) => {
                return errors.first().map(|(_, error)| error as _);
            }
            _ => {}
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
            #[cfg(feature = "tensor")]
            Self::TensorInitialization(error) => return Some(error),
            #[cfg(feature = "tensor")]
            Self::TensorExecution(error) => return Some(error),
            _ => {}
        }
        None
    }
}

#[cfg(feature = "rocm")]
impl From<fusion_pcu_rocm::RocmHostKernelError> for PcuExecutionError {
    fn from(error: fusion_pcu_rocm::RocmHostKernelError) -> Self {
        match error {
            fusion_pcu_rocm::RocmHostKernelError::CheckedExecutionFault(fault) => {
                Self::ArithmeticFault(fault)
            }
            other => Self::Execution(other),
        }
    }
}

#[cfg(feature = "rocm")]
impl From<fusion_pcu_rocm::RocmDeviceKernelError> for PcuExecutionError {
    fn from(error: fusion_pcu_rocm::RocmDeviceKernelError) -> Self {
        match error {
            fusion_pcu_rocm::RocmDeviceKernelError::CheckedExecutionFault(fault) => {
                Self::ArithmeticFault(fault)
            }
            other => Self::DeviceExecution(other),
        }
    }
}

#[cfg(all(feature = "rocm", feature = "tensor"))]
impl From<fusion_pcu_rocm::RocmTensorExecutionError> for PcuExecutionError {
    fn from(error: fusion_pcu_rocm::RocmTensorExecutionError) -> Self {
        match error {
            fusion_pcu_rocm::RocmTensorExecutionError::ExecutionFault(fault) => {
                Self::ArithmeticFault(fault)
            }
            other => Self::TensorExecution(other),
        }
    }
}

/// Per-source-function cache hint. A hint is never trusted without checking specialization identity.
#[doc(hidden)]
pub struct PcuHostCallSite {
    #[cfg(any(feature = "rocm", all(feature = "cuda", feature = "tensor")))]
    slot: AtomicUsize,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    provider_slot: AtomicUsize,
}
impl PcuHostCallSite {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(any(feature = "rocm", all(feature = "cuda", feature = "tensor")))]
            slot: AtomicUsize::new(usize::MAX),
            #[cfg(any(feature = "cuda", feature = "metal"))]
            provider_slot: AtomicUsize::new(usize::MAX),
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
    inner: Option<hosted::Preparation>,
    #[cfg(any(feature = "cuda", feature = "metal"))]
    shared: Option<provider_hosted::Preparation>,
}
impl PcuHostPreparation {
    /// Returns the policy captured for this cold preparation, without reading global state.
    #[must_use]
    pub const fn float_underflow_policy(&self) -> crate::PcuFloatUnderflowPolicy {
        #[cfg(feature = "rocm")]
        {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            if let Some(shared) = &self.shared {
                return shared.float_underflow_policy();
            }
            if let Some(inner) = &self.inner {
                inner.float_underflow_policy()
            } else {
                crate::PcuFloatUnderflowPolicy::IeeeAfterRounding
            }
        }
        #[cfg(all(not(feature = "rocm"), any(feature = "cuda", feature = "metal")))]
        {
            if let Some(shared) = &self.shared {
                shared.float_underflow_policy()
            } else {
                crate::PcuFloatUnderflowPolicy::IeeeAfterRounding
            }
        }
        #[cfg(all(not(feature = "rocm"), not(any(feature = "cuda", feature = "metal"))))]
        {
            crate::PcuFloatUnderflowPolicy::IeeeAfterRounding
        }
    }

    /// Returns the range policy captured for this cold preparation, without reading global state.
    #[must_use]
    pub const fn range_policy(&self) -> crate::PcuRangePolicy {
        #[cfg(feature = "rocm")]
        {
            #[cfg(any(feature = "cuda", feature = "metal"))]
            if let Some(shared) = &self.shared {
                return shared.range_policy();
            }
            if let Some(inner) = &self.inner {
                inner.range_policy()
            } else {
                crate::PcuRangePolicy::Reject
            }
        }
        #[cfg(all(not(feature = "rocm"), any(feature = "cuda", feature = "metal")))]
        {
            if let Some(shared) = &self.shared {
                shared.range_policy()
            } else {
                crate::PcuRangePolicy::Reject
            }
        }
        #[cfg(all(not(feature = "rocm"), not(any(feature = "cuda", feature = "metal"))))]
        {
            crate::PcuRangePolicy::Reject
        }
    }

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
            #[cfg(any(feature = "cuda", feature = "metal"))]
            if let Some(shared) = &mut self.shared {
                return shared.prepare(kernel);
            }
            self.inner
                .as_mut()
                .ok_or(PcuExecutionError::NoBackendEnabled)?
                .prepare(kernel)
        }
        #[cfg(all(not(feature = "rocm"), any(feature = "cuda", feature = "metal")))]
        {
            self.shared
                .as_mut()
                .ok_or(PcuExecutionError::NoBackendEnabled)?
                .prepare(kernel)
        }
        #[cfg(all(not(feature = "rocm"), not(any(feature = "cuda", feature = "metal"))))]
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
    #[cfg(feature = "std")]
    {
        policy::configure(policy)
    }
    #[cfg(not(feature = "std"))]
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

/// Clear this thread's prepared entries, provider session roots and discovery snapshots.
///
/// Direct hosted calls complete synchronously, so no command or device-resource lease escapes
/// this cache today.
///
/// # Errors
/// Rejects clearing during a nested active call.
#[allow(clippy::missing_const_for_fn)] // Hosted implementations perform runtime IO/state mutation.
pub fn clear_thread_cache() -> Result<(), PcuExecutionError> {
    #[cfg(any(feature = "cuda", feature = "metal"))]
    provider_hosted::clear_thread_cache()?;
    #[cfg(all(any(feature = "rocm", feature = "cuda"), feature = "tensor"))]
    {
        tensor::clear_cache()?;
        resident::clear_roots()?;
    }
    #[cfg(feature = "rocm")]
    {
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
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        provider_hosted::call_arguments(site, specialization, arguments, prepare)
    }
    #[cfg(all(feature = "rocm", not(any(feature = "cuda", feature = "metal"))))]
    {
        hosted::call_arguments(site, specialization, arguments, prepare)
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
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
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
        let route = policy::route();
        #[cfg(feature = "rocm")]
        if matches!(route.backend, PcuBackendChoice::Rocm) {
            return hosted::call_host(site, specialization, arguments, prepare);
        }
        if matches!(route.backend, PcuBackendChoice::Rocm) && !cfg!(feature = "rocm") {
            return Err(PcuExecutionError::NoBackendEnabled);
        }
        provider_hosted::call_host(site, specialization, arguments, prepare)
    }
    #[cfg(all(feature = "rocm", not(any(feature = "cuda", feature = "metal"))))]
    {
        hosted::call_host(site, specialization, arguments, prepare)
    }
    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
    {
        let _ = (site, specialization, arguments, prepare);
        Err(PcuExecutionError::NoBackendEnabled)
    }
}

#[cfg(all(feature = "cuda", feature = "tensor"))]
fn format_resident_rejections(
    f: &mut fmt::Formatter<'_>,
    errors: &[(crate::PcuObjectRef, PcuExecutionError)],
) -> fmt::Result {
    f.write_str("no compatible resident device")?;
    for (device, error) in errors {
        write!(
            f,
            "; provider {:?} device {}: {error}",
            device.provider, device.id
        )?;
    }
    Ok(())
}

#[cfg(any(feature = "cuda", feature = "metal"))]
fn format_invocation_rejections(
    f: &mut fmt::Formatter<'_>,
    rejected: &[(crate::PcuObjectRef, PcuExecutionError)],
    discovery: &[alloc::string::String],
) -> fmt::Result {
    f.write_str("no compatible source device")?;
    for (device, error) in rejected {
        write!(
            f,
            "; provider {:?} device {}: {error}",
            device.provider, device.id
        )?;
    }
    for error in discovery {
        write!(f, "; {error}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invocation_range_policy_defaults_to_rejection() {
        assert_eq!(
            PcuExecutionPolicy::default().range_policy,
            crate::PcuRangePolicy::Reject
        );
    }

    #[test]
    fn compound_numerical_mode_defaults_to_boundary() {
        assert_eq!(
            PcuExecutionPolicy::default().numerical_mode,
            crate::PcuNumericalMode::Boundary
        );
    }

    #[cfg(feature = "rocm")]
    #[test]
    fn terminal_arithmetic_faults_have_a_backend_neutral_consuming_error() {
        for kind in [
            crate::PcuExecutionFaultKind::DivideByZero,
            crate::PcuExecutionFaultKind::SignedDivisionOverflow,
            crate::PcuExecutionFaultKind::ArithmeticOverflow,
            crate::PcuExecutionFaultKind::ArithmeticUnderflow,
            crate::PcuExecutionFaultKind::InvalidFloatingOperand,
        ] {
            let fault = crate::PcuExecutionFault {
                kind,
                invocation_id: 37,
                recovered: false,
            };
            let host = PcuExecutionError::from(
                fusion_pcu_rocm::RocmHostKernelError::CheckedExecutionFault(fault),
            );
            let device = PcuExecutionError::from(
                fusion_pcu_rocm::RocmDeviceKernelError::CheckedExecutionFault(fault),
            );
            assert!(matches!(host, PcuExecutionError::ArithmeticFault(actual) if actual == fault));
            assert!(
                matches!(device, PcuExecutionError::ArithmeticFault(actual) if actual == fault)
            );
            #[cfg(feature = "tensor")]
            {
                let tensor = PcuExecutionError::from(
                    fusion_pcu_rocm::RocmTensorExecutionError::ExecutionFault(fault),
                );
                assert!(
                    matches!(tensor, PcuExecutionError::ArithmeticFault(actual) if actual == fault)
                );
            }
        }
        let recovered = crate::PcuExecutionFault {
            kind: crate::PcuExecutionFaultKind::ArithmeticOverflow,
            invocation_id: 12,
            recovered: true,
        };
        let error = PcuExecutionError::ArithmeticFault(recovered);
        assert_eq!(error.recovered_range_fault(), Some(recovered));
        let fatal = PcuExecutionError::ArithmeticFault(crate::PcuExecutionFault {
            recovered: false,
            ..recovered
        });
        assert_eq!(fatal.recovered_range_fault(), None);
        let unavailable = PcuExecutionError::from(
            fusion_pcu_rocm::RocmHostKernelError::PoisonedAfterUncertainCompletion,
        );
        assert!(matches!(
            unavailable,
            PcuExecutionError::Execution(
                fusion_pcu_rocm::RocmHostKernelError::PoisonedAfterUncertainCompletion
            )
        ));
    }

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

    #[cfg(not(any(feature = "rocm", feature = "cuda", feature = "metal")))]
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
