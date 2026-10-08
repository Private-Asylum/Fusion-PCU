//! Overridable hosted execution policy and cold preparation for direct PCU calls.
//!
//! Backend selection is runtime policy. The core IR and dialects never depend on this module.
//! Prepared state is thread-owned; no backend receives an invented `Send` or `Sync` promise.

use core::any::TypeId;
use core::fmt;
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
#[path = "global/numerical/numerical.rs"]
mod numerical;
#[cfg(feature = "std")]
mod policy;
#[cfg(feature = "vulkan")]
#[path = "global/vulkan/vulkan.rs"]
pub mod vulkan;
#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
use core::sync::atomic::AtomicUsize;

#[cfg(feature = "rocm")]
#[path = "global/hosted.rs"]
mod hosted;
#[cfg(any(
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
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
#[path = "global/selection/selection.rs"]
mod selection;
#[rustfmt::skip]
pub use selection::{
    PcuInvocationCandidate,
    PcuInvocationScorer,
};

mod tensor;
#[cfg(feature = "tensor")]
#[doc(hidden)]
#[rustfmt::skip]
pub use tensor::{
    __pcu_capture_tensor_program,
    __pcu_capture_tensor_program_outputs,
    PcuCapturedTensorProgram,
};
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
    PcuImmutableTensorPayload,
    call_consumed_owners_tensor_capture,
    call_consumed_pair_tensor_capture,
    call_consumed_tensor_capture,
    call_owned_tensor_capture,
    call_owned_tensors_capture,
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
    /// Require Vulkan compute; resident imports require a separate proved interop route.
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// Require the explicitly compiled CPU host provider; CPU is never silently compiled in.
    #[cfg(feature = "cpu")]
    Cpu,
    /// Require the pinned MLX provider; only its explicitly admitted scalar/tensor profiles run.
    #[cfg(feature = "mlx")]
    Mlx,
}

/// Runtime preferences for later calls; an already executing call retains its selected state.
#[derive(Clone, Copy, Debug)]
pub struct PcuExecutionPolicy {
    pub backend: PcuBackendChoice,
    /// Runtime device ordinal. An explicit request never substitutes another device.
    pub device: Option<u32>,
    /// Maximum cached specializations per host thread.
    pub cache_capacity: usize,
    /// ROCm/CUDA launch block size, checked by device preparation.
    /// Fixed implementation profiles (currently Vulkan Copy/Neg) own their workgroup width.
    pub block_size: u32,
    /// Default floating tininess policy for checked invocation and tensor arithmetic.
    /// Integer operations retain this setting in their identity but have no floating tininess.
    /// Explicit function flags override this value without changing unrelated options.
    pub float_underflow: crate::PcuFloatUnderflowPolicy,
    /// Default range handling for eligible checked integer and floating invocation arithmetic.
    /// Joint integer division requires Reject; no clamped quotient/remainder is specified.
    /// Owned tensor graph calls currently require `Reject` and fail explicitly for `Clamp`.
    pub range_policy: crate::PcuRangePolicy,
    /// Default compound numerical contract; explicit owned helper flags override this value.
    pub numerical_mode: crate::PcuNumericalMode,
    /// Independent compound arithmetic, precision and reproducibility defaults.
    /// Function-local overrides inherit unrelated fields. Library/API failures remain errors
    /// even under explicitly backend-defined compound numerical arithmetic.
    pub numerical_options: crate::PcuNumericalOptions,
    /// Host observation of internal checked stages, independent of numerical policy.
    /// Automatic permits backend-proven guarded execution within a synchronous function call;
    /// the call still resolves its promised errors before returning. Host-observed stages
    /// preserve intermediate checked submission boundaries without forcing payload readback.
    pub observation: crate::PcuExecutionObservationPolicy,
    /// Cold candidate scoring after explicit device filtering; higher scores rank first.
    pub score_device: fn(&crate::PcuDeviceDescriptor<'_>, u64) -> i128,
    /// Optional cold invocation ranking with actual IR and directly reported physical facts.
    /// Overrides `score_device` for invocation preparation; tensor roots retain device scoring.
    /// Cached warm calls and resident-affinity calls do not rank or query facts.
    pub score_invocation: Option<PcuInvocationScorer>,
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
            observation: crate::PcuExecutionObservationPolicy::Automatic,
            score_device: default_device_score,
            score_invocation: None,
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
    /// Cold candidate rejection for host or resident execution, independent of backend.
    ///
    /// Each physical reference retains provider, generation, kind and identifier;
    /// original typed causes and candidate order remain observable. Discovery
    /// diagnostics do not replace candidate errors or authorize CPU fallback.
    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    NoCompatibleDevice {
        rejected: alloc::vec::Vec<(crate::PcuObjectRef, Self)>,
        discovery: alloc::vec::Vec<alloc::string::String>,
    },
    BackendFailure(alloc::string::String),
    #[cfg(feature = "vulkan")]
    VulkanExecution(fusion_pcu_vulkan::PcuVulkanError),
    #[cfg(feature = "cpu")]
    CpuExecution(fusion_pcu_cpu::PcuCpuHostError),
    #[cfg(feature = "cpu")]
    CpuDiscovery(fusion_pcu_cpu::PcuCpuDiscoveryError),
    #[cfg(feature = "mlx")]
    MlxExecution(fusion_pcu_mlx::MlxError),
    /// Preserves typed MLX binding/admission errors without erasing their resource reference.
    #[cfg(feature = "mlx")]
    MlxHostExecution(fusion_pcu_mlx::MlxHostKernelError),
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
    PreparationDidNotProduceKernel,
    #[cfg(feature = "rocm")]
    KernelBuildDetails(std::string::String),
    #[cfg(feature = "rocm")]
    Discovery(fusion_pcu_rocm::HipError),
    #[cfg(feature = "rocm")]
    Execution(fusion_pcu_rocm::RocmHostKernelError),
    #[cfg(feature = "rocm")]
    BackendInitialization(fusion_pcu_rocm::RocmOwnedDispatchError),
}

impl fmt::Display for PcuExecutionError {
    #[allow(clippy::too_many_lines)] // Keep one exhaustive diagnostic arm per structured failure.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ArithmeticFault(fault) => format_arithmetic_fault(f, *fault),
            Self::NoBackendEnabled => f.write_str("no executable backend is enabled"),
            #[cfg(any(
                feature = "rocm",
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "cpu",
                feature = "mlx"
            ))]
            Self::NoCompatibleDevice {
                rejected,
                discovery,
            } => format_device_rejections(f, rejected, discovery),
            Self::BackendFailure(error) => f.write_str(error),
            #[cfg(feature = "cpu")]
            Self::CpuExecution(error) => write!(f, "PCU CPU execution failed: {error:?}"),
            #[cfg(feature = "cpu")]
            Self::CpuDiscovery(error) => write!(f, "PCU CPU discovery failed: {error:?}"),
            #[cfg(feature = "mlx")]
            Self::MlxExecution(error) => write!(f, "PCU MLX execution failed: {error}"),
            #[cfg(feature = "mlx")]
            Self::MlxHostExecution(error) => write!(f, "PCU MLX host execution failed: {error:?}"),
            #[cfg(feature = "vulkan")]
            Self::VulkanExecution(error) => write!(f, "PCU Vulkan execution failed: {error}"),
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
            Self::KernelBuild => f.write_str("PCU kernel construction failed"),
            Self::PreparationDidNotProduceKernel => {
                f.write_str("PCU cold preparation produced no executable")
            }
            #[cfg(feature = "rocm")]
            Self::KernelBuildDetails(error) => write!(f, "PCU kernel construction failed: {error}"),
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
    /// Returns common execution fault facts without discarding richer graph diagnostics.
    ///
    /// Tensor reference errors retain their node and, for strict compounds, reduction
    /// step in the original error. This view reports the row-major output element as
    /// the invocation ID. Admission, allocation and unrelated failures return `None`.
    #[must_use]
    #[cfg_attr(not(feature = "tensor"), allow(clippy::missing_const_for_fn))]
    // Tensor indices require checked conversion; provider-only matches are const-capable.
    pub fn arithmetic_fault(&self) -> Option<crate::PcuExecutionFault> {
        match self {
            Self::ArithmeticFault(fault) => Some(*fault),
            #[cfg(feature = "tensor")]
            Self::TensorBuild(error) => tensor_arithmetic_fault(error),
            #[cfg(all(feature = "rocm", feature = "tensor"))]
            Self::TensorExecution(fusion_pcu_rocm::RocmTensorExecutionError::Graph(error)) => {
                tensor_arithmetic_fault(error)
            }
            #[cfg(all(feature = "cuda", feature = "tensor"))]
            Self::CudaTensorExecution(fusion_pcu_cuda::CudaTensorExecutionError::Graph(error)) => {
                tensor_arithmetic_fault(error)
            }
            #[cfg(all(feature = "rocm", feature = "tensor"))]
            Self::TensorExecution(fusion_pcu_rocm::RocmTensorExecutionError::ExecutionFault(
                fault,
            )) => Some(*fault),
            #[cfg(all(feature = "cuda", feature = "tensor"))]
            Self::CudaTensorExecution(
                fusion_pcu_cuda::CudaTensorExecutionError::ExecutionFault(fault),
            ) => Some(*fault),
            _ => None,
        }
    }

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

#[cfg(feature = "tensor")]
fn tensor_arithmetic_fault(
    error: &crate::dialect::tensor::TensorError,
) -> Option<crate::PcuExecutionFault> {
    match error {
        crate::dialect::tensor::TensorError::ArithmeticFault {
            element_index,
            kind,
            ..
        }
        | crate::dialect::tensor::TensorError::CompoundArithmeticFault {
            element_index,
            kind,
            ..
        } => Some(crate::PcuExecutionFault {
            invocation_id: u64::try_from(*element_index).ok()?,
            kind: *kind,
            recovered: false,
        }),
        _ => None,
    }
}
impl core::error::Error for PcuExecutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        #[cfg(any(
            feature = "rocm",
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "cpu",
            feature = "mlx"
        ))]
        if let Self::NoCompatibleDevice { rejected, .. } = self {
            return rejected.first().map(|(_, error)| error as _);
        }
        #[cfg(feature = "tensor")]
        if let Self::TensorBuild(error) = self {
            return Some(error);
        }
        #[cfg(any(feature = "rocm", feature = "cuda", feature = "metal"))]
        if let Self::TensorStorage(error) = self {
            return Some(error);
        }
        #[cfg(all(feature = "cuda", feature = "tensor"))]
        if let Self::CudaTensorExecution(error) = self {
            return Some(error);
        }
        #[cfg(feature = "rocm")]
        match self {
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
        #[cfg(feature = "vulkan")]
        if let Self::VulkanExecution(error) = self {
            return Some(error);
        }
        #[cfg(feature = "mlx")]
        match self {
            Self::MlxExecution(error) => return Some(error),
            Self::MlxHostExecution(crate::PcuHostDispatchError::Backend(error)) => {
                return Some(error);
            }
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
///
/// Warm hits leave an already matching hint untouched, avoiding shared cache-line writes across
/// calling threads. Concurrent changes are harmless: entries live in thread-local caches and
/// every lookup validates the observed hint before use; the atomic never publishes entry state.
#[doc(hidden)]
pub struct PcuHostCallSite {
    #[cfg(any(feature = "rocm", all(feature = "cuda", feature = "tensor")))]
    slot: AtomicUsize,
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    provider_slot: AtomicUsize,
}
impl PcuHostCallSite {
    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    fn remember_hint(hint: &AtomicUsize, observed: usize, selected: usize) {
        if selected != observed {
            hint.store(selected, core::sync::atomic::Ordering::Relaxed);
        }
    }

    #[must_use]
    pub const fn new() -> Self {
        Self {
            #[cfg(any(feature = "rocm", all(feature = "cuda", feature = "tensor")))]
            slot: AtomicUsize::new(usize::MAX),
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "cpu",
                feature = "mlx"
            ))]
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
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    shared: Option<provider_hosted::Preparation>,
}
impl PcuHostPreparation {
    /// Returns all numerical defaults captured for this cold preparation.
    #[must_use]
    pub const fn numerical_requirements(&self) -> crate::PcuImplementationRequirements {
        #[cfg(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "cpu",
            feature = "mlx"
        ))]
        if let Some(shared) = &self.shared {
            return shared.numerical_requirements();
        }
        #[cfg(feature = "rocm")]
        if let Some(inner) = &self.inner {
            return inner.numerical_requirements();
        }
        crate::PcuImplementationRequirements::DEFAULT
    }

    /// Returns the underflow default without reading global state.
    #[must_use]
    pub const fn float_underflow_policy(&self) -> crate::PcuFloatUnderflowPolicy {
        self.numerical_requirements().float_underflow
    }

    /// Returns the range default without reading global state.
    #[must_use]
    pub const fn range_policy(&self) -> crate::PcuRangePolicy {
        self.numerical_requirements().range_policy
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
            #[cfg(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "cpu",
                feature = "mlx"
            ))]
            if let Some(shared) = &mut self.shared {
                return shared.prepare(kernel);
            }
            self.inner
                .as_mut()
                .ok_or(PcuExecutionError::NoBackendEnabled)?
                .prepare(kernel)
        }
        #[cfg(all(
            not(feature = "rocm"),
            any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "cpu",
                feature = "mlx"
            )
        ))]
        {
            self.shared
                .as_mut()
                .ok_or(PcuExecutionError::NoBackendEnabled)?
                .prepare(kernel)
        }
        #[cfg(all(
            not(feature = "rocm"),
            not(any(
                feature = "cuda",
                feature = "metal",
                feature = "vulkan",
                feature = "cpu",
                feature = "mlx"
            ))
        ))]
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
/// Escaped resident values retain their execution roots independently and remain valid. Known
/// CUDA/ROCm sessions release completed idle host publication caches while retaining live tickets
/// and quarantined endpoints. Later transfers may refill those caches. This does not trim every
/// backend workspace or impose a persistent memory limit.
///
/// # Errors
/// Rejects clearing during a nested active call or while an idle publication cache is locked.
#[allow(clippy::missing_const_for_fn)] // Hosted implementations perform runtime IO/state mutation.
pub fn clear_thread_cache() -> Result<(), PcuExecutionError> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    provider_hosted::clear_thread_cache()?;
    #[cfg(all(
        feature = "tensor",
        any(
            feature = "mlx",
            feature = "cpu",
            feature = "vulkan",
            feature = "metal"
        )
    ))]
    tensor::clear_opaque_cache()?;
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
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    {
        provider_hosted::call_arguments(site, specialization, arguments, prepare)
    }
    #[cfg(all(
        feature = "rocm",
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "cpu",
            feature = "mlx"
        ))
    ))]
    {
        hosted::call_arguments(site, specialization, arguments, prepare)
    }
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    )))]
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
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
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
    #[cfg(all(
        feature = "rocm",
        not(any(
            feature = "cuda",
            feature = "metal",
            feature = "vulkan",
            feature = "cpu",
            feature = "mlx"
        ))
    ))]
    {
        hosted::call_host(site, specialization, arguments, prepare)
    }
    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    )))]
    {
        let _ = (site, specialization, arguments, prepare);
        Err(PcuExecutionError::NoBackendEnabled)
    }
}

#[cfg(any(
    feature = "rocm",
    feature = "cuda",
    feature = "metal",
    feature = "vulkan",
    feature = "cpu",
    feature = "mlx"
))]
fn format_device_rejections(
    f: &mut fmt::Formatter<'_>,
    rejected: &[(crate::PcuObjectRef, PcuExecutionError)],
    discovery: &[alloc::string::String],
) -> fmt::Result {
    f.write_str("no compatible device")?;
    for (device, error) in rejected {
        write!(
            f,
            "; provider {:?} generation {} {:?} device {}: {error}",
            device.provider, device.generation, device.kind, device.id
        )?;
    }
    for error in discovery {
        write!(f, "; {error}")?;
    }
    Ok(())
}

fn format_arithmetic_fault(
    f: &mut fmt::Formatter<'_>,
    fault: crate::PcuExecutionFault,
) -> fmt::Result {
    write!(
        f,
        "PCU {}arithmetic fault {:?} at logical invocation {}",
        if fault.recovered { "recovered " } else { "" },
        fault.kind,
        fault.invocation_id
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    ))]
    #[test]
    fn candidate_rejections_preserve_full_identity_order_and_typed_cause() {
        let first = crate::PcuObjectRef {
            provider: crate::PcuProviderId(7),
            generation: 11,
            kind: crate::PcuObjectKind::Device,
            id: 3,
        };
        let second = crate::PcuObjectRef {
            provider: crate::PcuProviderId(8),
            ..first
        };
        let third = crate::PcuObjectRef {
            generation: 12,
            ..first
        };
        let error = PcuExecutionError::NoCompatibleDevice {
            rejected: alloc::vec![
                (first, PcuExecutionError::UnsupportedRangePolicy),
                (second, PcuExecutionError::InvalidPolicy),
                (third, PcuExecutionError::KernelBuild),
            ],
            discovery: alloc::vec!["provider unavailable before enumeration".into()],
        };
        let PcuExecutionError::NoCompatibleDevice {
            rejected,
            discovery,
        } = &error
        else {
            panic!("candidate rejection lost its structured envelope")
        };
        assert_eq!(
            rejected
                .iter()
                .map(|(device, _)| *device)
                .collect::<alloc::vec::Vec<_>>(),
            [first, second, third]
        );
        assert!(matches!(
            rejected[0].1,
            PcuExecutionError::UnsupportedRangePolicy
        ));
        assert_eq!(discovery, &["provider unavailable before enumeration"]);
        let message = error.to_string();
        let positions = [
            "provider PcuProviderId(7) generation 11",
            "provider PcuProviderId(8) generation 11",
            "provider PcuProviderId(7) generation 12",
            "provider unavailable before enumeration",
        ]
        .map(|text| message.find(text).unwrap());
        assert!(positions.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            core::error::Error::source(&error).unwrap().to_string(),
            PcuExecutionError::UnsupportedRangePolicy.to_string()
        );
        let discovery_only = PcuExecutionError::NoCompatibleDevice {
            rejected: alloc::vec![],
            discovery: alloc::vec!["no runtime installed".into()],
        };
        assert!(core::error::Error::source(&discovery_only).is_none());
        assert!(discovery_only.to_string().contains("no runtime installed"));
    }

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

    #[test]
    fn stage_observation_is_independent_of_numerical_defaults() {
        let automatic = PcuExecutionPolicy::default();
        let observed = PcuExecutionPolicy {
            observation: crate::PcuExecutionObservationPolicy::HostObservedStages,
            ..automatic
        };
        assert_eq!(
            automatic.observation,
            crate::PcuExecutionObservationPolicy::Automatic
        );
        assert_eq!(observed.numerical_mode, automatic.numerical_mode);
        assert_eq!(observed.numerical_options, automatic.numerical_options);
        assert_eq!(observed.range_policy, automatic.range_policy);
        assert_eq!(observed.float_underflow, automatic.float_underflow);
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

    #[cfg(feature = "tensor")]
    #[test]
    fn common_arithmetic_fault_view_preserves_original_graph_context() {
        use crate::dialect::tensor::Graph;
        use crate::dialect::tensor::TensorArithmeticStep;
        use crate::dialect::tensor::TensorError;
        let mut graph = Graph::try_new().unwrap();
        let value = graph.input_typed::<u32>([5]).unwrap().erase();
        let fault = crate::PcuExecutionFault {
            invocation_id: 2,
            kind: crate::PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: false,
        };
        let elemental = PcuExecutionError::TensorBuild(TensorError::ArithmeticFault {
            value,
            element_index: 2,
            kind: fault.kind,
        });
        assert_eq!(elemental.arithmetic_fault(), Some(fault));
        assert!(
            matches!(elemental, PcuExecutionError::TensorBuild(TensorError::ArithmeticFault {
            value: original, ..
        }) if original == value)
        );
        let compound = PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
            value,
            element_index: 2,
            reduction_index: 7,
            step: TensorArithmeticStep::Multiply,
            kind: fault.kind,
        });
        assert_eq!(compound.arithmetic_fault(), Some(fault));
        assert!(
            matches!(compound, PcuExecutionError::TensorBuild(TensorError::CompoundArithmeticFault {
            value: original, reduction_index: 7, step: TensorArithmeticStep::Multiply, ..
        }) if original == value)
        );
        let recovered = crate::PcuExecutionFault {
            recovered: true,
            ..fault
        };
        assert_eq!(
            PcuExecutionError::ArithmeticFault(recovered).arithmetic_fault(),
            Some(recovered)
        );
        assert_eq!(PcuExecutionError::NoBackendEnabled.arithmetic_fault(), None);
    }

    #[cfg(all(feature = "tensor", any(feature = "rocm", feature = "cuda")))]
    #[test]
    fn backend_wrapped_fault_views_preserve_graph_context_and_recovery() {
        #[rustfmt::skip]
        use crate::dialect::tensor::{
            Graph,
            TensorArithmeticStep,
            TensorError,
        };
        let mut graph = Graph::try_new().unwrap();
        let value = graph.input_typed::<u32>([5]).unwrap().erase();
        let fault = crate::PcuExecutionFault {
            invocation_id: 2,
            kind: crate::PcuExecutionFaultKind::ArithmeticUnderflow,
            recovered: false,
        };
        let recovered = crate::PcuExecutionFault {
            recovered: true,
            ..fault
        };
        #[cfg(feature = "rocm")]
        {
            let wrapped = PcuExecutionError::TensorExecution(
                fusion_pcu_rocm::RocmTensorExecutionError::Graph(
                    TensorError::CompoundArithmeticFault {
                        value,
                        element_index: 2,
                        reduction_index: 7,
                        step: TensorArithmeticStep::Multiply,
                        kind: fault.kind,
                    },
                ),
            );
            assert_eq!(wrapped.arithmetic_fault(), Some(fault));
            assert!(
                matches!(wrapped, PcuExecutionError::TensorExecution(fusion_pcu_rocm::RocmTensorExecutionError::Graph(TensorError::CompoundArithmeticFault {
                value: original, reduction_index: 7, step: TensorArithmeticStep::Multiply, ..
            })) if original == value)
            );
            assert_eq!(
                PcuExecutionError::TensorExecution(
                    fusion_pcu_rocm::RocmTensorExecutionError::ExecutionFault(recovered)
                )
                .arithmetic_fault(),
                Some(recovered)
            );
            assert_eq!(
                PcuExecutionError::TensorExecution(
                    fusion_pcu_rocm::RocmTensorExecutionError::FailedCompletion
                )
                .arithmetic_fault(),
                None
            );
        }
        #[cfg(feature = "cuda")]
        {
            let wrapped = PcuExecutionError::CudaTensorExecution(
                fusion_pcu_cuda::CudaTensorExecutionError::Graph(
                    TensorError::CompoundArithmeticFault {
                        value,
                        element_index: 2,
                        reduction_index: 7,
                        step: TensorArithmeticStep::Multiply,
                        kind: fault.kind,
                    },
                ),
            );
            assert_eq!(wrapped.arithmetic_fault(), Some(fault));
            assert!(
                matches!(wrapped, PcuExecutionError::CudaTensorExecution(fusion_pcu_cuda::CudaTensorExecutionError::Graph(TensorError::CompoundArithmeticFault {
                value: original, reduction_index: 7, step: TensorArithmeticStep::Multiply, ..
            })) if original == value)
            );
            assert_eq!(
                PcuExecutionError::CudaTensorExecution(
                    fusion_pcu_cuda::CudaTensorExecutionError::ExecutionFault(recovered)
                )
                .arithmetic_fault(),
                Some(recovered)
            );
            assert_eq!(
                PcuExecutionError::CudaTensorExecution(
                    fusion_pcu_cuda::CudaTensorExecutionError::FailedCompletion
                )
                .arithmetic_fault(),
                None
            );
        }
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
        let device = |id| crate::PcuObjectRef {
            provider: crate::PcuProviderId(7),
            generation: 11,
            kind: crate::PcuObjectKind::Device,
            id,
        };
        let error = PcuExecutionError::NoCompatibleDevice {
            rejected: alloc::vec![
                (
                    device(3),
                    PcuExecutionError::TensorExecution(
                        fusion_pcu_rocm::RocmTensorExecutionError::UnsupportedScalarType(
                            crate::PcuScalarType::U32,
                        ),
                    ),
                ),
                (
                    device(7),
                    PcuExecutionError::BackendInitialization(
                        fusion_pcu_rocm::RocmOwnedDispatchError::InvalidBlockSize,
                    ),
                ),
            ],
            discovery: alloc::vec![],
        };
        assert!(matches!(
            &error,
            PcuExecutionError::NoCompatibleDevice { rejected: rejections, .. }
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

    #[cfg(not(any(
        feature = "rocm",
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "cpu",
        feature = "mlx"
    )))]
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
