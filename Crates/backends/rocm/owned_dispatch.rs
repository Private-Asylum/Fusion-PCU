//! Owned, asynchronous PCU Dispatch adapter for one explicitly selected HIP device.
//!
//! This adapter supports the bounded scalar Dispatch profiles accepted by the `ROCm` lowerer.
//! It requires buffers allocated by this adapter's HIP runtime and retains exclusive buffer
//! leases until the HIP event proves that the kernel has stopped accessing them.

use std::{
    error::Error,
    fmt,
};

const INLINE_ARGUMENTS: usize = 8;
const FAULT_WORD_SENTINEL: u64 = u64::MAX;

fn kernel_uses_checked_div_rem(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    kernel.ops.iter().any(|op| match op {
        PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. }) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => body.iter().any(|body_op| {
            matches!(
                body_op,
                PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem { .. })
            )
        }),
        _ => false,
    })
}

const fn decode_fault_word(word: u64) -> Result<Option<PcuExecutionFault>, HipError> {
    if word == FAULT_WORD_SENTINEL {
        return Ok(None);
    }
    let kind = match word & 0b11 {
        1 => PcuExecutionFaultKind::DivideByZero,
        2 => PcuExecutionFaultKind::SignedDivisionOverflow,
        _ => return Err(HipError::InvalidExecutionFaultWord(word)),
    };
    Ok(Some(PcuExecutionFault {
        kind,
        invocation_id: word >> 2,
    }))
}

use fusion_pcu::{
    PcuBaseContract,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuCompletionState,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuDeviceIdentity,
    PcuDispatchDataOp,
    PcuDispatchFeatureCaps,
    PcuDispatchKernelIr,
    PcuDispatchOpCaps,
    PcuDispatchOp,
    PcuDispatchPolicyCaps,
    PcuDispatchSubmission,
    PcuDispatchSupport,
    PcuExecutorClass,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuExecutorOrigin,
    PcuExecutorSupport,
    PcuFeatureSupport,
    PcuInvocationParameters,
    PcuOwnedBinding,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuPreparedOwnedDispatch,
    PcuOwnedDispatchMemorySession,
    PcuObjectKind,
    PcuObjectRef,
    PcuOwnedDispatchBindingError,
    PcuPrimitiveCaps,
    PcuPrimitiveSupport,
    PcuMemoryAccess,
    PcuMemoryResource,
    PcuSupport,
    PcuValueTypeCaps,
    validate_owned_binding_requirements,
    PcuOwnedBindingRequirement,
};

use crate::{
    DeviceBuffer,
    HipCompletion,
    HipError,
    HipKernelArgument,
    HipRuntime,
    RocmDiscovery,
    RocmLowerError,
    RocmMemoryProvider,
    RocmMemoryResource,
    compile_hip_source,
    lower_dispatch_to_hip_rtc_source,
    lower_dispatch_to_hip_source,
};
use crate::HipCompletionBatch;

/// ROCm-owned dispatch setup or submission failure.
#[derive(Debug)]
pub enum RocmOwnedDispatchError {
    Hip(HipError),
    Lower(RocmLowerError),
    Compile(crate::HipCompileError),
    HipRtc(crate::HipRtcError),
    CompilerUnavailable,
    InvalidDeviceReference,
    InvalidBlockSize,
    RuntimeDeviceMismatch {
        expected: u32,
        actual: u32,
    },
    BufferSizeMismatch {
        binding: PcuBindingRef,
        metadata: u64,
        actual: usize,
    },
    BufferTooSmall {
        binding: PcuBindingRef,
        required: usize,
        available: usize,
    },
    DifferentRuntime(PcuBindingRef),
    MemoryAccessMismatch(PcuBindingRef),
    GeometryOverflow,
    Binding(PcuOwnedDispatchBindingError),
    CheckedDivisionBatchUnsupported,
}

impl fmt::Display for RocmOwnedDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => error.fmt(f),
            Self::Lower(error) => write!(f, "PCU Dispatch cannot lower to ROCm: {error}"),
            Self::Compile(error) => write!(f, "ROCm code object compilation failed: {error}"),
            Self::HipRtc(error) => write!(f, "ROCm runtime compilation failed: {error}"),
            Self::CompilerUnavailable => f.write_str("no usable ROCm source compiler is available"),
            Self::InvalidDeviceReference => f.write_str("invalid ROCm device reference"),
            Self::InvalidBlockSize => f.write_str("HIP block size must be nonzero"),
            Self::RuntimeDeviceMismatch { expected, actual } => write!(
                f,
                "selected ROCm device reference names device {expected}, runtime opened device {actual}"
            ),
            Self::BufferSizeMismatch {
                binding,
                metadata,
                actual,
            } => write!(
                f,
                "binding {binding:?} declares {metadata} bytes but its HIP allocation has {actual}"
            ),
            Self::BufferTooSmall {
                binding,
                required,
                available,
            } => write!(
                f,
                "binding {binding:?} needs {required} bytes but has {available}"
            ),
            Self::DifferentRuntime(binding) => write!(
                f,
                "binding {binding:?} belongs to another HIP runtime or device"
            ),
            Self::MemoryAccessMismatch(binding) => write!(
                f,
                "memory resource for binding {binding:?} does not permit the requested access"
            ),
            Self::GeometryOverflow => f.write_str("HIP launch geometry overflow"),
            Self::Binding(error) => write!(f, "invalid owned PCU binding: {error:?}"),
            Self::CheckedDivisionBatchUnsupported => f.write_str(
                "checked integer division cannot be submitted through the ordered HIP batch path",
            ),
        }
    }
}

impl Error for RocmOwnedDispatchError {}

impl From<HipError> for RocmOwnedDispatchError {
    fn from(error: HipError) -> Self {
        Self::Hip(error)
    }
}

impl From<RocmLowerError> for RocmOwnedDispatchError {
    fn from(error: RocmLowerError) -> Self {
        Self::Lower(error)
    }
}

impl From<crate::HipCompileError> for RocmOwnedDispatchError {
    fn from(error: crate::HipCompileError) -> Self {
        Self::Compile(error)
    }
}

/// Opened owned-dispatch session tied to one generation-bound discovery reference.
pub struct RocmOwnedDispatchBackend {
    runtime: HipRuntime,
    device: PcuDeviceIdentity,
    architecture: Option<String>,
    compiler: Option<crate::discovery::DispatchCompiler>,
    block_size: u32,
}

impl RocmOwnedDispatchBackend {
    #[cfg(feature = "tensor")]
    pub(crate) const fn tensor_runtime(&self) -> &HipRuntime {
        &self.runtime
    }

    #[cfg(feature = "tensor")]
    pub(crate) fn compile_tensor_source(
        &self,
        source: &str,
    ) -> Result<Vec<u8>, RocmOwnedDispatchError> {
        match self.compiler {
            Some(crate::discovery::DispatchCompiler::Hipcc) => {
                let architecture = self
                    .architecture
                    .as_deref()
                    .ok_or(HipError::MissingArchitecture)?;
                let hipcc_source = format!("#include <hip/hip_runtime.h>\n{source}");
                compile_hip_source(&hipcc_source, architecture).map_err(Into::into)
            }
            Some(crate::discovery::DispatchCompiler::HipRtc) => {
                crate::compile_hip_source_for_device(&self.runtime, source)
                    .map_err(RocmOwnedDispatchError::HipRtc)
            }
            None => Err(RocmOwnedDispatchError::CompilerUnavailable),
        }
    }

    /// Validate and open one discovered device for owned asynchronous Dispatch.
    ///
    /// The code-generation target is detected from the selected device's PCI identity and KFD
    /// topology when available. Library-backed tensor work can use this session without it;
    /// Dispatch preparation uses HIPRTC when an architecture or `hipcc` is unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error when the device reference, block size, or runtime identity is invalid, or
    /// when HIP cannot reopen the selected device.
    pub fn open(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        block_size: u32,
    ) -> Result<Self, RocmOwnedDispatchError> {
        if block_size == 0 {
            return Err(RocmOwnedDispatchError::InvalidBlockSize);
        }
        if device.kind != PcuObjectKind::Device {
            return Err(RocmOwnedDispatchError::InvalidDeviceReference);
        }
        let identity = PcuDeviceIdentity::from_device_ref(device)
            .ok_or(RocmOwnedDispatchError::InvalidDeviceReference)?;
        let compiler = discovery.dispatch_compiler(device).ok();
        let runtime = discovery.open_device(device)?;
        let runtime_info = runtime.device_info()?;
        let actual = u32::try_from(runtime_info.index)
            .map_err(|_| RocmOwnedDispatchError::InvalidDeviceReference)?;
        if actual != identity.device_id() {
            return Err(RocmOwnedDispatchError::RuntimeDeviceMismatch {
                expected: identity.device_id(),
                actual,
            });
        }
        let architecture = runtime_info.architecture;
        Ok(Self {
            runtime,
            device: identity,
            architecture,
            compiler,
            block_size,
        })
    }

    /// Allocate a buffer in the exact HIP runtime session accepted by this adapter.
    ///
    /// # Errors
    ///
    /// Returns the HIP allocation error when the device cannot allocate the requested size.
    pub fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, HipError> {
        self.runtime.allocate(bytes)
    }

    /// Create a memory provider for this same opened HIP device and its caller-assigned pool.
    #[must_use]
    pub fn memory_provider(&self, pool: fusion_pcu::PcuMemoryPoolId) -> RocmMemoryProvider {
        self.runtime.memory_provider(pool)
    }

    /// Create binding metadata from this adapter's device identity and the allocation's true size.
    ///
    /// # Errors
    ///
    /// Returns an error when the allocation belongs to another runtime or device.
    pub fn binding(
        &self,
        target: PcuBindingRef,
        access: PcuBindingAccess,
        binding_type: PcuBindingType,
        resource: DeviceBuffer,
    ) -> Result<PcuOwnedBinding<DeviceBuffer>, RocmOwnedDispatchError> {
        self.runtime
            .ensure_same_runtime(&resource.allocation.runtime)
            .map_err(|_| RocmOwnedDispatchError::DifferentRuntime(target))?;
        Ok(PcuOwnedBinding::new(
            target,
            self.device,
            resource.len() as u64,
            access,
            binding_type,
            resource,
        ))
    }

    /// Lower, compile, load, and resolve one Dispatch kernel for repeated submissions.
    ///
    /// The returned reusable executable captures this session's generation-bound device
    /// identity and HIP runtime. It may outlive this backend value, but every submission remains
    /// tied to that same runtime/device. Keep it alive across warm launches to avoid repeating
    /// source lowering, code-object compilation, module loading, function lookup, stream creation,
    /// and ABI binding-order construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel profile cannot be lowered or HIP cannot compile/load it.
    pub fn prepare_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        self.prepare_dispatch_ir(*submission.kernel, submission.shape)
    }

    /// Prepares a tensor-owned kernel on the tensor assessor's selected ordered stream.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_dispatch_owned_kernel_on_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: &crate::HipStreamHandle,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(RocmOwnedDispatchError::Hip(HipError::DifferentRuntime));
        }
        self.prepare_dispatch_ir_with_stream(kernel, shape, stream.clone())
    }

    /// Prepares a dynamically generated tensor kernel with HIPRTC as the first compiler choice.
    ///
    /// HIPRTC compiles against the selected runtime device and avoids launching a `hipcc` process
    /// for short-lived generated tensor programs. When discovery selected `hipcc` as available,
    /// a HIPRTC failure falls back to that compiler. This preference is deliberately local to the
    /// tensor-generated executable and does not change general Dispatch compiler selection.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_dynamic_tensor_kernel_on_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: &crate::HipStreamHandle,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(RocmOwnedDispatchError::Hip(HipError::DifferentRuntime));
        }
        self.prepare_dispatch_ir_with_stream_preference(kernel, shape, stream.clone(), true)
    }

    fn prepare_dispatch_ir(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        let stream = self.runtime.create_stream()?;
        self.prepare_dispatch_ir_with_stream(kernel, shape, stream)
    }

    fn prepare_dispatch_ir_with_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: crate::HipStreamHandle,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        self.prepare_dispatch_ir_with_stream_preference(kernel, shape, stream, false)
    }

    fn prepare_dispatch_ir_with_stream_preference(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: crate::HipStreamHandle,
        prefer_hiprtc: bool,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        let source = lower_dispatch_to_hip_source(&kernel)?;
        let checked_division = kernel_uses_checked_div_rem(&kernel);
        let logical_invocations = shape.invocation_count().get();
        if kernel.entry.logical_shape != [logical_invocations, 1, 1] {
            return Err(RocmOwnedDispatchError::Lower(
                RocmLowerError::InvalidKernelShape,
            ));
        }
        let grid_x = launch_grid(logical_invocations, self.block_size)?;
        let compiler = self
            .compiler
            .ok_or(RocmOwnedDispatchError::CompilerUnavailable)?;
        let image = if prefer_hiprtc {
            let rtc_image = lower_dispatch_to_hip_rtc_source(&kernel)
                .map_err(RocmOwnedDispatchError::Lower)
                .and_then(|rtc_source| {
                    crate::compile_hip_source_for_device(&self.runtime, &rtc_source)
                        .map_err(RocmOwnedDispatchError::HipRtc)
                });
            match rtc_image {
                Ok(image) => image,
                Err(_rtc_error) if compiler == crate::discovery::DispatchCompiler::Hipcc => {
                    let architecture = self
                        .architecture
                        .as_deref()
                        .ok_or(HipError::MissingArchitecture)?;
                    compile_hip_source(&source, architecture)?
                }
                Err(error) => return Err(error),
            }
        } else {
            match compiler {
                crate::discovery::DispatchCompiler::Hipcc => {
                    let architecture = self
                        .architecture
                        .as_deref()
                        .ok_or(HipError::MissingArchitecture)?;
                    compile_hip_source(&source, architecture)?
                }
                crate::discovery::DispatchCompiler::HipRtc => {
                    let rtc_source = crate::lower_dispatch_to_hip_rtc_source(&kernel)?;
                    crate::compile_hip_source_for_device(&self.runtime, &rtc_source)
                        .map_err(RocmOwnedDispatchError::HipRtc)?
                }
            }
        };
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_targets = kernel
            .bindings
            .iter()
            .map(|binding| PcuBindingRef::new(binding.set, binding.binding))
            .collect();
        let binding_requirements = kernel
            .bindings
            .iter()
            .map(|binding| {
                PcuOwnedBindingRequirement::from_verified_binding(
                    &kernel,
                    PcuBindingRef::new(binding.set, binding.binding),
                    shape,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(RocmOwnedDispatchError::Binding)?;
        Ok(RocmPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream,
            binding_targets,
            checked_division,
        })
    }
}

/// Reusable compiled `ROCm` executable for one PCU Dispatch descriptor.
///
/// This is distinct from the core `PcuPreparedDispatch` assessment value: it owns the compiled
/// HIP function, owned binding requirements, and a reusable stream needed to submit the same kernel
/// repeatedly. The source IR may be dropped after preparation.
pub struct RocmPreparedDispatch {
    runtime: HipRuntime,
    device: PcuDeviceIdentity,
    binding_requirements: Vec<PcuOwnedBindingRequirement>,
    shape: fusion_pcu::PcuInvocationShape,
    grid_x: u32,
    block_size: u32,
    function: crate::HipKernel,
    stream: crate::HipStreamHandle,
    binding_targets: Vec<PcuBindingRef>,
    checked_division: bool,
}

impl RocmPreparedDispatch {
    /// Submit this executable with a fresh set of owned bindings.
    ///
    /// Binding metadata, allocation size, and captured runtime/device identity are checked on
    /// every call. A stale or unavailable device is reported through HIP operation failures; this
    /// warm path does not query a fresh device snapshot. HIP launch argument storage, access
    /// leases, and completion events remain per launch.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bindings, a mismatched runtime/device identity, or HIP launch
    /// failure. A HIP error after enqueue is conservatively handled by the HIP launch contract.
    pub fn submit(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<RocmOwnedCompletion, RocmOwnedDispatchError> {
        validate_owned_binding_requirements(&self.binding_requirements, self.device, bindings)
            .map_err(RocmOwnedDispatchError::Binding)?;
        for binding in bindings {
            let actual = binding.resource.len();
            if binding.byte_len != actual as u64 {
                return Err(RocmOwnedDispatchError::BufferSizeMismatch {
                    binding: binding.target,
                    metadata: binding.byte_len,
                    actual,
                });
            }
            self.runtime
                .ensure_same_runtime(&binding.resource.allocation.runtime)
                .map_err(|_| RocmOwnedDispatchError::DifferentRuntime(binding.target))?;
        }
        // Kernel arguments are borrowed only during `launch`; HIP copies their pointer values
        // into owned aligned storage before returning. Keep common small interfaces on the stack
        // without constraining larger kernels to an arbitrary binding-count limit.
        let mut fault_word = if self.checked_division {
            let mut buffer = self.runtime.allocate(core::mem::size_of::<u64>())?;
            buffer.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
            Some(buffer)
        } else {
            None
        };
        let argument_count = self.binding_targets.len() + usize::from(fault_word.is_some());
        let mut inline_arguments: [HipKernelArgument<'_>; INLINE_ARGUMENTS] =
            std::array::from_fn(|_| HipKernelArgument::Bytes(&[]));
        let mut overflow_arguments = Vec::new();
        let arguments: &[HipKernelArgument<'_>] = if argument_count <= INLINE_ARGUMENTS {
            for (slot, target) in inline_arguments
                .iter_mut()
                .zip(self.binding_targets.iter().copied())
            {
                let binding =
                    find_binding(target, bindings).map_err(RocmOwnedDispatchError::Binding)?;
                *slot = HipKernelArgument::Buffer(&binding.resource);
            }
            if let Some(buffer) = fault_word.as_ref() {
                inline_arguments[self.binding_targets.len()] = HipKernelArgument::Buffer(buffer);
            }
            &inline_arguments[..argument_count]
        } else {
            overflow_arguments.reserve(argument_count);
            for target in self.binding_targets.iter().copied() {
                let binding =
                    find_binding(target, bindings).map_err(RocmOwnedDispatchError::Binding)?;
                overflow_arguments.push(HipKernelArgument::Buffer(&binding.resource));
            }
            if let Some(buffer) = fault_word.as_ref() {
                overflow_arguments.push(HipKernelArgument::Buffer(buffer));
            }
            &overflow_arguments
        };

        // SAFETY: lowering validates one typed scalar pointer per declared binding in
        // declaration order (f32 map or u32 identity-copy profile). Core admission validates full binding coverage,
        // device metadata, access, and type. This adapter additionally verifies actual allocation
        // length and HIP runtime identity, and HipKernel::launch acquires the shared exclusive
        // allocation gates and retains module, stream, and allocations through event completion.
        let hip = unsafe {
            self.function.launch(
                &self.stream,
                [self.grid_x, 1, 1],
                [self.block_size, 1, 1],
                0,
                arguments,
            )?
        };
        Ok(RocmOwnedCompletion {
            hip: Some(hip),
            fault_word: fault_word.take(),
            terminal: None,
        })
    }

    /// Submit this executable directly into an ordered HIP completion batch.
    ///
    /// This path avoids creating a per-launch HIP event. The batch must use the stream captured
    /// by this prepared dispatch and must be finished after the final queued operation. If this
    /// method returns a HIP launch error, the batch is poisoned and must be dropped; its drop
    /// path synchronizes the stream or quarantines its retained resources.
    ///
    /// # Errors
    /// Returns an error for invalid bindings, a mismatched runtime/device/stream, a poisoned
    /// batch, or HIP launch failure.
    pub fn submit_into_batch(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        batch: &mut HipCompletionBatch,
    ) -> Result<(), RocmOwnedDispatchError> {
        if self.checked_division {
            return Err(RocmOwnedDispatchError::CheckedDivisionBatchUnsupported);
        }
        validate_owned_binding_requirements(&self.binding_requirements, self.device, bindings)
            .map_err(RocmOwnedDispatchError::Binding)?;
        for binding in bindings {
            let actual = binding.resource.len();
            if binding.byte_len != actual as u64 {
                return Err(RocmOwnedDispatchError::BufferSizeMismatch {
                    binding: binding.target,
                    metadata: binding.byte_len,
                    actual,
                });
            }
            self.runtime
                .ensure_same_runtime(&binding.resource.allocation.runtime)
                .map_err(|_| RocmOwnedDispatchError::DifferentRuntime(binding.target))?;
        }
        let mut inline_arguments: [HipKernelArgument<'_>; INLINE_ARGUMENTS] =
            std::array::from_fn(|_| HipKernelArgument::Bytes(&[]));
        let mut overflow_arguments = Vec::new();
        let arguments: &[HipKernelArgument<'_>] = if self.binding_targets.len() <= INLINE_ARGUMENTS
        {
            for (slot, target) in inline_arguments
                .iter_mut()
                .zip(self.binding_targets.iter().copied())
            {
                let binding =
                    find_binding(target, bindings).map_err(RocmOwnedDispatchError::Binding)?;
                *slot = HipKernelArgument::Buffer(&binding.resource);
            }
            &inline_arguments[..self.binding_targets.len()]
        } else {
            overflow_arguments.reserve(self.binding_targets.len());
            for target in self.binding_targets.iter().copied() {
                let binding =
                    find_binding(target, bindings).map_err(RocmOwnedDispatchError::Binding)?;
                overflow_arguments.push(HipKernelArgument::Buffer(&binding.resource));
            }
            &overflow_arguments
        };

        // SAFETY: the same validated binding ABI and allocation checks as `submit` apply. The
        // launch primitive retains all leases in `batch` before enqueue and poisons it on error.
        unsafe {
            self.function.launch_into_batch(
                batch,
                [self.grid_x, 1, 1],
                [self.block_size, 1, 1],
                0,
                arguments,
            )
        }?;
        Ok(())
    }
}

impl PcuBaseContract for RocmOwnedDispatchBackend {
    fn support(&self) -> PcuSupport {
        owned_dispatch_support()
    }

    fn executors(&self) -> &'static [PcuExecutorDescriptor] {
        &OWNED_EXECUTORS
    }
}

impl PcuOwnedDispatchBackend for RocmOwnedDispatchBackend {
    type Resource = DeviceBuffer;
    type Bindings = Vec<PcuOwnedBinding<DeviceBuffer>>;
    type Completion = RocmOwnedCompletion;
    type Error = RocmOwnedDispatchError;
    type Prepared<'kernel, 'parameters>
        = RocmPreparedDispatch
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
        let prepared = self.prepare_dispatch_owned_direct(submission, parameters)?;
        prepared.submit_owned_direct(bindings)
    }

    fn prepare_dispatch_owned_direct<'kernel, 'parameters>(
        &self,
        submission: PcuDispatchSubmission<'kernel>,
        parameters: PcuInvocationParameters<'parameters>,
    ) -> Result<Self::Prepared<'kernel, 'parameters>, Self::Error> {
        if !parameters.is_empty() {
            return Err(RocmOwnedDispatchError::Lower(
                RocmLowerError::UnsupportedKernelInterface,
            ));
        }
        self.prepare_dispatch(submission)
    }
}

impl PcuPreparedOwnedDispatch for RocmPreparedDispatch {
    type Resource = DeviceBuffer;
    type Bindings = Vec<PcuOwnedBinding<DeviceBuffer>>;
    type Completion = RocmOwnedCompletion;
    type Error = RocmOwnedDispatchError;

    type BindingSchema = [PcuOwnedBindingRequirement];

    fn binding_schema(&self) -> &Self::BindingSchema {
        &self.binding_requirements
    }

    fn shape(&self) -> fusion_pcu::PcuInvocationShape {
        self.shape
    }

    fn device_identity(&self) -> PcuDeviceIdentity {
        self.device
    }

    fn submit_owned_direct(
        &self,
        bindings: Self::Bindings,
    ) -> Result<Self::Completion, Self::Error> {
        self.submit(&bindings)
    }
}

impl PcuOwnedDispatchMemorySession for RocmOwnedDispatchBackend {
    type MemoryProvider = RocmMemoryProvider;

    fn memory_provider(&self, pool: fusion_pcu::PcuMemoryPoolId) -> Self::MemoryProvider {
        Self::memory_provider(self, pool)
    }

    fn bind(
        &self,
        target: PcuBindingRef,
        access: PcuBindingAccess,
        binding_type: PcuBindingType,
        resource: &RocmMemoryResource,
    ) -> Result<PcuOwnedBinding<Self::Resource>, Self::Error> {
        if !memory_access_supports_binding(resource.access(), access) {
            return Err(RocmOwnedDispatchError::MemoryAccessMismatch(target));
        }
        let buffer = resource.device_buffer();
        self.runtime
            .ensure_same_runtime(&buffer.allocation.runtime)
            .map_err(|_| RocmOwnedDispatchError::DifferentRuntime(target))?;
        Ok(PcuOwnedBinding::new(
            target,
            self.device,
            resource.size_bytes(),
            access,
            binding_type,
            buffer.clone(),
        ))
    }
}

fn memory_access_supports_binding(available: PcuMemoryAccess, requested: PcuBindingAccess) -> bool {
    match requested {
        PcuBindingAccess::ReadOnly => matches!(
            available,
            PcuMemoryAccess::ReadOnly | PcuMemoryAccess::ReadWrite
        ),
        PcuBindingAccess::WriteOnly => matches!(
            available,
            PcuMemoryAccess::WriteOnly | PcuMemoryAccess::ReadWrite
        ),
        PcuBindingAccess::ReadWrite => available == PcuMemoryAccess::ReadWrite,
    }
}

/// Core completion contract over the HIP event-backed completion token.
pub struct RocmOwnedCompletion {
    hip: Option<HipCompletion>,
    fault_word: Option<DeviceBuffer>,
    terminal: Option<PcuCompletionOutcome>,
}

impl PcuOwnedCompletion for RocmOwnedCompletion {
    type Error = HipError;

    fn state(&self) -> Result<PcuCompletionState, Self::Error> {
        Ok(match self.terminal {
            Some(PcuCompletionOutcome::Succeeded) => PcuCompletionState::Succeeded,
            Some(PcuCompletionOutcome::Failed | PcuCompletionOutcome::Fault(_)) => {
                PcuCompletionState::Failed
            }
            None => PcuCompletionState::Running,
        })
    }

    fn wait(&mut self) -> Result<PcuCompletionOutcome, Self::Error> {
        if let Some(terminal) = self.terminal {
            return Ok(terminal);
        }
        let hip = self
            .hip
            .as_mut()
            .expect("nonterminal completion retains HIP token");
        hip.wait()?;
        let outcome = if let Some(fault_word) = self.fault_word.as_ref() {
            let mut bytes = [0_u8; core::mem::size_of::<u64>()];
            fault_word.copy_to(&mut bytes)?;
            let word = u64::from_le_bytes(bytes);
            decode_fault_word(word)?
                .map_or(PcuCompletionOutcome::Succeeded, PcuCompletionOutcome::Fault)
        } else {
            PcuCompletionOutcome::Succeeded
        };
        self.hip.take();
        self.fault_word.take();
        self.terminal = Some(outcome);
        Ok(outcome)
    }
}

fn launch_grid(invocations: u32, block_size: u32) -> Result<u32, RocmOwnedDispatchError> {
    if block_size == 0 {
        return Err(RocmOwnedDispatchError::InvalidBlockSize);
    }
    let grid = u64::from(invocations).div_ceil(u64::from(block_size));
    if grid * u64::from(block_size) > u64::from(u32::MAX) {
        return Err(RocmOwnedDispatchError::GeometryOverflow);
    }
    u32::try_from(grid).map_err(|_| RocmOwnedDispatchError::GeometryOverflow)
}

fn find_binding<R>(
    target: PcuBindingRef,
    provided: &[PcuOwnedBinding<R>],
) -> Result<&PcuOwnedBinding<R>, PcuOwnedDispatchBindingError> {
    provided
        .iter()
        .find(|binding| binding.target == target)
        .ok_or(PcuOwnedDispatchBindingError::Missing(target))
}

const fn owned_dispatch_support() -> PcuSupport {
    let mut support = PcuSupport::unsupported();
    support.caps = fusion_pcu::PcuCaps::ENUMERATE_EXECUTORS
        .union(fusion_pcu::PcuCaps::DISPATCH)
        .union(fusion_pcu::PcuCaps::COMPUTE_DISPATCH);
    support.implementation = fusion_pcu::PcuImplementationKind::Native;
    support.executor_count = 1;
    support.primitive_support = PcuPrimitiveSupport {
        primitives: PcuFeatureSupport::new(PcuPrimitiveCaps::DISPATCH, PcuPrimitiveCaps::empty()),
    };
    support.value_type_support = PcuFeatureSupport::new(
        PcuValueTypeCaps::FLOAT32
            .union(PcuValueTypeCaps::FLOAT64)
            .union(PcuValueTypeCaps::INT8)
            .union(PcuValueTypeCaps::UINT8)
            .union(PcuValueTypeCaps::UINT16)
            .union(PcuValueTypeCaps::UINT32)
            .union(PcuValueTypeCaps::INT32)
            .union(PcuValueTypeCaps::INT16)
            .union(PcuValueTypeCaps::UINT64)
            .union(PcuValueTypeCaps::INT64)
            .union(PcuValueTypeCaps::SCALAR_VALUES),
        PcuValueTypeCaps::empty(),
    );
    let mut dispatch = PcuDispatchSupport::unsupported();
    dispatch.flags = PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION);
    dispatch.instructions = PcuFeatureSupport::new(
        PcuDispatchOpCaps::VALUE_CONSTANT
            .union(PcuDispatchOpCaps::ALU_ADD)
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL)
            .union(PcuDispatchOpCaps::ALU_DIV)
            .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
            .union(PcuDispatchOpCaps::CONTROL_RETURN)
            .union(PcuDispatchOpCaps::CONTROL_LOOP)
            .union(PcuDispatchOpCaps::BINDING_LOAD)
            .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
            .union(PcuDispatchOpCaps::BINDING_STORE),
        PcuDispatchOpCaps::empty(),
    );
    dispatch.scalar_alu = PcuFeatureSupport::new(
        fusion_pcu::PcuDispatchScalarAluSupport::empty()
            .with(fusion_pcu::PcuScalarType::F32, f32_alu_caps())
            .with(fusion_pcu::PcuScalarType::F64, f32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U16, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I16, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::U8, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I8, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
            .with(fusion_pcu::PcuScalarType::I64, int_alu_caps()),
        fusion_pcu::PcuDispatchScalarAluSupport::empty(),
    );
    dispatch.features = PcuFeatureSupport::new(
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES),
        PcuDispatchFeatureCaps::empty(),
    );
    support.dispatch_support = dispatch;
    support
}

const OWNED_DISPATCH_INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::VALUE_CONSTANT
    .union(PcuDispatchOpCaps::ALU_ADD)
    .union(PcuDispatchOpCaps::ALU_SUB)
    .union(PcuDispatchOpCaps::ALU_MUL)
    .union(PcuDispatchOpCaps::ALU_DIV)
    .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
    .union(PcuDispatchOpCaps::BINDING_STORE);

const fn f32_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_ADD
        .union(PcuDispatchOpCaps::ALU_SUB)
        .union(PcuDispatchOpCaps::ALU_MUL)
        .union(PcuDispatchOpCaps::ALU_DIV)
}

const fn int_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_ADD
        .union(PcuDispatchOpCaps::ALU_SUB)
        .union(PcuDispatchOpCaps::ALU_MUL)
}

const fn checked_u32_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps().union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
}

const fn checked_u64_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps().union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
}

const fn checked_i32_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps().union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
}

const OWNED_EXECUTORS: [PcuExecutorDescriptor; 1] = [PcuExecutorDescriptor {
    id: PcuExecutorId(0),
    name: "rocm-owned-dispatch",
    class: PcuExecutorClass::Compute,
    origin: PcuExecutorOrigin::TopologyBound,
    support: PcuExecutorSupport {
        primitives: PcuPrimitiveCaps::DISPATCH,
        dispatch_policy: PcuDispatchPolicyCaps::SERIAL
            .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION),
        value_types: PcuValueTypeCaps::FLOAT32
            .union(PcuValueTypeCaps::FLOAT64)
            .union(PcuValueTypeCaps::INT8)
            .union(PcuValueTypeCaps::UINT8)
            .union(PcuValueTypeCaps::UINT16)
            .union(PcuValueTypeCaps::UINT32)
            .union(PcuValueTypeCaps::INT32)
            .union(PcuValueTypeCaps::INT16)
            .union(PcuValueTypeCaps::UINT64)
            .union(PcuValueTypeCaps::INT64)
            .union(PcuValueTypeCaps::SCALAR_VALUES),
        dispatch_instructions: OWNED_DISPATCH_INSTRUCTIONS,
        dispatch_scalar_alu: fusion_pcu::PcuDispatchScalarAluSupport::empty()
            .with(fusion_pcu::PcuScalarType::F32, f32_alu_caps())
            .with(fusion_pcu::PcuScalarType::F64, f32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U16, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I16, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::U8, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I8, int_alu_caps())
            .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
            .with(fusion_pcu::PcuScalarType::I64, int_alu_caps()),
        dispatch_features: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES),
        stream_instructions: fusion_pcu::PcuStreamCapabilities::empty(),
        command_instructions: fusion_pcu::PcuCommandOpCaps::empty(),
        transaction_features: fusion_pcu::PcuTransactionFeatureCaps::empty(),
        signal_instructions: fusion_pcu::PcuSignalOpCaps::empty(),
    },
}];

#[cfg(test)]
mod tests {
    use super::{
        decode_fault_word,
        FAULT_WORD_SENTINEL,
        RocmOwnedDispatchError,
        OWNED_DISPATCH_INSTRUCTIONS,
        OWNED_EXECUTORS,
        owned_dispatch_support,
        find_binding,
        memory_access_supports_binding,
        launch_grid,
    };
    use fusion_pcu::{
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingType,
        PcuMemoryAccess,
        PcuDeviceIdentity,
        PcuDispatchOpCaps,
        PcuObjectKind,
        PcuObjectRef,
        PcuOwnedBinding,
        PcuProviderId,
        PcuValueType,
        PcuExecutionFault,
        PcuExecutionFaultKind,
    };

    struct Noop;

    #[test]
    fn checked_division_fault_word_decodes_sentinel_and_logical_invocation() {
        assert_eq!(decode_fault_word(FAULT_WORD_SENTINEL), Ok(None));
        assert_eq!(
            decode_fault_word((37_u64 << 2) | 1),
            Ok(Some(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: 37,
            }))
        );
        assert_eq!(
            decode_fault_word((37_u64 << 2) | 2),
            Ok(Some(PcuExecutionFault {
                kind: PcuExecutionFaultKind::SignedDivisionOverflow,
                invocation_id: 37,
            }))
        );
        assert!(decode_fault_word((37_u64 << 2) | 3).is_err());
    }

    #[test]
    fn launch_geometry_covers_non_multiple_shapes_without_id_wrap() {
        assert_eq!(launch_grid(65, 64).unwrap(), 2);
        assert_eq!(launch_grid(1, 1).unwrap(), 1);
        assert!(matches!(
            launch_grid(u32::MAX, 4),
            Err(RocmOwnedDispatchError::GeometryOverflow)
        ));
        assert!(matches!(
            launch_grid(10, 0),
            Err(RocmOwnedDispatchError::InvalidBlockSize)
        ));
    }

    #[test]
    fn owned_dispatch_advertises_element_zero_load_support() {
        let required =
            PcuDispatchOpCaps::BINDING_LOAD.union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO);
        assert!(
            owned_dispatch_support()
                .dispatch_support
                .instructions
                .direct
                .contains(required)
        );
        assert!(OWNED_DISPATCH_INSTRUCTIONS.contains(required));
        assert!(!PcuDispatchOpCaps::BINDING_LOAD.contains(required));
    }

    #[test]
    fn owned_dispatch_advertises_f64_type_with_its_alu_operations() {
        let required = fusion_pcu::PcuValueTypeCaps::FLOAT64
            .union(fusion_pcu::PcuValueTypeCaps::SCALAR_VALUES);
        assert!(
            owned_dispatch_support()
                .value_type_support
                .direct
                .contains(required)
        );
        assert!(OWNED_EXECUTORS[0].support.value_types.contains(required));
        assert!(
            owned_dispatch_support()
                .dispatch_support
                .scalar_alu
                .direct
                .for_scalar(fusion_pcu::PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_ADD)
        );
    }

    #[test]
    fn owned_dispatch_advertises_only_supported_u32_alu_operations() {
        let supported = PcuDispatchOpCaps::ALU_ADD
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL);
        let support = owned_dispatch_support()
            .dispatch_support
            .scalar_alu
            .direct
            .for_scalar(fusion_pcu::PcuScalarType::U32);
        assert!(support.contains(supported));
        assert!(!support.contains(PcuDispatchOpCaps::ALU_DIV));
        assert!(support.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
        assert!(
            owned_dispatch_support()
                .dispatch_support
                .instructions
                .direct
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
        assert!(OWNED_DISPATCH_INSTRUCTIONS.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
        assert!(
            OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::U32)
                .contains(supported)
        );
        assert!(
            OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::U32)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
        assert!(
            !OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::U32)
                .contains(PcuDispatchOpCaps::ALU_DIV)
        );
        let u64_support = OWNED_EXECUTORS[0]
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::U64);
        assert!(u64_support.contains(supported));
        assert!(u64_support.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
        assert!(!u64_support.contains(PcuDispatchOpCaps::ALU_DIV));
        let u16_support = OWNED_EXECUTORS[0]
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::U16);
        assert!(u16_support.contains(supported));
        assert!(!u16_support.contains(PcuDispatchOpCaps::ALU_DIV));
        let u8_support = OWNED_EXECUTORS[0]
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::U8);
        assert!(u8_support.contains(supported));
        assert!(!u8_support.contains(PcuDispatchOpCaps::ALU_DIV));
        let i16_support = OWNED_EXECUTORS[0]
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::I16);
        assert!(i16_support.contains(supported));
        assert!(!i16_support.contains(PcuDispatchOpCaps::ALU_DIV));
        let i8_support = OWNED_EXECUTORS[0]
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::I8);
        assert!(i8_support.contains(supported));
        assert!(!i8_support.contains(PcuDispatchOpCaps::ALU_DIV));
        let i32_support = OWNED_EXECUTORS[0]
            .support
            .dispatch_scalar_alu
            .for_scalar(fusion_pcu::PcuScalarType::I32);
        assert!(i32_support.contains(supported));
        assert!(i32_support.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
        assert!(!i32_support.contains(PcuDispatchOpCaps::ALU_DIV));
        assert!(
            !OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::I64)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
    }

    #[test]
    fn owned_buffers_are_reordered_to_match_kernel_abi_declaration_order() {
        let identity = PcuDeviceIdentity::from_device_ref(PcuObjectRef {
            provider: PcuProviderId(7),
            generation: 3,
            kind: PcuObjectKind::Device,
            id: 0,
        })
        .unwrap();
        let first = PcuBindingRef::new(0, 2);
        let second = PcuBindingRef::new(0, 5);
        let provided = [first, second].map(|target| {
            PcuOwnedBinding::new(
                target,
                identity,
                4,
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                Noop,
            )
        });

        assert!(std::ptr::eq(
            std::ptr::from_ref(find_binding(second, &provided).unwrap()),
            std::ptr::from_ref(&provided[1])
        ));
        assert!(std::ptr::eq(
            std::ptr::from_ref(find_binding(first, &provided).unwrap()),
            std::ptr::from_ref(&provided[0])
        ));
    }

    #[test]
    fn memory_access_must_cover_dispatch_binding_access() {
        use PcuBindingAccess::{
            ReadOnly,
            ReadWrite,
            WriteOnly,
        };
        use PcuMemoryAccess::{
            ReadOnly as MemoryReadOnly,
            ReadWrite as MemoryReadWrite,
            WriteOnly as MemoryWriteOnly,
        };

        assert!(memory_access_supports_binding(MemoryReadOnly, ReadOnly));
        assert!(!memory_access_supports_binding(MemoryReadOnly, WriteOnly));
        assert!(!memory_access_supports_binding(MemoryReadOnly, ReadWrite));
        assert!(memory_access_supports_binding(MemoryWriteOnly, WriteOnly));
        assert!(!memory_access_supports_binding(MemoryWriteOnly, ReadOnly));
        assert!(memory_access_supports_binding(MemoryReadWrite, ReadOnly));
        assert!(memory_access_supports_binding(MemoryReadWrite, WriteOnly));
        assert!(memory_access_supports_binding(MemoryReadWrite, ReadWrite));
    }
}
