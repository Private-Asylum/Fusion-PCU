//! Owned, asynchronous PCU Dispatch adapter for one explicitly selected HIP device.
//!
//! This adapter supports the bounded scalar Dispatch profiles accepted by the `ROCm` lowerer.
//! It requires buffers allocated by this adapter's HIP runtime and retains exclusive buffer
//! leases until the HIP event proves that the kernel has stopped accessing them.

#[rustfmt::skip]
use std::{
    error::Error,
    fmt,
};

const INLINE_ARGUMENTS: usize = 8;
const FAULT_WORD_SENTINEL: u64 = u64::MAX;
const RECOVERED_FAULT_BIT: u64 = 1 << 63;

const fn validate_batch_fault_semantics(
    checked_arithmetic: bool,
) -> Result<(), RocmOwnedDispatchError> {
    if checked_arithmetic {
        // Checked arithmetic publishes a terminal fault only after its completion token is waited.
        // A batch can enqueue dependent work before that observation, so accepting the launch
        // would let consumers read invalid arithmetic results.
        return Err(RocmOwnedDispatchError::CheckedArithmeticBatchUnsupported);
    }
    Ok(())
}

fn kernel_uses_checked_arithmetic(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    ops_use_checked_arithmetic(kernel.ops)
}

fn validate_dispatch_requirements(
    kernel: &PcuDispatchKernelIr<'_>,
    support: &PcuSupport,
) -> Result<(), RocmOwnedDispatchError> {
    if support.supports_kernel_direct(fusion_pcu::PcuKernel::Dispatch(*kernel)) {
        Ok(())
    } else {
        Err(RocmOwnedDispatchError::UnsupportedRequirements)
    }
}

fn ops_use_checked_arithmetic(ops: &[PcuDispatchOp<'_>]) -> bool {
    ops.iter().any(|op| match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::CheckedDivRem { .. }
            | PcuDispatchDataOp::CheckedIntegerBinary { .. }
            | PcuDispatchDataOp::CheckedFloatBinary { .. }
            | PcuDispatchDataOp::CheckedFloatUnary { .. }
            | PcuDispatchDataOp::CheckedFloatConvert { .. },
        ) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => ops_use_checked_arithmetic(body),
        _ => false,
    })
}

// ROCm lowering bounds direct invocation shapes and grid-stride extents to u32, leaving
// three tag bits for the logical index. Recovered range faults set the high bit; fatal
// faults leave it clear, so `atomicMin` always gives fatal faults priority. Within either
// class, the smallest logical index wins, and clamp-mode code records only the first
// recoverable range fault per invocation.
const fn decode_fault_word(word: u64) -> Result<Option<PcuExecutionFault>, HipError> {
    if word == FAULT_WORD_SENTINEL {
        return Ok(None);
    }
    let recovered = word & RECOVERED_FAULT_BIT != 0;
    let payload = word & !RECOVERED_FAULT_BIT;
    let kind = match payload & 0b111 {
        1 => PcuExecutionFaultKind::DivideByZero,
        2 => PcuExecutionFaultKind::SignedDivisionOverflow,
        3 => PcuExecutionFaultKind::ArithmeticOverflow,
        4 => PcuExecutionFaultKind::ArithmeticUnderflow,
        5 => PcuExecutionFaultKind::InvalidFloatingOperand,
        _ => return Err(HipError::InvalidExecutionFaultWord(word)),
    };
    if recovered
        && !matches!(
            kind,
            PcuExecutionFaultKind::ArithmeticOverflow | PcuExecutionFaultKind::ArithmeticUnderflow
        )
    {
        return Err(HipError::InvalidExecutionFaultWord(word));
    }
    Ok(Some(PcuExecutionFault {
        kind,
        invocation_id: payload >> 3,
        recovered,
    }))
}

#[rustfmt::skip]
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

#[rustfmt::skip]
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
    Rocblas(crate::RocblasError),
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
    UnsupportedRequirements,
    CheckedArithmeticBatchUnsupported,
    CheckedArithmeticBatchClosed,
    CheckedArithmeticRequired,
    CheckedBatchUnavailable,
    CheckedSequentialDispatchPoisoned,
    CheckedBatchFaultWordUnavailable,
    CheckedFaultWordSize {
        actual: usize,
    },
}

impl fmt::Display for RocmOwnedDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => error.fmt(f),
            Self::Rocblas(error) => error.fmt(f),
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
            Self::UnsupportedRequirements => f.write_str(
                "ROCm does not advertise the scalar types, instructions, ALU operations, or features required by this dispatch kernel",
            ),
            Self::CheckedArithmeticBatchUnsupported => f.write_str(
                "checked arithmetic cannot be submitted through the ordered HIP batch path",
            ),
            Self::CheckedArithmeticBatchClosed => f.write_str(
                "the checked ROCm batch has already submitted its final checked dispatch",
            ),
            Self::CheckedArithmeticRequired => {
                f.write_str("a checked ROCm batch must end with a checked arithmetic dispatch")
            }
            Self::CheckedBatchUnavailable => {
                f.write_str("the checked ROCm batch no longer has an open HIP batch")
            }
            Self::CheckedSequentialDispatchPoisoned => {
                f.write_str("the checked ROCm sequential dispatch has uncertain in-flight work")
            }
            Self::CheckedBatchFaultWordUnavailable => {
                f.write_str("the checked ROCm completion has no retained fault word")
            }
            Self::CheckedFaultWordSize { actual } => write!(
                f,
                "checked dispatch fault word requires {} bytes, received {actual}",
                core::mem::size_of::<u64>()
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

impl From<crate::RocblasError> for RocmOwnedDispatchError {
    fn from(error: crate::RocblasError) -> Self {
        Self::Rocblas(error)
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
    /// Create an ordered HIP stream on this selected backend runtime and device.
    ///
    /// # Errors
    ///
    /// Returns a HIP stream creation error.
    pub fn create_stream(&self) -> Result<crate::HipStreamHandle, RocmOwnedDispatchError> {
        self.runtime.create_stream().map_err(Into::into)
    }

    /// Create a rocBLAS handle for this backend's selected runtime and device.
    ///
    /// # Errors
    ///
    /// Returns a rocBLAS or HIP initialization error.
    #[cfg(feature = "tensor")]
    pub fn create_rocblas(&self) -> Result<crate::Rocblas, crate::RocblasError> {
        crate::Rocblas::new(&self.runtime)
    }

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
        self.prepare_dispatch_on_stream(kernel, shape, stream)
    }

    /// Prepare a Dispatch executable on a caller-selected stream from this runtime.
    ///
    /// This lets callers build an ordered batch from multiple prepared kernels while preserving
    /// the batch stream identity. The stream must belong to this backend's HIP runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another runtime, or if lowering or compilation
    /// of the kernel fails.
    pub fn prepare_dispatch_on_stream(
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

    /// Prepares only the private, validated ordered `MatMul` factory's buffer/fault ABI.
    /// Arbitrary consumer source cannot enter the safe owned-submission path here.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_strict_matmul_on_stream(
        &self,
        spec: crate::tensor::strict_matmul::StrictMatMulSpec,
        stream: &crate::HipStreamHandle,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(RocmOwnedDispatchError::Hip(HipError::DifferentRuntime));
        }
        let compiler = self
            .compiler
            .ok_or(RocmOwnedDispatchError::CompilerUnavailable)?;
        let source = spec.source();
        let image = match crate::compile_hip_source_for_device(&self.runtime, &source) {
            Ok(image) => image,
            Err(_) if compiler == crate::discovery::DispatchCompiler::Hipcc => {
                self.compile_tensor_source(&source)?
            }
            Err(error) => return Err(RocmOwnedDispatchError::HipRtc(error)),
        };
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_requirements = spec.requirements().to_vec();
        let binding_targets = binding_requirements
            .iter()
            .map(|requirement| requirement.target)
            .collect();
        let shape = spec.shape();
        let grid_x = launch_grid(shape.invocation_count().get(), self.block_size)?;
        Ok(RocmPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream: stream.clone(),
            binding_targets,
            checked_arithmetic: true,
        })
    }

    fn prepare_dispatch_ir(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        let validated = self.validate_dispatch_ir(kernel, shape)?;
        let stream = self.runtime.create_stream()?;
        self.prepare_dispatch_ir_after_validation(kernel, shape, stream, false, validated)
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
        let validated = self.validate_dispatch_ir(kernel, shape)?;
        self.prepare_dispatch_ir_after_validation(kernel, shape, stream, prefer_hiprtc, validated)
    }

    fn validate_dispatch_ir(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
    ) -> Result<ValidatedDispatch, RocmOwnedDispatchError> {
        let source = lower_dispatch_to_hip_source(&kernel)?;
        let logical_invocations = shape.invocation_count().get();
        if kernel.entry.logical_shape != [logical_invocations, 1, 1] {
            return Err(RocmOwnedDispatchError::Lower(
                RocmLowerError::InvalidKernelShape,
            ));
        }
        let grid_x = launch_grid(logical_invocations, self.block_size)?;
        validate_dispatch_requirements(&kernel, &owned_dispatch_support())?;
        Ok(ValidatedDispatch {
            source,
            grid_x,
            checked_arithmetic: kernel_uses_checked_arithmetic(&kernel),
        })
    }

    fn prepare_dispatch_ir_after_validation(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: crate::HipStreamHandle,
        prefer_hiprtc: bool,
        validated: ValidatedDispatch,
    ) -> Result<RocmPreparedDispatch, RocmOwnedDispatchError> {
        let ValidatedDispatch {
            source,
            grid_x,
            checked_arithmetic,
        } = validated;
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
            checked_arithmetic,
        })
    }
}

struct ValidatedDispatch {
    source: String,
    grid_x: u32,
    checked_arithmetic: bool,
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
    checked_arithmetic: bool,
}

impl RocmPreparedDispatch {
    pub(crate) const fn requires_checked_fault_word(&self) -> bool {
        self.checked_arithmetic
    }

    /// Clone the stream captured by this executable for preparing related ordered work.
    #[must_use]
    pub fn stream_handle(&self) -> crate::HipStreamHandle {
        self.stream.clone()
    }

    /// Clone this prepared executable's compiled HIP kernel for low-level manual orchestration.
    ///
    /// The returned kernel keeps its module loaded. Direct launches remain unsafe because callers
    /// must supply the exact ABI and buffer access pattern documented by [`crate::HipKernel`].
    #[must_use]
    pub fn hip_kernel(&self) -> crate::HipKernel {
        self.function.clone()
    }

    /// Return this executable's exact three-dimensional grid and block geometry.
    #[must_use]
    pub const fn launch_geometry(&self) -> ([u32; 3], [u32; 3]) {
        ([self.grid_x, 1, 1], [self.block_size, 1, 1])
    }

    /// Start a checked-terminal batch on this executable's captured stream.
    #[must_use]
    pub fn checked_batch(&self) -> RocmCheckedDispatchBatch {
        RocmCheckedDispatchBatch::new(&self.stream)
    }

    /// Allocate reusable status storage for checked submissions that are waited in sequence.
    ///
    /// This opt-in path requires sequential use. Its returned owner permits one in-flight
    /// submission, and `submit_and_wait` holds an exclusive borrow until HIP completion and status
    /// readback finish. Public [`Self::submit`] keeps independent status storage per call and
    /// supports overlapping submissions. After a terminal success, this owner's observed sentinel
    /// is reused without another host-to-device reset. Faults and known prelaunch rejections
    /// require a reset before the next attempt. An uncertain HIP failure poisons the owner because
    /// the status word may still be in use by the device.
    ///
    /// # Errors
    ///
    /// Returns [`RocmOwnedDispatchError::CheckedArithmeticRequired`] for an unchecked executable,
    /// or a HIP allocation/initialization error.
    pub fn sequential_checked(
        &self,
    ) -> Result<RocmSequentialCheckedDispatch<'_>, RocmOwnedDispatchError> {
        if !self.checked_arithmetic {
            return Err(RocmOwnedDispatchError::CheckedArithmeticRequired);
        }
        let mut fault_word = self.runtime.allocate(core::mem::size_of::<u64>())?;
        fault_word.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
        Ok(RocmSequentialCheckedDispatch {
            dispatch: self,
            fault_word,
            state: FaultWordState::Sentinel,
            poisoned: false,
        })
    }

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
        self.validate_bindings(bindings)?;
        let fault_word = if self.checked_arithmetic {
            Some(self.runtime.allocate(core::mem::size_of::<u64>())?)
        } else {
            None
        };
        self.submit_validated(bindings, fault_word, self.checked_arithmetic)
    }

    #[cfg(test)] // GPU conformance tests reuse explicit status storage with reset-always semantics.
    /// Submits a checked kernel with caller-owned status storage.
    ///
    /// This crate-private path is for synchronous typed wrappers that wait for each completion
    /// before reusing the supplied storage. HIP allocation access gates reject an overlapping
    /// launch through another clone. The completion retains its own allocation lease, so uncertain
    /// waits keep the storage alive alongside the caller's owner.
    /// Public `submit` remains independently safe for overlapping submissions by allocating one
    /// status word per completion.
    #[allow(clippy::needless_pass_by_ref_mut)] // Typed wrappers hold exclusive status ownership and wait before reuse.
    pub(crate) fn submit_with_fault_word(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        fault_word: &mut DeviceBuffer,
    ) -> Result<RocmOwnedCompletion, RocmOwnedDispatchError> {
        self.submit_with_fault_word_state(bindings, fault_word, true)
    }

    /// Submit with caller-proven sentinel state for synchronous prepared wrappers.
    ///
    /// `reset_fault_word` may be false only after the prior launch reached terminal success and
    /// its status readback observed `FAULT_WORD_SENTINEL`. The ordinary submit method always
    /// resets caller-provided storage.
    #[allow(clippy::needless_pass_by_ref_mut)] // Typed wrappers hold exclusive status ownership and wait before reuse.
    pub(crate) fn submit_with_fault_word_state(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        fault_word: &mut DeviceBuffer,
        reset_fault_word: bool,
    ) -> Result<RocmOwnedCompletion, RocmOwnedDispatchError> {
        if !self.checked_arithmetic {
            return Err(RocmOwnedDispatchError::CheckedArithmeticRequired);
        }
        self.validate_bindings(bindings)?;
        if fault_word.len() != core::mem::size_of::<u64>() {
            return Err(RocmOwnedDispatchError::CheckedFaultWordSize {
                actual: fault_word.len(),
            });
        }
        self.runtime
            .ensure_same_runtime(&fault_word.allocation.runtime)
            .map_err(|_| HipError::DifferentRuntime)?;
        self.submit_validated(bindings, Some(fault_word.clone()), reset_fault_word)
    }

    fn validate_bindings(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), RocmOwnedDispatchError> {
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
        Ok(())
    }

    fn submit_validated(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        mut fault_word: Option<DeviceBuffer>,
        reset_fault_word: bool,
    ) -> Result<RocmOwnedCompletion, RocmOwnedDispatchError> {
        // Kernel arguments are borrowed only during `launch`; HIP copies their pointer values
        // into owned aligned storage before returning. Keep common small interfaces on the stack
        // without constraining larger kernels to an arbitrary binding-count limit.
        if self.checked_arithmetic && reset_fault_word {
            let buffer = fault_word
                .as_mut()
                .ok_or(RocmOwnedDispatchError::CheckedBatchFaultWordUnavailable)?;
            buffer.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
        }
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
    /// by this prepared dispatch and must be finished after the final queued operation. Checked
    /// `DivRem` is rejected because its fault word is observed only after completion; subsequent
    /// queued work could otherwise consume invalid arithmetic results before the fault
    /// becomes visible. If this method returns a HIP launch error, the batch is poisoned and must
    /// be dropped; its drop path synchronizes the stream or quarantines its retained resources.
    ///
    /// # Errors
    /// Returns an error for invalid bindings, a mismatched runtime/device/stream, a poisoned
    /// batch, or HIP launch failure.
    pub fn submit_into_batch(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        batch: &mut HipCompletionBatch,
    ) -> Result<(), RocmOwnedDispatchError> {
        validate_batch_fault_semantics(self.checked_arithmetic)?;
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

    fn submit_checked_into_batch(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        batch: &mut HipCompletionBatch,
        fault_word: &DeviceBuffer,
    ) -> Result<(), RocmOwnedDispatchError> {
        if !self.checked_arithmetic {
            return Err(RocmOwnedDispatchError::CheckedArithmeticRequired);
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
        self.runtime
            .ensure_same_runtime(&fault_word.allocation.runtime)
            .map_err(|_| RocmOwnedDispatchError::Hip(HipError::DifferentRuntime))?;
        let argument_count = self.binding_targets.len() + 1;
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
            inline_arguments[self.binding_targets.len()] = HipKernelArgument::Buffer(fault_word);
            &inline_arguments[..argument_count]
        } else {
            overflow_arguments.reserve(argument_count);
            for target in self.binding_targets.iter().copied() {
                let binding =
                    find_binding(target, bindings).map_err(RocmOwnedDispatchError::Binding)?;
                overflow_arguments.push(HipKernelArgument::Buffer(&binding.resource));
            }
            overflow_arguments.push(HipKernelArgument::Buffer(fault_word));
            &overflow_arguments
        };

        // SAFETY: the checked DivRem lowerer appends one u64 fault pointer after the declared
        // binding pointers. The batch launch retains every argument allocation until its final
        // event completes, including the fault word owned by the wrapper.
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

/// Checked prepared dispatch with owned status storage for strictly sequential use.
///
/// This wrapper is deliberately synchronous. Its mutable borrow prevents safe Rust callers from
/// starting another launch with this status word before the previous completion is terminal.
pub struct RocmSequentialCheckedDispatch<'a> {
    dispatch: &'a RocmPreparedDispatch,
    fault_word: DeviceBuffer,
    state: FaultWordState,
    poisoned: bool,
}

impl RocmSequentialCheckedDispatch<'_> {
    /// Submit a checked dispatch and wait for its terminal status before returning.
    ///
    /// # Errors
    ///
    /// Returns validation, HIP launch/wait/readback, or invalid checked-status errors. Uncertain
    /// launch/wait/readback errors poison this owner; a terminal arithmetic fault is returned as
    /// `Ok(PcuCompletionOutcome::Fault(_))` so callers may recover and submit again.
    pub fn submit_and_wait(
        &mut self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<PcuCompletionOutcome, RocmOwnedDispatchError> {
        if self.poisoned {
            return Err(RocmOwnedDispatchError::CheckedSequentialDispatchPoisoned);
        }
        let reset = self.state.begin_submission();
        let submission =
            self.dispatch
                .submit_with_fault_word_state(bindings, &mut self.fault_word, reset);
        let mut completion = match submission {
            Ok(completion) => completion,
            Err(error) => {
                if is_certain_checked_prelaunch_error(&error) {
                    self.state = FaultWordState::NeedsReset;
                } else {
                    self.poisoned = true;
                }
                return Err(error);
            }
        };
        let outcome = match PcuOwnedCompletion::wait(&mut completion) {
            Ok(outcome) => outcome,
            Err(error) => {
                self.poisoned = true;
                return Err(RocmOwnedDispatchError::Hip(error));
            }
        };
        self.state = FaultWordState::after_terminal(outcome);
        Ok(outcome)
    }
}

const fn is_certain_checked_prelaunch_error(error: &RocmOwnedDispatchError) -> bool {
    matches!(
        error,
        RocmOwnedDispatchError::Binding(_)
            | RocmOwnedDispatchError::BufferSizeMismatch { .. }
            | RocmOwnedDispatchError::DifferentRuntime(_)
            | RocmOwnedDispatchError::MemoryAccessMismatch(_)
            | RocmOwnedDispatchError::CheckedFaultWordSize { .. }
    )
}

/// Ordered `ROCm` batch that may end in one checked `DivRem` dispatch.
///
/// Unchecked dispatches may be appended before the checked dispatch. Once the checked dispatch
/// is submitted this wrapper exposes no path to enqueue later work. The caller must also avoid
/// externally enqueueing dependent work on the same stream until `finish`'s completion has been
/// waited and its checked outcome observed; this wrapper cannot constrain other stream users.
pub struct RocmCheckedDispatchBatch {
    batch: Option<HipCompletionBatch>,
    checked_attempted: bool,
    checked_submitted: bool,
    fault_word: Option<DeviceBuffer>,
}

impl RocmCheckedDispatchBatch {
    /// Start an ordered batch on `stream`.
    #[must_use]
    pub fn new(stream: &crate::HipStreamHandle) -> Self {
        Self {
            batch: Some(HipCompletionBatch::new(stream)),
            checked_attempted: false,
            checked_submitted: false,
            fault_word: None,
        }
    }

    /// Append an unchecked dispatch before the checked terminal dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error if the checked dispatch was already attempted, if the HIP batch is no
    /// longer open, or if binding validation or HIP launch fails.
    pub fn submit_unchecked(
        &mut self,
        dispatch: &RocmPreparedDispatch,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), RocmOwnedDispatchError> {
        if self.checked_attempted {
            return Err(RocmOwnedDispatchError::CheckedArithmeticBatchClosed);
        }
        let batch = self
            .batch
            .as_mut()
            .ok_or(RocmOwnedDispatchError::CheckedBatchUnavailable)?;
        dispatch.submit_into_batch(bindings, batch)
    }

    /// Append the checked `DivRem` dispatch as the final launch in this wrapper's batch.
    ///
    /// Its fault word stays owned across event synchronization and subsequent device readback.
    /// If synchronization or readback fails, the returned completion retains the necessary
    /// owners for a retry. Enqueue failures poison the underlying HIP batch, whose drop path
    /// synchronizes or quarantines every retained allocation. The caller must not enqueue
    /// external work that depends on the checked outputs until this batch's result has been read.
    ///
    /// # Errors
    ///
    /// Returns an error if the executable is not checked `DivRem`, this wrapper is already closed,
    /// allocation or initialization fails, bindings are invalid, or HIP launch fails.
    pub fn submit_checked_last(
        &mut self,
        dispatch: &RocmPreparedDispatch,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), RocmOwnedDispatchError> {
        if self.checked_attempted {
            return Err(RocmOwnedDispatchError::CheckedArithmeticBatchClosed);
        }
        if !dispatch.checked_arithmetic {
            return Err(RocmOwnedDispatchError::CheckedArithmeticRequired);
        }
        let mut fault_word = dispatch.runtime.allocate(core::mem::size_of::<u64>())?;
        fault_word.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
        // Prevent another enqueue attempt before entering the substrate. A HIP enqueue failure
        // poisons the batch, so this wrapper is closed even when submission returns an error.
        // Store the allocation first; on enqueue failure the wrapper keeps the owner until its
        // poisoned batch has synchronized or quarantined resources.
        self.checked_attempted = true;
        self.fault_word = Some(fault_word);
        let batch = self
            .batch
            .as_mut()
            .ok_or(RocmOwnedDispatchError::CheckedBatchUnavailable)?;
        let fault_word = self
            .fault_word
            .as_ref()
            .ok_or(RocmOwnedDispatchError::CheckedBatchFaultWordUnavailable)?;
        dispatch.submit_checked_into_batch(bindings, batch, fault_word)?;
        self.checked_submitted = true;
        Ok(())
    }

    /// Finish the stream batch and return a completion that reports checked execution faults.
    ///
    /// # Errors
    ///
    /// Returns an error if no checked dispatch was successfully submitted or if HIP cannot
    /// record the final completion event.
    pub fn finish(mut self) -> Result<RocmCheckedBatchCompletion, RocmOwnedDispatchError> {
        if !self.checked_submitted {
            return Err(RocmOwnedDispatchError::CheckedArithmeticRequired);
        }
        let batch = self
            .batch
            .as_mut()
            .ok_or(RocmOwnedDispatchError::CheckedBatchUnavailable)?;
        let hip = batch.finish()?;
        self.batch.take();
        Ok(RocmCheckedBatchCompletion {
            hip: Some(hip),
            fault_word: self.fault_word.take(),
            terminal: None,
        })
    }
}

/// Retryable completion for a `ROCm` batch ending in checked `DivRem`.
pub struct RocmCheckedBatchCompletion {
    hip: Option<crate::HipBatchCompletion>,
    fault_word: Option<DeviceBuffer>,
    terminal: Option<PcuCompletionOutcome>,
}

impl RocmCheckedBatchCompletion {
    /// Wait for all launches and read the terminal checked-arithmetic status.
    ///
    /// A returned fault means outputs produced by the checked dispatch are invalid. HIP wait and
    /// readback errors retain this object's owners, so the caller may retry.
    ///
    /// # Errors
    ///
    /// Returns a HIP wait/readback error, an invalid fault word error, or an invalid-state error
    /// if the fault-word owner is unavailable.
    pub fn wait(&mut self) -> Result<PcuCompletionOutcome, RocmOwnedDispatchError> {
        if let Some(terminal) = self.terminal {
            return Ok(terminal);
        }
        if let Some(hip) = self.hip.as_mut() {
            hip.wait()?;
            self.hip.take();
        }
        let fault_word = self
            .fault_word
            .as_ref()
            .ok_or(RocmOwnedDispatchError::CheckedBatchFaultWordUnavailable)?;
        let mut bytes = [0_u8; core::mem::size_of::<u64>()];
        fault_word.copy_to(&mut bytes)?;
        let outcome = decode_fault_word(u64::from_le_bytes(bytes))?
            .map_or(PcuCompletionOutcome::Succeeded, PcuCompletionOutcome::Fault);
        self.fault_word.take();
        self.terminal = Some(outcome);
        Ok(outcome)
    }
}

impl PcuOwnedCompletion for RocmCheckedBatchCompletion {
    type Error = RocmOwnedDispatchError;

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
        Self::wait(self)
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

impl RocmOwnedCompletion {
    pub(crate) fn can_handoff_to_batch(
        &self,
        batch: &crate::HipCompletionBatch,
    ) -> Result<bool, HipError> {
        if self.fault_word.is_some() || self.terminal.is_some() {
            return Ok(false);
        }
        self.hip
            .as_ref()
            .map_or(Ok(false), |completion| batch.can_wait_for(completion))
    }

    pub(crate) const fn take_hip_for_handoff(&mut self) -> Option<HipCompletion> {
        if self.fault_word.is_none() && self.terminal.is_none() {
            self.hip.take()
        } else {
            None
        }
    }
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
            .union(PcuValueTypeCaps::FLOAT16)
            .union(PcuValueTypeCaps::BFLOAT16)
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
            .union(PcuDispatchOpCaps::VALUE_CAST)
            .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
            .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
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
            .with(fusion_pcu::PcuScalarType::F64, f64_alu_caps())
            .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U16, checked_u16_alu_caps())
            .with(fusion_pcu::PcuScalarType::I16, checked_i16_alu_caps())
            .with(fusion_pcu::PcuScalarType::U8, checked_u8_alu_caps())
            .with(fusion_pcu::PcuScalarType::I8, checked_i8_alu_caps())
            .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
            .with(fusion_pcu::PcuScalarType::I64, checked_i64_alu_caps()),
        fusion_pcu::PcuDispatchScalarAluSupport::empty(),
    );
    dispatch.features = PcuFeatureSupport::new(
        PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES)
            .union(PcuDispatchFeatureCaps::RANGE_CLAMP),
        PcuDispatchFeatureCaps::empty(),
    );
    support.dispatch_support = dispatch;
    support
}

const OWNED_DISPATCH_INSTRUCTIONS: PcuDispatchOpCaps = PcuDispatchOpCaps::VALUE_CONSTANT
    .union(PcuDispatchOpCaps::VALUE_CAST)
    .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
    .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
    .union(PcuDispatchOpCaps::BINDING_STORE);

const fn f32_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
}

const fn f64_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
}

const fn int_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
}

const fn checked_u8_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_u32_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_u16_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_u64_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_i32_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_i16_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_i8_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
}

const fn checked_i64_alu_caps() -> PcuDispatchOpCaps {
    int_alu_caps()
        .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
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
            .union(PcuValueTypeCaps::FLOAT16)
            .union(PcuValueTypeCaps::BFLOAT16)
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
            .with(fusion_pcu::PcuScalarType::F64, f64_alu_caps())
            .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U16, checked_u16_alu_caps())
            .with(fusion_pcu::PcuScalarType::I16, checked_i16_alu_caps())
            .with(fusion_pcu::PcuScalarType::U8, checked_u8_alu_caps())
            .with(fusion_pcu::PcuScalarType::I8, checked_i8_alu_caps())
            .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
            .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
            .with(fusion_pcu::PcuScalarType::I64, checked_i64_alu_caps()),
        dispatch_features: PcuDispatchFeatureCaps::MUTABLE_RESOURCES
            .union(PcuDispatchFeatureCaps::READ_ONLY_RESOURCES)
            .union(PcuDispatchFeatureCaps::RANGE_CLAMP),
        stream_instructions: fusion_pcu::PcuStreamCapabilities::empty(),
        command_instructions: fusion_pcu::PcuCommandOpCaps::empty(),
        transaction_features: fusion_pcu::PcuTransactionFeatureCaps::empty(),
        signal_instructions: fusion_pcu::PcuSignalOpCaps::empty(),
    },
}];

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        decode_fault_word,
        validate_batch_fault_semantics,
        validate_dispatch_requirements,
        ops_use_checked_arithmetic,
        FAULT_WORD_SENTINEL,
        RocmOwnedDispatchError,
        RocmCheckedBatchCompletion,
        OWNED_DISPATCH_INSTRUCTIONS,
        OWNED_EXECUTORS,
        owned_dispatch_support,
        find_binding,
        memory_access_supports_binding,
        launch_grid,
        RECOVERED_FAULT_BIT,
    };
    #[rustfmt::skip]
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
        PcuOwnedCompletion,
        PcuCompletionOutcome,
        PcuCompletionState,
        PcuProviderId,
        PcuValueType,
        PcuDispatchOp,
        PcuDispatchDataOp,
        PcuDispatchValueId,
        PcuExecutionFault,
        PcuExecutionFaultKind,
        PcuExecutionGraphError,
        PcuExecutionNode,
        PcuExecutionResourceId,
        PcuExecutionResourceUse,
        validate_execution_fault_gates,
        validate_execution_graph,
    };
    use fusion_pcu::model::PcuIntegerDivFlags;

    struct Noop;

    #[test]
    fn owned_dispatch_rejects_legacy_raw_value_alu_capabilities() {
        let raw = PcuDispatchOpCaps::ALU_ADD
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL)
            .union(PcuDispatchOpCaps::ALU_DIV)
            .union(PcuDispatchOpCaps::ALU_MIN)
            .union(PcuDispatchOpCaps::ALU_MAX);
        let support = owned_dispatch_support();
        assert_eq!(
            support.dispatch_support.instructions.direct.bits() & raw.bits(),
            0
        );
        assert_eq!(OWNED_DISPATCH_INSTRUCTIONS.bits() & raw.bits(), 0);
        for scalar in [
            fusion_pcu::PcuScalarType::F32,
            fusion_pcu::PcuScalarType::F64,
            fusion_pcu::PcuScalarType::I8,
            fusion_pcu::PcuScalarType::U8,
            fusion_pcu::PcuScalarType::I16,
            fusion_pcu::PcuScalarType::U16,
            fusion_pcu::PcuScalarType::I32,
            fusion_pcu::PcuScalarType::U32,
            fusion_pcu::PcuScalarType::I64,
            fusion_pcu::PcuScalarType::U64,
        ] {
            assert_eq!(
                support
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar)
                    .bits()
                    & raw.bits(),
                0
            );
            assert_eq!(
                OWNED_EXECUTORS[0]
                    .support
                    .dispatch_scalar_alu
                    .for_scalar(scalar)
                    .bits()
                    & raw.bits(),
                0
            );
        }
    }

    #[test]
    fn checked_batch_completion_implements_common_owned_completion_contract() {
        fn state<C: PcuOwnedCompletion>(completion: &C) -> PcuCompletionState {
            completion
                .state()
                .ok()
                .expect("terminal state is available")
        }

        let completion = RocmCheckedBatchCompletion {
            hip: None,
            fault_word: None,
            terminal: Some(PcuCompletionOutcome::Fault(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: 7,
                recovered: false,
            })),
        };
        assert_eq!(state(&completion), PcuCompletionState::Failed);
    }

    #[test]
    fn ordered_batch_acceptance_preserves_checked_arithmetic_fault_boundary() {
        assert!(validate_batch_fault_semantics(false).is_ok());
        assert!(matches!(
            validate_batch_fault_semantics(true),
            Err(RocmOwnedDispatchError::CheckedArithmeticBatchUnsupported)
        ));

        let result = PcuExecutionResourceUse {
            resource: PcuExecutionResourceId(0),
            access: PcuMemoryAccess::WriteOnly,
        };
        let consumed = PcuExecutionResourceUse {
            access: PcuMemoryAccess::ReadOnly,
            ..result
        };
        // An in-order HIP stream establishes ordering, but a later launch must not read a
        // checked kernel's output before its terminal fault word has been observed.
        let nodes = [
            PcuExecutionNode {
                dependencies: &[],
                resources: &[result],
            },
            PcuExecutionNode {
                dependencies: &[0],
                resources: &[consumed],
            },
        ];
        assert!(validate_execution_graph(&nodes, 1, &mut [false; 2]).is_ok());
        assert_eq!(
            validate_execution_fault_gates(&nodes, &[true, false], &[], &mut [false; 2]),
            Err(PcuExecutionGraphError::FaultOutputNotSuccessGated {
                producer: 0,
                consumer: 1,
                resource: PcuExecutionResourceId(0),
            })
        );
    }

    #[test]
    fn checked_arithmetic_scan_finds_nested_region_operations() {
        let checked = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
            value_type: PcuValueType::Scalar(fusion_pcu::PcuScalarType::I32),
            flags: PcuIntegerDivFlags::CHECKED,
            quotient: PcuDispatchValueId(0),
            remainder: PcuDispatchValueId(1),
            lhs: PcuDispatchValueId(2),
            rhs: PcuDispatchValueId(3),
        });
        let inner = [checked];
        let outer = [PcuDispatchOp::GridStrideLoop {
            extent: 4,
            body: &inner,
        }];
        assert!(ops_use_checked_arithmetic(&outer));
        let checked_cast = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
            conversion: fusion_pcu::PcuDispatchCheckedFloatConversion::F64ToF32,
            underflow_policy: fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: fusion_pcu::PcuRangePolicy::Reject,
            result: PcuDispatchValueId(4),
            value: PcuDispatchValueId(3),
        });
        let cast_body = [checked_cast];
        let cast_loop = [PcuDispatchOp::GridStrideLoop {
            extent: 4,
            body: &cast_body,
        }];
        assert!(ops_use_checked_arithmetic(&cast_loop));

        let checked_relu = PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatUnary {
            value_type: PcuValueType::f32(),
            op: fusion_pcu::PcuDispatchFloatUnaryOp::Relu,
            underflow_policy: fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
            range_policy: fusion_pcu::PcuRangePolicy::Reject,
            result: PcuDispatchValueId(5),
            value: PcuDispatchValueId(4),
        });
        assert!(ops_use_checked_arithmetic(&[checked_relu]));
        assert!(validate_batch_fault_semantics(true).is_err());
    }

    #[test]
    fn checked_arithmetic_fault_word_decodes_sentinel_and_logical_invocation() {
        assert_eq!(decode_fault_word(FAULT_WORD_SENTINEL), Ok(None));
        assert_eq!(
            decode_fault_word((37_u64 << 3) | 1),
            Ok(Some(PcuExecutionFault {
                kind: PcuExecutionFaultKind::DivideByZero,
                invocation_id: 37,
                recovered: false,
            }))
        );
        assert_eq!(
            decode_fault_word((37_u64 << 3) | 2),
            Ok(Some(PcuExecutionFault {
                kind: PcuExecutionFaultKind::SignedDivisionOverflow,
                invocation_id: 37,
                recovered: false,
            }))
        );
        for (tag, kind) in [
            (3, PcuExecutionFaultKind::ArithmeticOverflow),
            (4, PcuExecutionFaultKind::ArithmeticUnderflow),
            (5, PcuExecutionFaultKind::InvalidFloatingOperand),
        ] {
            assert_eq!(
                decode_fault_word((u64::from(u32::MAX) << 3) | tag),
                Ok(Some(PcuExecutionFault {
                    kind,
                    invocation_id: u64::from(u32::MAX),
                    recovered: false,
                }))
            );
        }
        for tag in [0, 6, 7] {
            assert!(decode_fault_word((37_u64 << 3) | tag).is_err());
        }
    }

    #[test]
    fn range_fault_words_decode_recovery_and_fatal_faults_sort_first() {
        for (tag, kind) in [
            (3, PcuExecutionFaultKind::ArithmeticOverflow),
            (4, PcuExecutionFaultKind::ArithmeticUnderflow),
        ] {
            assert_eq!(
                decode_fault_word(RECOVERED_FAULT_BIT | (19_u64 << 3) | tag),
                Ok(Some(PcuExecutionFault {
                    kind,
                    invocation_id: 19,
                    recovered: true,
                }))
            );
        }
        assert!(decode_fault_word(RECOVERED_FAULT_BIT | (2_u64 << 3) | 1).is_err());
        assert!(decode_fault_word(RECOVERED_FAULT_BIT | (2_u64 << 3) | 5).is_err());

        let lowest_recovered = RECOVERED_FAULT_BIT | 3;
        let highest_fatal = (u64::from(u32::MAX) << 3) | 5;
        assert!(highest_fatal < lowest_recovered);
        assert_eq!(
            decode_fault_word(core::cmp::min(lowest_recovered, highest_fatal)),
            Ok(Some(PcuExecutionFault {
                kind: PcuExecutionFaultKind::InvalidFloatingOperand,
                invocation_id: u64::from(u32::MAX),
                recovered: false,
            }))
        );
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
    fn owned_dispatch_advertises_prepared_conversion_floor() {
        let required_types = fusion_pcu::PcuValueTypeCaps::FLOAT16
            .union(fusion_pcu::PcuValueTypeCaps::BFLOAT16)
            .union(fusion_pcu::PcuValueTypeCaps::FLOAT32)
            .union(fusion_pcu::PcuValueTypeCaps::SCALAR_VALUES);
        assert!(
            owned_dispatch_support()
                .value_type_support
                .direct
                .contains(required_types)
        );
        assert!(
            OWNED_EXECUTORS[0]
                .support
                .value_types
                .contains(required_types)
        );
        assert!(
            owned_dispatch_support()
                .dispatch_support
                .instructions
                .direct
                .contains(PcuDispatchOpCaps::VALUE_CAST)
        );
        assert!(OWNED_DISPATCH_INSTRUCTIONS.contains(PcuDispatchOpCaps::VALUE_CAST));
    }

    #[test]
    fn owned_dispatch_advertises_checked_float_binary_for_f32_and_f64() {
        let checked = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY;
        let checked_unary = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY;
        for scalar_type in [
            fusion_pcu::PcuScalarType::F32,
            fusion_pcu::PcuScalarType::F64,
        ] {
            assert!(
                owned_dispatch_support()
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar_type)
                    .contains(checked)
            );
            assert!(
                OWNED_EXECUTORS[0]
                    .support
                    .dispatch_scalar_alu
                    .for_scalar(scalar_type)
                    .contains(checked)
            );
            assert!(
                owned_dispatch_support()
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar_type)
                    .contains(checked_unary)
            );
            assert!(
                OWNED_EXECUTORS[0]
                    .support
                    .dispatch_scalar_alu
                    .for_scalar(scalar_type)
                    .contains(checked_unary)
            );
            assert_eq!(
                owned_dispatch_support()
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar_type)
                    .bits()
                    & (PcuDispatchOpCaps::ALU_ADD
                        .union(PcuDispatchOpCaps::ALU_SUB)
                        .union(PcuDispatchOpCaps::ALU_MUL)
                        .union(PcuDispatchOpCaps::ALU_DIV)
                        .union(PcuDispatchOpCaps::ALU_MIN)
                        .union(PcuDispatchOpCaps::ALU_MAX))
                    .bits(),
                0
            );
        }
        assert!(OWNED_DISPATCH_INSTRUCTIONS.contains(checked));
        assert!(OWNED_DISPATCH_INSTRUCTIONS.contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT));
        assert!(
            owned_dispatch_support()
                .dispatch_support
                .instructions
                .direct
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
        );
    }

    fn checked_conversion_kernel_is_supported(
        conversion: fusion_pcu::PcuDispatchCheckedFloatConversion,
        grid: bool,
        support: &fusion_pcu::PcuSupport,
    ) -> Result<(), RocmOwnedDispatchError> {
        #[rustfmt::skip]
        use fusion_pcu::{
            PcuBinding,
            PcuBindingStorageClass,
            PcuDispatchControlOp,
            PcuDispatchEntryPoint,
            PcuDispatchFeatureCaps,
            PcuDispatchIndex,
            PcuDispatchKernelIr,
        };

        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                conversion.source_type(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                conversion.target_type(),
            ),
        ];
        let index = if grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatConvert {
                conversion,
                underflow_policy: fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                result: PcuDispatchValueId(2),
                value: PcuDispatchValueId(1),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 1),
                index,
                value: PcuDispatchValueId(2),
            }),
        ];
        let direct_ops = [
            body[0],
            body[1],
            body[2],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 8,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: fusion_pcu::PcuKernelId(0xfeed),
            entry: PcuDispatchEntryPoint {
                name: "checked_float_conversion_caps",
                logical_shape: [8, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &grid_ops } else { &direct_ops },
            type_caps: fusion_pcu::PcuValueTypeCaps::FLOAT32
                | fusion_pcu::PcuValueTypeCaps::FLOAT64,
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };
        validate_dispatch_requirements(&kernel, support)
    }

    fn checked_float_binary_kernel_is_supported(
        value_type: fusion_pcu::PcuValueType,
        op: fusion_pcu::PcuDispatchFloatBinaryOp,
        grid: bool,
        support: &fusion_pcu::PcuSupport,
    ) -> bool {
        #[rustfmt::skip]
        use fusion_pcu::{
            PcuBinding,
            PcuBindingStorageClass,
            PcuDispatchControlOp,
            PcuDispatchEntryPoint,
            PcuDispatchFeatureCaps,
            PcuDispatchIndex,
            PcuDispatchKernelIr,
            PcuFloatUnderflowPolicy,
            PcuKernel,
        };

        let bindings = [0, 1].map(|slot| {
            PcuBinding::value(
                Some(if slot == 0 { "lhs" } else { "rhs" }),
                0,
                slot,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                value_type,
            )
        });
        let bindings = [
            bindings[0],
            bindings[1],
            PcuBinding::value(
                Some("output"),
                0,
                2,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                value_type,
            ),
        ];
        let index = if grid {
            PcuDispatchIndex::GridStrideId
        } else {
            PcuDispatchIndex::InvocationId
        };
        let body = [
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(1),
                binding: PcuBindingRef::new(0, 0),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
                result: PcuDispatchValueId(2),
                binding: PcuBindingRef::new(0, 1),
                index,
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::CheckedFloatBinary {
                value_type,
                op,
                underflow_policy: PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                result: PcuDispatchValueId(3),
                lhs: PcuDispatchValueId(1),
                rhs: PcuDispatchValueId(2),
            }),
            PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
                binding: PcuBindingRef::new(0, 2),
                index,
                value: PcuDispatchValueId(3),
            }),
        ];
        let direct_ops = [
            body[0],
            body[1],
            body[2],
            body[3],
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let grid_ops = [
            PcuDispatchOp::GridStrideLoop {
                extent: 8,
                body: &body,
            },
            PcuDispatchOp::Control(PcuDispatchControlOp::Return),
        ];
        let kernel = PcuDispatchKernelIr {
            id: fusion_pcu::PcuKernelId(0xbeef),
            entry: PcuDispatchEntryPoint {
                name: "checked_float_binary_caps",
                logical_shape: [8, 1, 1],
            },
            bindings: &bindings,
            ports: &[],
            parameters: &[],
            ops: if grid { &grid_ops } else { &direct_ops },
            type_caps: if value_type == fusion_pcu::PcuValueType::f32() {
                fusion_pcu::PcuValueTypeCaps::FLOAT32
            } else {
                fusion_pcu::PcuValueTypeCaps::FLOAT64
            },
            feature_caps: PcuDispatchFeatureCaps::empty(),
        };
        support.supports_kernel_direct(PcuKernel::Dispatch(kernel))
    }

    #[test]
    fn checked_float_conversion_caps_admit_both_widths_and_require_both_width_caps() {
        #[rustfmt::skip]
        use fusion_pcu::{
            PcuDispatchCheckedFloatConversion as Conversion,
            PcuScalarType,
        };

        let support = owned_dispatch_support();
        for conversion in [Conversion::F32ToF64, Conversion::F64ToF32] {
            for grid in [false, true] {
                assert!(checked_conversion_kernel_is_supported(conversion, grid, &support).is_ok());
            }
        }

        // Keep every pre-existing F32/F64 ALU claim while removing only the conversion claim.
        let f32_without_conversion = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_ADD)
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL)
            .union(PcuDispatchOpCaps::ALU_DIV)
            .union(PcuDispatchOpCaps::ALU_MAX);
        let f64_without_conversion = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_ADD)
            .union(PcuDispatchOpCaps::ALU_SUB)
            .union(PcuDispatchOpCaps::ALU_MUL)
            .union(PcuDispatchOpCaps::ALU_DIV)
            .union(PcuDispatchOpCaps::ALU_MIN)
            .union(PcuDispatchOpCaps::ALU_MAX);
        for (missing_width, without_conversion) in [
            (PcuScalarType::F32, f32_without_conversion),
            (PcuScalarType::F64, f64_without_conversion),
        ] {
            let mut incomplete = support;
            let complete = incomplete.dispatch_support.scalar_alu.direct;
            incomplete.dispatch_support.scalar_alu.direct =
                fusion_pcu::PcuDispatchScalarAluSupport::empty()
                    .with(
                        PcuScalarType::F32,
                        if missing_width == PcuScalarType::F32 {
                            without_conversion
                        } else {
                            complete.for_scalar(PcuScalarType::F32)
                        },
                    )
                    .with(
                        PcuScalarType::F64,
                        if missing_width == PcuScalarType::F64 {
                            without_conversion
                        } else {
                            complete.for_scalar(PcuScalarType::F64)
                        },
                    );
            for conversion in [Conversion::F32ToF64, Conversion::F64ToF32] {
                for grid in [false, true] {
                    assert!(matches!(
                        checked_conversion_kernel_is_supported(conversion, grid, &incomplete),
                        Err(RocmOwnedDispatchError::UnsupportedRequirements)
                    ));
                }
            }
        }

        let mut missing_instruction = support;
        missing_instruction.dispatch_support.instructions.direct = PcuDispatchOpCaps::BINDING_LOAD
            .union(PcuDispatchOpCaps::BINDING_STORE)
            .union(PcuDispatchOpCaps::CONTROL_RETURN);
        let mut missing_feature = support;
        missing_feature.dispatch_support.features.direct =
            fusion_pcu::PcuDispatchFeatureCaps::READ_ONLY_RESOURCES;
        for conversion in [Conversion::F32ToF64, Conversion::F64ToF32] {
            for grid in [false, true] {
                assert!(matches!(
                    checked_conversion_kernel_is_supported(conversion, grid, &missing_instruction),
                    Err(RocmOwnedDispatchError::UnsupportedRequirements)
                ));
                assert!(matches!(
                    checked_conversion_kernel_is_supported(conversion, grid, &missing_feature),
                    Err(RocmOwnedDispatchError::UnsupportedRequirements)
                ));
            }
        }
    }

    #[test]
    fn owned_admission_preserves_checked_float_binary_profiles() {
        use fusion_pcu::model::PcuDispatchFloatBinaryOp as Binary;

        let support = owned_dispatch_support();
        for value_type in [PcuValueType::f32(), PcuValueType::f64()] {
            for op in [Binary::Add, Binary::Sub, Binary::Mul, Binary::Div] {
                for grid in [false, true] {
                    assert!(checked_float_binary_kernel_is_supported(
                        value_type, op, grid, &support
                    ));
                }
            }
        }
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
        #[rustfmt::skip]
        use PcuBindingAccess::{
            ReadOnly,
            ReadWrite,
            WriteOnly,
        };
        #[rustfmt::skip]
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

mod execution;
#[rustfmt::skip]
pub use execution::{
    RocmExecutionStep,
    RocmOwnedExecution,
    RocmOwnedExecutionError,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmTwoSlotExecutionStep,
};

#[cfg(test)]
#[path = "owned_dispatch/checked_integer_tests.rs"]
mod checked_integer_tests;

#[cfg(test)]
#[path = "owned_dispatch/checked_float_tests.rs"]
mod checked_float_tests;

#[cfg(test)]
#[path = "owned_dispatch/checked_unary_tests.rs"]
mod checked_unary_tests;

mod fault_word;
#[allow(clippy::redundant_pub_crate)]
// Keep the state machine internal even if this module is exposed later.
pub(crate) use fault_word::FaultWordState;
