//! Owned, asynchronous PCU Dispatch adapter for one explicitly selected CUDA device.
//!
//! This adapter supports the bounded scalar Dispatch profiles accepted by the `CUDA` lowerer.
//! It requires buffers allocated by this adapter's CUDA runtime and retains exclusive buffer
//! leases until the CUDA event proves that the kernel has stopped accessing them.

#[rustfmt::skip]
use std::{
    error::Error,
    fmt,
};

#[path = "fault_law.rs"]
mod fault_law;

const INLINE_ARGUMENTS: usize = 8;
const FAULT_WORD_SENTINEL: u64 = u64::MAX;
const RECOVERED_FAULT_BIT: u64 = 1 << 63;

const fn validate_batch_fault_semantics(
    checked_arithmetic: bool,
) -> Result<(), CudaOwnedDispatchError> {
    if checked_arithmetic {
        // Checked arithmetic publishes a terminal fault only after its completion token is waited.
        // A batch can enqueue dependent work before that observation, so accepting the launch
        // would let consumers read invalid arithmetic results.
        return Err(CudaOwnedDispatchError::CheckedArithmeticBatchUnsupported);
    }
    Ok(())
}

pub fn kernel_uses_checked_arithmetic(kernel: &PcuDispatchKernelIr<'_>) -> bool {
    ops_use_checked_arithmetic(kernel.ops)
}

fn validate_dispatch_requirements(
    kernel: &PcuDispatchKernelIr<'_>,
    support: &PcuSupport,
) -> Result<(), CudaOwnedDispatchError> {
    if support.supports_kernel_direct(fusion_pcu::PcuKernel::Dispatch(*kernel)) {
        Ok(())
    } else {
        Err(CudaOwnedDispatchError::UnsupportedRequirements)
    }
}

fn ops_use_checked_arithmetic(ops: &[PcuDispatchOp<'_>]) -> bool {
    ops.iter().any(|op| match op {
        PcuDispatchOp::Data(
            PcuDispatchDataOp::CheckedDivRem { .. }
            | PcuDispatchDataOp::CheckedIntegerBinary { .. }
            | PcuDispatchDataOp::CheckedFloatBinary { .. }
            | PcuDispatchDataOp::CheckedFloatConvert { .. }
            | PcuDispatchDataOp::CheckedFloatUnary { .. },
        ) => true,
        PcuDispatchOp::GridStrideLoop { body, .. } => ops_use_checked_arithmetic(body),
        _ => false,
    })
}

// CUDA lowering bounds direct invocation shapes and grid-stride extents to u32, leaving
// three tag bits for the logical index. Recovered range faults set the high bit; fatal
// faults leave it clear, so `atomicMin` always gives fatal faults priority. Within either
// class, the smallest logical index wins, and clamp-mode code records only the first
// recoverable range fault per invocation.
pub const fn decode_fault_word(word: u64) -> Result<Option<PcuExecutionFault>, CudaError> {
    decode_encoded_fault_word(word, true)
}

const fn decode_encoded_fault_word(
    word: u64,
    scalar_abi: bool,
) -> Result<Option<PcuExecutionFault>, CudaError> {
    if word == FAULT_WORD_SENTINEL {
        return Ok(None);
    }
    let recovered = word & RECOVERED_FAULT_BIT != 0;
    let payload = word & !RECOVERED_FAULT_BIT;
    // Only the backend's admitted one-dimensional u32 map index is encoded here.
    // Reject malformed statuses before classifying recovered output as publishable.
    if scalar_abi && payload >> 3 > 0xffff_ffff {
        return Err(CudaError::InvalidExecutionFaultWord(word));
    }
    let kind = match payload & 0b111 {
        1 => PcuExecutionFaultKind::DivideByZero,
        2 => PcuExecutionFaultKind::SignedDivisionOverflow,
        3 => PcuExecutionFaultKind::ArithmeticOverflow,
        4 => PcuExecutionFaultKind::ArithmeticUnderflow,
        5 => PcuExecutionFaultKind::InvalidFloatingOperand,
        _ => return Err(CudaError::InvalidExecutionFaultWord(word)),
    };
    if recovered
        && !matches!(
            kind,
            PcuExecutionFaultKind::ArithmeticOverflow | PcuExecutionFaultKind::ArithmeticUnderflow
        )
    {
        return Err(CudaError::InvalidExecutionFaultWord(word));
    }
    Ok(Some(PcuExecutionFault {
        kind,
        invocation_id: payload >> 3,
        recovered,
    }))
}

#[cfg(test)]
fn decode_fault_word_in_extent(
    word: u64,
    extent: u32,
) -> Result<Option<PcuExecutionFault>, CudaError> {
    match decode_fault_word(word) {
        Ok(Some(fault)) if !fault.is_within_logical_extent(u64::from(extent)) => {
            Err(CudaError::InvalidExecutionFaultWord(word))
        }
        result => result,
    }
}

// The raw encoding and actual extent are independent of the operation's fault law.
// Success has no fault to validate, so the common sentinel path skips the policy check.
fn decode_fault_word_under_law(
    word: u64,
    extent: u64,
    scalar_abi: bool,
    law: Option<fault_law::Retained>,
) -> Result<Option<PcuExecutionFault>, CudaError> {
    match decode_encoded_fault_word(word, scalar_abi) {
        Ok(Some(fault))
            if !fault.is_within_logical_extent(extent)
                || law.is_some_and(|law| !law.accepts(fault)) =>
        {
            Err(CudaError::InvalidExecutionFaultWord(word))
        }
        result => result,
    }
}

/// Detached terminal-status contract retained when the executable becomes a native graph.
#[derive(Clone, Copy)]
pub struct CudaPreparedFaultContract {
    extent: u64,
    scalar_abi: bool,
    law: Option<fault_law::Retained>,
}

impl CudaPreparedFaultContract {
    pub(super) fn decode(self, word: u64) -> Result<Option<PcuExecutionFault>, CudaError> {
        decode_fault_word_under_law(word, self.extent, self.scalar_abi, self.law)
    }

    #[cfg(test)]
    pub(super) const fn scalar(extent: u64, law: PcuCheckedScalarFaultLaw) -> Self {
        Self {
            extent,
            scalar_abi: true,
            law: Some(fault_law::Retained::Scalar(law)),
        }
    }

    #[cfg(all(test, feature = "tensor"))]
    pub(super) const fn compound(
        domain: fusion_pcu::dialect::tensor::TensorStrictFaultDomain,
    ) -> Self {
        Self {
            extent: domain.event_extent(),
            scalar_abi: false,
            law: Some(fault_law::Retained::Compound(domain)),
        }
    }
}

pub fn checked_scalar_fault_law(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Option<PcuCheckedScalarFaultLaw> {
    fault_law::capture(kernel)
}

// Cold preparation has already validated the canonical map geometry. A grid's fault
// index is its logical induction variable, independent of submitted hardware lanes.
pub const fn checked_fault_extent(kernel: &PcuDispatchKernelIr<'_>) -> u32 {
    match kernel.ops {
        [
            PcuDispatchOp::GridStrideLoop { extent, .. },
            PcuDispatchOp::Control(fusion_pcu::PcuDispatchControlOp::Return),
        ] => *extent,
        _ => kernel.entry.logical_shape[0],
    }
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
    PcuCheckedScalarFaultLaw,
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
    CudaCompletion,
    CudaError,
    CudaKernelArgument,
    CudaRuntime,
    CudaDiscovery,
    CudaLowerError,
    CudaMemoryProvider,
    CudaMemoryResource,
    compile_cuda_source,
    lower_dispatch_to_cuda_rtc_source,
    lower_dispatch_to_cuda_source,
};
use crate::CudaCompletionBatch;

/// CUDA-owned dispatch setup or submission failure.
#[derive(Debug)]
pub enum CudaOwnedDispatchError {
    Cuda(CudaError),
    Cublas(crate::CublasError),
    Lower(CudaLowerError),
    Compile(crate::CudaCompileError),
    CudaRtc(crate::CudaRtcError),
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
    CheckedSequentialDispatchPoisoned,
    CheckedBatchUnavailable,
    CheckedBatchFaultWordUnavailable,
    CheckedFaultWordSize {
        actual: usize,
    },
}

impl fmt::Display for CudaOwnedDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda(error) => error.fmt(f),
            Self::Cublas(error) => error.fmt(f),
            Self::Lower(error) => write!(f, "PCU Dispatch cannot lower to CUDA: {error}"),
            Self::Compile(error) => write!(f, "CUDA code object compilation failed: {error}"),
            Self::CudaRtc(error) => write!(f, "CUDA runtime compilation failed: {error}"),
            Self::CompilerUnavailable => f.write_str("no usable CUDA source compiler is available"),
            Self::InvalidDeviceReference => f.write_str("invalid CUDA device reference"),
            Self::InvalidBlockSize => f.write_str("CUDA block size must be nonzero"),
            Self::RuntimeDeviceMismatch { expected, actual } => write!(
                f,
                "selected CUDA device reference names device {expected}, runtime opened device {actual}"
            ),
            Self::BufferSizeMismatch {
                binding,
                metadata,
                actual,
            } => write!(
                f,
                "binding {binding:?} declares {metadata} bytes but its CUDA allocation has {actual}"
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
                "binding {binding:?} belongs to another CUDA runtime or device"
            ),
            Self::MemoryAccessMismatch(binding) => write!(
                f,
                "memory resource for binding {binding:?} does not permit the requested access"
            ),
            Self::GeometryOverflow => f.write_str("CUDA launch geometry overflow"),
            Self::Binding(error) => write!(f, "invalid owned PCU binding: {error:?}"),
            Self::UnsupportedRequirements => f.write_str(
                "CUDA does not advertise the scalar types, instructions, ALU operations, or features required by this dispatch kernel",
            ),
            Self::CheckedArithmeticBatchUnsupported => f.write_str(
                "checked arithmetic cannot be submitted through the ordered CUDA batch path",
            ),
            Self::CheckedArithmeticBatchClosed => f.write_str(
                "the checked CUDA batch has already submitted its final checked dispatch",
            ),
            Self::CheckedArithmeticRequired => {
                f.write_str("a checked CUDA batch must end with a checked arithmetic dispatch")
            }
            Self::CheckedSequentialDispatchPoisoned => {
                f.write_str("sequential checked dispatch is poisoned after an uncertain CUDA failure")
            }
            Self::CheckedBatchUnavailable => {
                f.write_str("the checked CUDA batch no longer has an open CUDA batch")
            }
            Self::CheckedBatchFaultWordUnavailable => {
                f.write_str("the checked CUDA completion has no retained fault word")
            }
            Self::CheckedFaultWordSize { actual } => write!(
                f,
                "checked dispatch fault word requires {} bytes, received {actual}",
                core::mem::size_of::<u64>()
            ),
        }
    }
}

impl Error for CudaOwnedDispatchError {}

impl From<CudaError> for CudaOwnedDispatchError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}

impl From<crate::CublasError> for CudaOwnedDispatchError {
    fn from(error: crate::CublasError) -> Self {
        Self::Cublas(error)
    }
}

impl From<CudaLowerError> for CudaOwnedDispatchError {
    fn from(error: CudaLowerError) -> Self {
        Self::Lower(error)
    }
}

impl From<crate::CudaCompileError> for CudaOwnedDispatchError {
    fn from(error: crate::CudaCompileError) -> Self {
        Self::Compile(error)
    }
}

/// Opened owned-dispatch session tied to one generation-bound discovery reference.
pub struct CudaOwnedDispatchBackend {
    runtime: CudaRuntime,
    device: PcuDeviceIdentity,
    architecture: Option<String>,
    compiler: Option<crate::discovery::DispatchCompiler>,
    block_size: u32,
}

impl CudaOwnedDispatchBackend {
    /// Create an ordered CUDA stream on this selected backend runtime and device.
    ///
    /// # Errors
    ///
    /// Returns a CUDA stream creation error.
    pub fn create_stream(&self) -> Result<crate::CudaStreamHandle, CudaOwnedDispatchError> {
        self.runtime.create_stream().map_err(Into::into)
    }

    /// Create a cuBLAS handle for this backend's selected runtime and device.
    ///
    /// # Errors
    ///
    /// Returns a cuBLAS or CUDA initialization error.
    #[cfg(feature = "tensor")]
    pub fn create_cublas(&self) -> Result<crate::Cublas, crate::CublasError> {
        crate::Cublas::new(&self.runtime)
    }

    #[cfg(feature = "tensor")]
    pub(crate) const fn tensor_runtime(&self) -> &CudaRuntime {
        &self.runtime
    }

    #[cfg(feature = "tensor")]
    pub(crate) fn compile_tensor_source(
        &self,
        source: &str,
    ) -> Result<Vec<u8>, CudaOwnedDispatchError> {
        match self.compiler {
            Some(crate::discovery::DispatchCompiler::Nvcc) => {
                let architecture = self
                    .architecture
                    .as_deref()
                    .ok_or(CudaError::MissingArchitecture)?;
                let nvcc_source = format!("#include <cuda_runtime.h>\n{source}");
                compile_cuda_source(&nvcc_source, architecture).map_err(Into::into)
            }
            Some(crate::discovery::DispatchCompiler::Nvrtc) => {
                crate::compile_cuda_source_for_device(&self.runtime, source)
                    .map_err(CudaOwnedDispatchError::CudaRtc)
            }
            None => Err(CudaOwnedDispatchError::CompilerUnavailable),
        }
    }

    /// Internal verified strict tensor profile. It derives source, exact binding schema and
    /// shape together; arbitrary user source cannot acquire a trusted checked dispatch.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_strict_matmul_dispatch(
        &self,
        profile: crate::tensor::strict_matmul::Profile,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        let shape = fusion_pcu::PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(profile.output_count())
                .expect("profile verifies nonempty output"),
        );
        let requirements = profile.requirements();
        let source = profile.source();
        let grid_x = launch_grid(shape.invocation_count().get(), self.block_size)?;
        let image = self.compile_tensor_source(&source)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_targets = requirements
            .iter()
            .map(|requirement| requirement.target)
            .collect();
        Ok(CudaPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements: requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream: stream.clone(),
            binding_targets,
            fault_extent: profile.fault_extent(),
            scalar_fault_word: false,
            fault_law: Some(fault_law::Retained::Compound(profile.fault_domain())),
            checked_arithmetic: true,
        })
    }

    /// Internal verified strict tensor profile. It derives source, exact binding schema and
    /// shape together; arbitrary user source cannot acquire a trusted checked dispatch.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_strict_sgd_dispatch(
        &self,
        profile: crate::tensor::strict_sgd::Profile,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        let shape = fusion_pcu::PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(profile.count()).expect("profile verifies nonempty output"),
        );
        let requirements = profile.requirements();
        let source = profile.source();
        let grid_x = launch_grid(shape.invocation_count().get(), self.block_size)?;
        let image = self.compile_tensor_source(&source)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_targets = requirements
            .iter()
            .map(|requirement| requirement.target)
            .collect();
        Ok(CudaPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements: requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream: stream.clone(),
            binding_targets,
            fault_extent: profile.fault_extent(),
            scalar_fault_word: false,
            fault_law: Some(fault_law::Retained::Compound(profile.fault_domain())),
            checked_arithmetic: true,
        })
    }

    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_relu_backward_dispatch(
        &self,
        profile: crate::tensor::relu_backward::Profile,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        let shape = fusion_pcu::PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(profile.count()).expect("profile verifies nonempty output"),
        );
        let requirements = profile.requirements().to_vec();
        let source = profile.source();
        let grid_x = launch_grid(shape.invocation_count().get(), self.block_size)?;
        let image = self.compile_tensor_source(&source)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_targets = requirements
            .iter()
            .map(|requirement| requirement.target)
            .collect();
        Ok(CudaPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements: requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream: stream.clone(),
            binding_targets,
            fault_extent: u64::from(shape.invocation_count().get()),
            scalar_fault_word: true,
            fault_law: profile.fault_law().map(fault_law::Retained::Scalar),
            checked_arithmetic: profile.checked(),
        })
    }

    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_strict_mse_dispatch(
        &self,
        profile: crate::tensor::strict_mse::Profile,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        let shape = fusion_pcu::PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(profile.count()).expect("profile verifies nonempty output"),
        );
        let requirements = profile.requirements().to_vec();
        let source = profile.source();
        let grid_x = launch_grid(shape.invocation_count().get(), self.block_size)?;
        let image = self.compile_tensor_source(&source)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_targets = requirements
            .iter()
            .map(|requirement| requirement.target)
            .collect();
        Ok(CudaPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements: requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream: stream.clone(),
            binding_targets,
            fault_extent: profile.fault_extent(),
            scalar_fault_word: false,
            fault_law: Some(fault_law::Retained::Compound(profile.fault_domain())),
            checked_arithmetic: profile.checked(),
        })
    }

    /// Private explicitly native MSE storage ABI. Tensor policy admission happens before this
    /// path; arbitrary unchecked scalar IR cannot use this trusted source preparation method.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_native_mse_dispatch(
        &self,
        count: u32,
        scalar: fusion_pcu::PcuScalarType,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        let shape = fusion_pcu::PcuInvocationShape::invocations(
            core::num::NonZeroU32::new(count)
                .ok_or(CudaOwnedDispatchError::UnsupportedRequirements)?,
        );
        let requirements = crate::tensor::native_mse::requirements(count, scalar);
        let grid_x = launch_grid(count, self.block_size)?;
        let image =
            self.compile_tensor_source(&crate::tensor::native_mse::source(count, scalar))?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let binding_targets = requirements
            .iter()
            .map(|requirement| requirement.target)
            .collect();
        Ok(CudaPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements: requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream: stream.clone(),
            binding_targets,
            fault_extent: u64::from(shape.invocation_count().get()),
            scalar_fault_word: true,
            fault_law: None,
            checked_arithmetic: false,
        })
    }

    /// Validate and open one discovered device for owned asynchronous Dispatch.
    ///
    /// The code-generation target is detected from the selected device's CUDA compute capability. Library-backed tensor work can use this session without it;
    /// Dispatch preparation uses NVRTC when an architecture or `nvcc` is unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error when the device reference, block size, or runtime identity is invalid, or
    /// when CUDA cannot reopen the selected device.
    pub fn open(
        discovery: &CudaDiscovery,
        device: PcuObjectRef,
        block_size: u32,
    ) -> Result<Self, CudaOwnedDispatchError> {
        if block_size == 0 {
            return Err(CudaOwnedDispatchError::InvalidBlockSize);
        }
        if device.kind != PcuObjectKind::Device {
            return Err(CudaOwnedDispatchError::InvalidDeviceReference);
        }
        let identity = PcuDeviceIdentity::from_device_ref(device)
            .ok_or(CudaOwnedDispatchError::InvalidDeviceReference)?;
        let compiler = discovery.dispatch_compiler(device).ok();
        let runtime = discovery.open_device(device)?;
        let runtime_info = runtime.device_info()?;
        let actual = u32::try_from(runtime_info.index)
            .map_err(|_| CudaOwnedDispatchError::InvalidDeviceReference)?;
        if actual != identity.device_id() {
            return Err(CudaOwnedDispatchError::RuntimeDeviceMismatch {
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

    /// Allocate a buffer in the exact CUDA runtime session accepted by this adapter.
    ///
    /// # Errors
    ///
    /// Returns the CUDA allocation error when the device cannot allocate the requested size.
    pub fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        self.runtime.allocate(bytes)
    }

    /// Create a memory provider for this same opened CUDA device and its caller-assigned pool.
    #[must_use]
    pub fn memory_provider(&self, pool: fusion_pcu::PcuMemoryPoolId) -> CudaMemoryProvider {
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
    ) -> Result<PcuOwnedBinding<DeviceBuffer>, CudaOwnedDispatchError> {
        self.runtime
            .ensure_same_runtime(&resource.allocation.runtime)
            .map_err(|_| CudaOwnedDispatchError::DifferentRuntime(target))?;
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
    /// identity and CUDA runtime. It may outlive this backend value, but every submission remains
    /// tied to that same runtime/device. Keep it alive across warm launches to avoid repeating
    /// source lowering, code-object compilation, module loading, function lookup, stream creation,
    /// and ABI binding-order construction.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel profile cannot be lowered or CUDA cannot compile/load it.
    pub fn prepare_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        self.prepare_dispatch_ir(*submission.kernel, submission.shape)
    }

    /// Prepares a tensor-owned kernel on the tensor assessor's selected ordered stream.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_dispatch_owned_kernel_on_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        self.prepare_dispatch_on_stream(kernel, shape, stream)
    }

    /// Prepare a Dispatch executable on a caller-selected stream from this runtime.
    ///
    /// This lets callers build an ordered batch from multiple prepared kernels while preserving
    /// the batch stream identity. The stream must belong to this backend's CUDA runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the stream belongs to another runtime, or if lowering or compilation
    /// of the kernel fails.
    pub fn prepare_dispatch_on_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        self.prepare_dispatch_ir_with_stream(kernel, shape, stream.clone())
    }

    /// Prepares a dynamically generated tensor kernel with NVRTC as the first compiler choice.
    ///
    /// NVRTC compiles against the selected runtime device and avoids launching a `nvcc` process
    /// for short-lived generated tensor programs. When discovery selected `nvcc` as available,
    /// a NVRTC failure falls back to that compiler. This preference is deliberately local to the
    /// tensor-generated executable and does not change general Dispatch compiler selection.
    #[cfg(feature = "tensor")]
    pub(crate) fn prepare_dynamic_tensor_kernel_on_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: &crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        if !stream.belongs_to_runtime(&self.runtime) {
            return Err(CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime));
        }
        self.prepare_dispatch_ir_with_stream_preference(kernel, shape, stream.clone(), true)
    }

    fn prepare_dispatch_ir(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        let validated = self.validate_dispatch_ir(kernel, shape)?;
        let stream = self.runtime.create_stream()?;
        self.prepare_dispatch_ir_after_validation(kernel, shape, stream, false, validated)
    }

    fn prepare_dispatch_ir_with_stream(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: crate::CudaStreamHandle,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        self.prepare_dispatch_ir_with_stream_preference(kernel, shape, stream, false)
    }

    fn prepare_dispatch_ir_with_stream_preference(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
        stream: crate::CudaStreamHandle,
        prefer_cudartc: bool,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        let validated = self.validate_dispatch_ir(kernel, shape)?;
        self.prepare_dispatch_ir_after_validation(kernel, shape, stream, prefer_cudartc, validated)
    }

    fn validate_dispatch_ir(
        &self,
        kernel: fusion_pcu::PcuDispatchKernelIr<'_>,
        shape: fusion_pcu::PcuInvocationShape,
    ) -> Result<ValidatedDispatch, CudaOwnedDispatchError> {
        if !crate::admission::checked_numeric_contract(&kernel) {
            return Err(CudaOwnedDispatchError::UnsupportedRequirements);
        }
        let source = lower_dispatch_to_cuda_source(&kernel)?;
        if kernel_uses_checked_arithmetic(&kernel) && checked_scalar_fault_law(&kernel).is_none() {
            return Err(CudaOwnedDispatchError::UnsupportedRequirements);
        }
        let logical_invocations = shape.invocation_count().get();
        if kernel.entry.logical_shape != [logical_invocations, 1, 1] {
            return Err(CudaOwnedDispatchError::Lower(
                CudaLowerError::InvalidKernelShape,
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
        stream: crate::CudaStreamHandle,
        prefer_cudartc: bool,
        validated: ValidatedDispatch,
    ) -> Result<CudaPreparedDispatch, CudaOwnedDispatchError> {
        let ValidatedDispatch {
            source,
            grid_x,
            checked_arithmetic,
        } = validated;
        let compiler = self
            .compiler
            .ok_or(CudaOwnedDispatchError::CompilerUnavailable)?;
        let image = if prefer_cudartc {
            let rtc_image = lower_dispatch_to_cuda_rtc_source(&kernel)
                .map_err(CudaOwnedDispatchError::Lower)
                .and_then(|rtc_source| {
                    crate::compile_cuda_source_for_device(&self.runtime, &rtc_source)
                        .map_err(CudaOwnedDispatchError::CudaRtc)
                });
            match rtc_image {
                Ok(image) => image,
                Err(_rtc_error) if compiler == crate::discovery::DispatchCompiler::Nvcc => {
                    let architecture = self
                        .architecture
                        .as_deref()
                        .ok_or(CudaError::MissingArchitecture)?;
                    compile_cuda_source(&source, architecture)?
                }
                Err(error) => return Err(error),
            }
        } else {
            match compiler {
                crate::discovery::DispatchCompiler::Nvcc => {
                    let architecture = self
                        .architecture
                        .as_deref()
                        .ok_or(CudaError::MissingArchitecture)?;
                    compile_cuda_source(&source, architecture)?
                }
                crate::discovery::DispatchCompiler::Nvrtc => {
                    let rtc_source = crate::lower_dispatch_to_cuda_rtc_source(&kernel)?;
                    crate::compile_cuda_source_for_device(&self.runtime, &rtc_source)
                        .map_err(CudaOwnedDispatchError::CudaRtc)?
                }
            }
        };
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let operands = crate::codegen::lower::map_binding_projection(&kernel);
        let active = |binding: &&fusion_pcu::PcuBinding<'_>| {
            operands.is_none_or(|schema| {
                schema.contains_output(binding.reference())
                    || schema.input_bindings().contains(&binding.reference())
            })
        };
        let binding_targets = kernel
            .bindings
            .iter()
            .filter(active)
            .map(|binding| PcuBindingRef::new(binding.set, binding.binding))
            .collect();
        let binding_requirements = kernel
            .bindings
            .iter()
            .filter(active)
            .map(|binding| {
                PcuOwnedBindingRequirement::from_verified_binding(
                    &kernel,
                    PcuBindingRef::new(binding.set, binding.binding),
                    shape,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(CudaOwnedDispatchError::Binding)?;
        Ok(CudaPreparedDispatch {
            runtime: self.runtime.clone(),
            device: self.device,
            binding_requirements,
            shape,
            grid_x,
            block_size: self.block_size,
            function,
            stream,
            binding_targets,
            fault_extent: u64::from(checked_fault_extent(&kernel)),
            scalar_fault_word: true,
            fault_law: checked_scalar_fault_law(&kernel).map(fault_law::Retained::Scalar),
            checked_arithmetic,
        })
    }
}

struct ValidatedDispatch {
    source: String,
    grid_x: u32,
    checked_arithmetic: bool,
}

/// Reusable compiled `CUDA` executable for one PCU Dispatch descriptor.
///
/// This is distinct from the core `PcuPreparedDispatch` assessment value: it owns the compiled
/// CUDA function, owned binding requirements, and a reusable stream needed to submit the same kernel
/// repeatedly. The source IR may be dropped after preparation.
pub struct CudaPreparedDispatch {
    runtime: CudaRuntime,
    device: PcuDeviceIdentity,
    binding_requirements: Vec<PcuOwnedBindingRequirement>,
    shape: fusion_pcu::PcuInvocationShape,
    grid_x: u32,
    block_size: u32,
    function: crate::CudaKernel,
    stream: crate::CudaStreamHandle,
    binding_targets: Vec<PcuBindingRef>,
    fault_extent: u64,
    scalar_fault_word: bool,
    fault_law: Option<fault_law::Retained>,
    checked_arithmetic: bool,
}

impl CudaPreparedDispatch {
    pub(super) const fn checked_fault_contract(&self) -> Option<CudaPreparedFaultContract> {
        if self.checked_arithmetic {
            Some(CudaPreparedFaultContract {
                extent: self.fault_extent,
                scalar_abi: self.scalar_fault_word,
                law: self.fault_law,
            })
        } else {
            None
        }
    }

    pub(crate) const fn requires_checked_fault_word(&self) -> bool {
        self.checked_arithmetic
    }

    /// Clone the stream captured by this executable for preparing related ordered work.
    #[must_use]
    pub fn stream_handle(&self) -> crate::CudaStreamHandle {
        self.stream.clone()
    }

    /// Clone this prepared executable's compiled CUDA kernel for low-level manual orchestration.
    ///
    /// The returned kernel keeps its module loaded. Direct launches remain unsafe because callers
    /// must supply the exact ABI and buffer access pattern documented by [`crate::CudaKernel`].
    #[must_use]
    pub fn cuda_kernel(&self) -> crate::CudaKernel {
        self.function.clone()
    }

    /// Return this executable's exact three-dimensional grid and block geometry.
    #[must_use]
    pub const fn launch_geometry(&self) -> ([u32; 3], [u32; 3]) {
        ([self.grid_x, 1, 1], [self.block_size, 1, 1])
    }

    /// Start a checked-terminal batch on this executable's captured stream.
    #[must_use]
    pub fn checked_batch(&self) -> CudaCheckedDispatchBatch {
        CudaCheckedDispatchBatch::new(&self.stream)
    }

    /// Allocate reusable status storage for checked submissions that are waited in sequence.
    ///
    /// This opt-in path requires sequential use. Its returned owner permits one in-flight
    /// submission, and `submit_and_wait` holds an exclusive borrow until CUDA completion and status
    /// readback finish. Public [`Self::submit`] keeps independent status storage per call and
    /// supports overlapping submissions. After a terminal success, this owner's observed sentinel
    /// is reused without another host-to-device reset. Faults and known prelaunch rejections
    /// require a reset before the next attempt. An uncertain CUDA failure poisons the owner because
    /// the status word may still be in use by the device.
    ///
    /// # Errors
    ///
    /// Returns [`CudaOwnedDispatchError::CheckedArithmeticRequired`] for an unchecked executable,
    /// or a CUDA allocation/initialization error.
    pub fn sequential_checked(
        &self,
    ) -> Result<CudaSequentialCheckedDispatch<'_>, CudaOwnedDispatchError> {
        if !self.checked_arithmetic {
            return Err(CudaOwnedDispatchError::CheckedArithmeticRequired);
        }
        let mut fault_word = self.runtime.allocate(core::mem::size_of::<u64>())?;
        fault_word.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
        Ok(CudaSequentialCheckedDispatch {
            dispatch: self,
            fault_word,
            state: FaultWordState::Sentinel,
            poisoned: false,
        })
    }

    /// Submit this executable with a fresh set of owned bindings.
    ///
    /// Binding metadata, allocation size, and captured runtime/device identity are checked on
    /// every call. A stale or unavailable device is reported through CUDA operation failures; this
    /// warm path does not query a fresh device snapshot. CUDA launch argument storage, access
    /// leases, and completion events remain per launch.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid bindings, a mismatched runtime/device identity, or CUDA launch
    /// failure. A CUDA error after enqueue is conservatively handled by the CUDA launch contract.
    pub fn submit(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<CudaOwnedCompletion, CudaOwnedDispatchError> {
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
    /// before reusing the supplied storage. CUDA allocation access gates reject an overlapping
    /// launch through another clone. The completion retains its own allocation lease, so uncertain
    /// waits keep the storage alive alongside the caller's owner.
    /// Public `submit` remains independently safe for overlapping submissions by allocating one
    /// status word per completion.
    #[allow(clippy::needless_pass_by_ref_mut)] // Typed wrappers hold exclusive status ownership and wait before reuse.
    pub(crate) fn submit_with_fault_word(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        fault_word: &mut DeviceBuffer,
    ) -> Result<CudaOwnedCompletion, CudaOwnedDispatchError> {
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
    ) -> Result<CudaOwnedCompletion, CudaOwnedDispatchError> {
        if !self.checked_arithmetic {
            return Err(CudaOwnedDispatchError::CheckedArithmeticRequired);
        }
        self.validate_bindings(bindings)?;
        if fault_word.len() != core::mem::size_of::<u64>() {
            return Err(CudaOwnedDispatchError::CheckedFaultWordSize {
                actual: fault_word.len(),
            });
        }
        self.runtime
            .ensure_same_runtime(&fault_word.allocation.runtime)
            .map_err(|_| CudaError::DifferentRuntime)?;
        self.submit_validated(bindings, Some(fault_word.clone()), reset_fault_word)
    }

    fn validate_bindings(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), CudaOwnedDispatchError> {
        validate_owned_binding_requirements(&self.binding_requirements, self.device, bindings)
            .map_err(CudaOwnedDispatchError::Binding)?;
        for binding in bindings {
            let actual = binding.resource.len();
            if binding.byte_len != actual as u64 {
                return Err(CudaOwnedDispatchError::BufferSizeMismatch {
                    binding: binding.target,
                    metadata: binding.byte_len,
                    actual,
                });
            }
            self.runtime
                .ensure_same_runtime(&binding.resource.allocation.runtime)
                .map_err(|_| CudaOwnedDispatchError::DifferentRuntime(binding.target))?;
        }
        Ok(())
    }

    fn submit_validated(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        mut fault_word: Option<DeviceBuffer>,
        reset_fault_word: bool,
    ) -> Result<CudaOwnedCompletion, CudaOwnedDispatchError> {
        // Kernel arguments are borrowed only during `launch`; CUDA copies their pointer values
        // into owned aligned storage before returning. Keep common small interfaces on the stack
        // without constraining larger kernels to an arbitrary binding-count limit.
        if self.checked_arithmetic && reset_fault_word {
            let buffer = fault_word
                .as_mut()
                .ok_or(CudaOwnedDispatchError::CheckedBatchFaultWordUnavailable)?;
            buffer.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
        }
        let argument_count = self.binding_targets.len() + usize::from(fault_word.is_some());
        let mut inline_arguments: [CudaKernelArgument<'_>; INLINE_ARGUMENTS] =
            std::array::from_fn(|_| CudaKernelArgument::Bytes(&[]));
        let mut overflow_arguments = Vec::new();
        let arguments: &[CudaKernelArgument<'_>] = if argument_count <= INLINE_ARGUMENTS {
            for (slot, target) in inline_arguments
                .iter_mut()
                .zip(self.binding_targets.iter().copied())
            {
                let binding =
                    find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
                *slot = CudaKernelArgument::Buffer(&binding.resource);
            }
            if let Some(buffer) = fault_word.as_ref() {
                inline_arguments[self.binding_targets.len()] = CudaKernelArgument::Buffer(buffer);
            }
            &inline_arguments[..argument_count]
        } else {
            overflow_arguments.reserve(argument_count);
            for target in self.binding_targets.iter().copied() {
                let binding =
                    find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
                overflow_arguments.push(CudaKernelArgument::Buffer(&binding.resource));
            }
            if let Some(buffer) = fault_word.as_ref() {
                overflow_arguments.push(CudaKernelArgument::Buffer(buffer));
            }
            &overflow_arguments
        };

        // SAFETY: lowering validates one typed scalar pointer per declared binding in
        // declaration order (f32 map or u32 identity-copy profile). Core admission validates full binding coverage,
        // device metadata, access, and type. This adapter additionally verifies actual allocation
        // length and CUDA runtime identity, and CudaKernel::launch acquires the shared exclusive
        // allocation gates and retains module, stream, and allocations through event completion.
        let cuda = unsafe {
            self.function.launch(
                &self.stream,
                [self.grid_x, 1, 1],
                [self.block_size, 1, 1],
                0,
                arguments,
            )?
        };
        Ok(CudaOwnedCompletion {
            cuda: Some(cuda),
            fault_word: fault_word.take(),
            terminal: None,
            fault_extent: self.fault_extent,
            scalar_fault_word: self.scalar_fault_word,
            fault_law: self.fault_law,
        })
    }

    /// Submit this executable directly into an ordered CUDA completion batch.
    ///
    /// This path avoids creating a per-launch CUDA event. The batch must use the stream captured
    /// by this prepared dispatch and must be finished after the final queued operation. Checked
    /// `DivRem` is rejected because its fault word is observed only after completion; subsequent
    /// queued work could otherwise consume invalid arithmetic results before the fault
    /// becomes visible. If this method returns a CUDA launch error, the batch is poisoned and must
    /// be dropped; its drop path synchronizes the stream or quarantines its retained resources.
    ///
    /// # Errors
    /// Returns an error for invalid bindings, a mismatched runtime/device/stream, a poisoned
    /// batch, or CUDA launch failure.
    pub fn submit_into_batch(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        batch: &mut CudaCompletionBatch,
    ) -> Result<(), CudaOwnedDispatchError> {
        validate_batch_fault_semantics(self.checked_arithmetic)?;
        validate_owned_binding_requirements(&self.binding_requirements, self.device, bindings)
            .map_err(CudaOwnedDispatchError::Binding)?;
        for binding in bindings {
            let actual = binding.resource.len();
            if binding.byte_len != actual as u64 {
                return Err(CudaOwnedDispatchError::BufferSizeMismatch {
                    binding: binding.target,
                    metadata: binding.byte_len,
                    actual,
                });
            }
            self.runtime
                .ensure_same_runtime(&binding.resource.allocation.runtime)
                .map_err(|_| CudaOwnedDispatchError::DifferentRuntime(binding.target))?;
        }
        let mut inline_arguments: [CudaKernelArgument<'_>; INLINE_ARGUMENTS] =
            std::array::from_fn(|_| CudaKernelArgument::Bytes(&[]));
        let mut overflow_arguments = Vec::new();
        let arguments: &[CudaKernelArgument<'_>] = if self.binding_targets.len() <= INLINE_ARGUMENTS
        {
            for (slot, target) in inline_arguments
                .iter_mut()
                .zip(self.binding_targets.iter().copied())
            {
                let binding =
                    find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
                *slot = CudaKernelArgument::Buffer(&binding.resource);
            }
            &inline_arguments[..self.binding_targets.len()]
        } else {
            overflow_arguments.reserve(self.binding_targets.len());
            for target in self.binding_targets.iter().copied() {
                let binding =
                    find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
                overflow_arguments.push(CudaKernelArgument::Buffer(&binding.resource));
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

    pub(crate) fn submit_checked_into_batch(
        &self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
        batch: &mut CudaCompletionBatch,
        fault_word: &DeviceBuffer,
    ) -> Result<(), CudaOwnedDispatchError> {
        if !self.checked_arithmetic {
            return Err(CudaOwnedDispatchError::CheckedArithmeticRequired);
        }
        validate_owned_binding_requirements(&self.binding_requirements, self.device, bindings)
            .map_err(CudaOwnedDispatchError::Binding)?;
        for binding in bindings {
            let actual = binding.resource.len();
            if binding.byte_len != actual as u64 {
                return Err(CudaOwnedDispatchError::BufferSizeMismatch {
                    binding: binding.target,
                    metadata: binding.byte_len,
                    actual,
                });
            }
            self.runtime
                .ensure_same_runtime(&binding.resource.allocation.runtime)
                .map_err(|_| CudaOwnedDispatchError::DifferentRuntime(binding.target))?;
        }
        self.runtime
            .ensure_same_runtime(&fault_word.allocation.runtime)
            .map_err(|_| CudaOwnedDispatchError::Cuda(CudaError::DifferentRuntime))?;
        let argument_count = self.binding_targets.len() + 1;
        let mut inline_arguments: [CudaKernelArgument<'_>; INLINE_ARGUMENTS] =
            std::array::from_fn(|_| CudaKernelArgument::Bytes(&[]));
        let mut overflow_arguments = Vec::new();
        let arguments: &[CudaKernelArgument<'_>] = if argument_count <= INLINE_ARGUMENTS {
            for (slot, target) in inline_arguments
                .iter_mut()
                .zip(self.binding_targets.iter().copied())
            {
                let binding =
                    find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
                *slot = CudaKernelArgument::Buffer(&binding.resource);
            }
            inline_arguments[self.binding_targets.len()] = CudaKernelArgument::Buffer(fault_word);
            &inline_arguments[..argument_count]
        } else {
            overflow_arguments.reserve(argument_count);
            for target in self.binding_targets.iter().copied() {
                let binding =
                    find_binding(target, bindings).map_err(CudaOwnedDispatchError::Binding)?;
                overflow_arguments.push(CudaKernelArgument::Buffer(&binding.resource));
            }
            overflow_arguments.push(CudaKernelArgument::Buffer(fault_word));
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
pub struct CudaSequentialCheckedDispatch<'a> {
    dispatch: &'a CudaPreparedDispatch,
    fault_word: DeviceBuffer,
    state: FaultWordState,
    poisoned: bool,
}

impl CudaSequentialCheckedDispatch<'_> {
    /// Submit a checked dispatch and wait for its terminal status before returning.
    ///
    /// # Errors
    ///
    /// Returns validation, CUDA launch/wait/readback, or invalid checked-status errors. Uncertain
    /// launch/wait/readback errors poison this owner; a terminal arithmetic fault is returned as
    /// `Ok(PcuCompletionOutcome::Fault(_))` so callers may recover and submit again.
    pub fn submit_and_wait(
        &mut self,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<PcuCompletionOutcome, CudaOwnedDispatchError> {
        if self.poisoned {
            return Err(CudaOwnedDispatchError::CheckedSequentialDispatchPoisoned);
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
                return Err(CudaOwnedDispatchError::Cuda(error));
            }
        };
        self.state = FaultWordState::after_terminal(outcome);
        Ok(outcome)
    }
}

const fn is_certain_checked_prelaunch_error(error: &CudaOwnedDispatchError) -> bool {
    matches!(
        error,
        CudaOwnedDispatchError::Binding(_)
            | CudaOwnedDispatchError::BufferSizeMismatch { .. }
            | CudaOwnedDispatchError::DifferentRuntime(_)
            | CudaOwnedDispatchError::MemoryAccessMismatch(_)
            | CudaOwnedDispatchError::CheckedFaultWordSize { .. }
    )
}

/// Ordered `CUDA` batch that may end in one checked `DivRem` dispatch.
///
/// Unchecked dispatches may be appended before the checked dispatch. Once the checked dispatch
/// is submitted this wrapper exposes no path to enqueue later work. The caller must also avoid
/// externally enqueueing dependent work on the same stream until `finish`'s completion has been
/// waited and its checked outcome observed; this wrapper cannot constrain other stream users.
pub struct CudaCheckedDispatchBatch {
    batch: Option<CudaCompletionBatch>,
    checked_attempted: bool,
    checked_submitted: bool,
    fault_word: Option<DeviceBuffer>,
    fault_extent: u64,
    scalar_fault_word: bool,
    fault_law: Option<fault_law::Retained>,
}

impl CudaCheckedDispatchBatch {
    /// Start an ordered batch on `stream`.
    #[must_use]
    pub fn new(stream: &crate::CudaStreamHandle) -> Self {
        Self {
            batch: Some(CudaCompletionBatch::new(stream)),
            checked_attempted: false,
            checked_submitted: false,
            fault_word: None,
            fault_extent: 0,
            scalar_fault_word: true,
            fault_law: None,
        }
    }

    /// Append an unchecked dispatch before the checked terminal dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error if the checked dispatch was already attempted, if the CUDA batch is no
    /// longer open, or if binding validation or CUDA launch fails.
    pub fn submit_unchecked(
        &mut self,
        dispatch: &CudaPreparedDispatch,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), CudaOwnedDispatchError> {
        if self.checked_attempted {
            return Err(CudaOwnedDispatchError::CheckedArithmeticBatchClosed);
        }
        let batch = self
            .batch
            .as_mut()
            .ok_or(CudaOwnedDispatchError::CheckedBatchUnavailable)?;
        dispatch.submit_into_batch(bindings, batch)
    }

    /// Append the checked `DivRem` dispatch as the final launch in this wrapper's batch.
    ///
    /// Its fault word stays owned across event synchronization and subsequent device readback.
    /// If synchronization or readback fails, the returned completion retains the necessary
    /// owners for a retry. Enqueue failures poison the underlying CUDA batch, whose drop path
    /// synchronizes or quarantines every retained allocation. The caller must not enqueue
    /// external work that depends on the checked outputs until this batch's result has been read.
    ///
    /// # Errors
    ///
    /// Returns an error if the executable is not checked `DivRem`, this wrapper is already closed,
    /// allocation or initialization fails, bindings are invalid, or CUDA launch fails.
    pub fn submit_checked_last(
        &mut self,
        dispatch: &CudaPreparedDispatch,
        bindings: &[PcuOwnedBinding<DeviceBuffer>],
    ) -> Result<(), CudaOwnedDispatchError> {
        if self.checked_attempted {
            return Err(CudaOwnedDispatchError::CheckedArithmeticBatchClosed);
        }
        if !dispatch.checked_arithmetic {
            return Err(CudaOwnedDispatchError::CheckedArithmeticRequired);
        }
        let mut fault_word = dispatch.runtime.allocate(core::mem::size_of::<u64>())?;
        fault_word.copy_from(&FAULT_WORD_SENTINEL.to_le_bytes())?;
        // Prevent another enqueue attempt before entering the substrate. A CUDA enqueue failure
        // poisons the batch, so this wrapper is closed even when submission returns an error.
        // Store the allocation first; on enqueue failure the wrapper keeps the owner until its
        // poisoned batch has synchronized or quarantined resources.
        self.checked_attempted = true;
        self.fault_word = Some(fault_word);
        let batch = self
            .batch
            .as_mut()
            .ok_or(CudaOwnedDispatchError::CheckedBatchUnavailable)?;
        let fault_word = self
            .fault_word
            .as_ref()
            .ok_or(CudaOwnedDispatchError::CheckedBatchFaultWordUnavailable)?;
        dispatch.submit_checked_into_batch(bindings, batch, fault_word)?;
        self.checked_submitted = true;
        self.fault_extent = dispatch.fault_extent;
        self.scalar_fault_word = dispatch.scalar_fault_word;
        self.fault_law = dispatch.fault_law;
        Ok(())
    }

    /// Finish the stream batch and return a completion that reports checked execution faults.
    ///
    /// # Errors
    ///
    /// Returns an error if no checked dispatch was successfully submitted or if CUDA cannot
    /// record the final completion event.
    pub fn finish(mut self) -> Result<CudaCheckedBatchCompletion, CudaOwnedDispatchError> {
        if !self.checked_submitted {
            return Err(CudaOwnedDispatchError::CheckedArithmeticRequired);
        }
        let batch = self
            .batch
            .as_mut()
            .ok_or(CudaOwnedDispatchError::CheckedBatchUnavailable)?;
        let cuda = batch.finish()?;
        self.batch.take();
        Ok(CudaCheckedBatchCompletion {
            cuda: Some(cuda),
            fault_word: self.fault_word.take(),
            terminal: None,
            fault_extent: self.fault_extent,
            scalar_fault_word: self.scalar_fault_word,
            fault_law: self.fault_law,
        })
    }
}

/// Retryable completion for a `CUDA` batch ending in checked `DivRem`.
pub struct CudaCheckedBatchCompletion {
    cuda: Option<crate::CudaBatchCompletion>,
    fault_word: Option<DeviceBuffer>,
    terminal: Option<PcuCompletionOutcome>,
    fault_extent: u64,
    scalar_fault_word: bool,
    fault_law: Option<fault_law::Retained>,
}

impl CudaCheckedBatchCompletion {
    /// Wait for all launches and read the terminal checked-arithmetic status.
    ///
    /// A returned fault means outputs produced by the checked dispatch are invalid. CUDA wait and
    /// readback errors retain this object's owners, so the caller may retry.
    ///
    /// # Errors
    ///
    /// Returns a CUDA wait/readback error, an invalid fault word error, or an invalid-state error
    /// if the fault-word owner is unavailable.
    pub fn wait(&mut self) -> Result<PcuCompletionOutcome, CudaOwnedDispatchError> {
        if let Some(terminal) = self.terminal {
            return Ok(terminal);
        }
        if let Some(cuda) = self.cuda.as_mut() {
            cuda.wait()?;
            self.cuda.take();
        }
        let fault_word = self
            .fault_word
            .as_ref()
            .ok_or(CudaOwnedDispatchError::CheckedBatchFaultWordUnavailable)?;
        let mut bytes = [0_u8; core::mem::size_of::<u64>()];
        fault_word.copy_to(&mut bytes)?;
        let outcome = decode_fault_word_under_law(
            u64::from_le_bytes(bytes),
            self.fault_extent,
            self.scalar_fault_word,
            self.fault_law,
        )?
        .map_or(PcuCompletionOutcome::Succeeded, PcuCompletionOutcome::Fault);
        self.fault_word.take();
        self.terminal = Some(outcome);
        Ok(outcome)
    }
}

impl PcuOwnedCompletion for CudaCheckedBatchCompletion {
    type Error = CudaOwnedDispatchError;

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

impl PcuBaseContract for CudaOwnedDispatchBackend {
    fn support(&self) -> PcuSupport {
        owned_dispatch_support()
    }

    fn executors(&self) -> &'static [PcuExecutorDescriptor] {
        &OWNED_EXECUTORS
    }
}

impl PcuOwnedDispatchBackend for CudaOwnedDispatchBackend {
    type Resource = DeviceBuffer;
    type Bindings = Vec<PcuOwnedBinding<DeviceBuffer>>;
    type Completion = CudaOwnedCompletion;
    type Error = CudaOwnedDispatchError;
    type Prepared<'kernel, 'parameters>
        = CudaPreparedDispatch
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
            return Err(CudaOwnedDispatchError::Lower(
                CudaLowerError::UnsupportedKernelInterface,
            ));
        }
        self.prepare_dispatch(submission)
    }
}

impl PcuPreparedOwnedDispatch for CudaPreparedDispatch {
    type Resource = DeviceBuffer;
    type Bindings = Vec<PcuOwnedBinding<DeviceBuffer>>;
    type Completion = CudaOwnedCompletion;
    type Error = CudaOwnedDispatchError;

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

impl PcuOwnedDispatchMemorySession for CudaOwnedDispatchBackend {
    type MemoryProvider = CudaMemoryProvider;

    fn memory_provider(&self, pool: fusion_pcu::PcuMemoryPoolId) -> Self::MemoryProvider {
        Self::memory_provider(self, pool)
    }

    fn bind(
        &self,
        target: PcuBindingRef,
        access: PcuBindingAccess,
        binding_type: PcuBindingType,
        resource: &CudaMemoryResource,
    ) -> Result<PcuOwnedBinding<Self::Resource>, Self::Error> {
        if !memory_access_supports_binding(resource.access(), access) {
            return Err(CudaOwnedDispatchError::MemoryAccessMismatch(target));
        }
        let buffer = resource.device_buffer();
        self.runtime
            .ensure_same_runtime(&buffer.allocation.runtime)
            .map_err(|_| CudaOwnedDispatchError::DifferentRuntime(target))?;
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

/// Core completion contract over the CUDA event-backed completion token.
pub struct CudaOwnedCompletion {
    cuda: Option<CudaCompletion>,
    fault_word: Option<DeviceBuffer>,
    terminal: Option<PcuCompletionOutcome>,
    fault_extent: u64,
    scalar_fault_word: bool,
    fault_law: Option<fault_law::Retained>,
}

impl CudaOwnedCompletion {
    pub(crate) fn can_handoff_to_batch(
        &self,
        batch: &crate::CudaCompletionBatch,
    ) -> Result<bool, CudaError> {
        if self.fault_word.is_some() || self.terminal.is_some() {
            return Ok(false);
        }
        self.cuda
            .as_ref()
            .map_or(Ok(false), |completion| batch.can_wait_for(completion))
    }

    pub(crate) const fn take_cuda_for_handoff(&mut self) -> Option<CudaCompletion> {
        if self.fault_word.is_none() && self.terminal.is_none() {
            self.cuda.take()
        } else {
            None
        }
    }
}

impl PcuOwnedCompletion for CudaOwnedCompletion {
    type Error = CudaError;

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
        let cuda = self
            .cuda
            .as_mut()
            .expect("nonterminal completion retains CUDA token");
        cuda.wait()?;
        let outcome = if let Some(fault_word) = self.fault_word.as_ref() {
            let mut bytes = [0_u8; core::mem::size_of::<u64>()];
            fault_word.copy_to(&mut bytes)?;
            let word = u64::from_le_bytes(bytes);
            decode_fault_word_under_law(
                word,
                self.fault_extent,
                self.scalar_fault_word,
                self.fault_law,
            )?
            .map_or(PcuCompletionOutcome::Succeeded, PcuCompletionOutcome::Fault)
        } else {
            PcuCompletionOutcome::Succeeded
        };
        self.cuda.take();
        self.fault_word.take();
        self.terminal = Some(outcome);
        Ok(outcome)
    }
}

fn launch_grid(invocations: u32, block_size: u32) -> Result<u32, CudaOwnedDispatchError> {
    if block_size == 0 {
        return Err(CudaOwnedDispatchError::InvalidBlockSize);
    }
    let grid = u64::from(invocations).div_ceil(u64::from(block_size));
    if grid * u64::from(block_size) > u64::from(u32::MAX) {
        return Err(CudaOwnedDispatchError::GeometryOverflow);
    }
    u32::try_from(grid).map_err(|_| CudaOwnedDispatchError::GeometryOverflow)
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

// Cold typed arithmetic offers are shared by discovery and executor descriptors.
const fn scalar_alu_support() -> fusion_pcu::PcuDispatchScalarAluSupport {
    fusion_pcu::PcuDispatchScalarAluSupport::empty()
        .with(fusion_pcu::PcuScalarType::F32, f32_alu_caps())
        .with(
            fusion_pcu::PcuScalarType::F16,
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
        )
        .with(
            fusion_pcu::PcuScalarType::BF16,
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
        )
        .with(
            fusion_pcu::PcuScalarType::F8E4M3FN,
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
        )
        .with(
            fusion_pcu::PcuScalarType::F8E5M2,
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY),
        )
        .with(fusion_pcu::PcuScalarType::F64, f64_alu_caps())
        .with(fusion_pcu::PcuScalarType::U32, checked_u32_alu_caps())
        .with(fusion_pcu::PcuScalarType::U16, checked_u16_alu_caps())
        .with(fusion_pcu::PcuScalarType::I16, checked_i16_alu_caps())
        .with(fusion_pcu::PcuScalarType::U8, checked_u8_alu_caps())
        .with(fusion_pcu::PcuScalarType::I8, checked_i8_alu_caps())
        .with(fusion_pcu::PcuScalarType::I32, checked_i32_alu_caps())
        .with(fusion_pcu::PcuScalarType::U64, checked_u64_alu_caps())
        .with(fusion_pcu::PcuScalarType::I64, checked_i64_alu_caps())
        .with(
            fusion_pcu::PcuScalarType::I128,
            PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        )
        .with(
            fusion_pcu::PcuScalarType::U128,
            PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        )
        .with(
            fusion_pcu::PcuScalarType::I256,
            PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        )
        .with(
            fusion_pcu::PcuScalarType::U256,
            PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        )
        .with(
            fusion_pcu::PcuScalarType::I512,
            PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        )
        .with(
            fusion_pcu::PcuScalarType::U512,
            PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY
                .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM),
        )
}

// Storage transport is independent of arithmetic offers: all 22 sealed byte carriers.
const fn scalar_storage_caps() -> PcuValueTypeCaps {
    PcuValueTypeCaps::FLOAT32
        .union(PcuValueTypeCaps::FLOAT16)
        .union(PcuValueTypeCaps::BFLOAT16)
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::F8E4M3FN,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::F8E5M2,
        ))
        .union(PcuValueTypeCaps::FLOAT64)
        .union(PcuValueTypeCaps::INT8)
        .union(PcuValueTypeCaps::UINT8)
        .union(PcuValueTypeCaps::UINT16)
        .union(PcuValueTypeCaps::UINT32)
        .union(PcuValueTypeCaps::INT32)
        .union(PcuValueTypeCaps::INT16)
        .union(PcuValueTypeCaps::UINT64)
        .union(PcuValueTypeCaps::INT64)
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::I128,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::U128,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::I256,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::U256,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::I512,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::U512,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::F128,
        ))
        .union(PcuValueTypeCaps::for_scalar(
            fusion_pcu::PcuScalarType::F256,
        ))
        .union(PcuValueTypeCaps::SCALAR_VALUES)
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
    support.value_type_support =
        PcuFeatureSupport::new(scalar_storage_caps(), PcuValueTypeCaps::empty());
    let mut dispatch = PcuDispatchSupport::unsupported();
    dispatch.flags = PcuDispatchPolicyCaps::SERIAL.union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION);
    dispatch.instructions = PcuFeatureSupport::new(
        PcuDispatchOpCaps::VALUE_CONSTANT
            .union(PcuDispatchOpCaps::VALUE_CAST)
            .union(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
            .union(PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
            .union(PcuDispatchOpCaps::CONTROL_RETURN)
            .union(PcuDispatchOpCaps::CONTROL_LOOP)
            .union(PcuDispatchOpCaps::BINDING_LOAD)
            .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
            .union(PcuDispatchOpCaps::BINDING_STORE),
        PcuDispatchOpCaps::empty(),
    );
    dispatch.scalar_alu = PcuFeatureSupport::new(
        scalar_alu_support(),
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
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
    .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::CONTROL_LOOP)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_LOAD_ELEMENT_ZERO)
    .union(PcuDispatchOpCaps::BINDING_STORE);

const fn f32_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
}

const fn f64_alu_caps() -> PcuDispatchOpCaps {
    PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_CONVERT)
        .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY)
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
    name: "cuda-owned-dispatch",
    class: PcuExecutorClass::Compute,
    origin: PcuExecutorOrigin::TopologyBound,
    support: PcuExecutorSupport {
        primitives: PcuPrimitiveCaps::DISPATCH,
        dispatch_policy: PcuDispatchPolicyCaps::SERIAL
            .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION),
        value_types: scalar_storage_caps(),
        dispatch_instructions: OWNED_DISPATCH_INSTRUCTIONS,
        dispatch_scalar_alu: scalar_alu_support(),
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
        CudaOwnedDispatchError,
        CudaCheckedBatchCompletion,
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
    fn checked_batch_completion_implements_common_owned_completion_contract() {
        fn state<C: PcuOwnedCompletion>(completion: &C) -> PcuCompletionState {
            completion
                .state()
                .ok()
                .expect("terminal state is available")
        }

        let completion = CudaCheckedBatchCompletion {
            cuda: None,
            fault_word: None,
            fault_extent: 8,
            scalar_fault_word: true,
            fault_law: fusion_pcu::PcuCheckedScalarFaultLaw::integer_div_rem(
                fusion_pcu::PcuScalarType::I32,
            )
            .map(super::fault_law::Retained::Scalar),
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
            Err(CudaOwnedDispatchError::CheckedArithmeticBatchUnsupported)
        ));

        let result = PcuExecutionResourceUse {
            resource: PcuExecutionResourceId(0),
            access: PcuMemoryAccess::WriteOnly,
        };
        let consumed = PcuExecutionResourceUse {
            access: PcuMemoryAccess::ReadOnly,
            ..result
        };
        // An in-order CUDA stream establishes ordering, but a later launch must not read a
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
    fn established_status_codes_and_noncanonical_words_are_checked() {
        use fusion_pcu::PcuExecutionFaultKind;
        for (code, kind) in [
            (1, PcuExecutionFaultKind::DivideByZero),
            (2, PcuExecutionFaultKind::SignedDivisionOverflow),
            (3, PcuExecutionFaultKind::ArithmeticOverflow),
        ] {
            let fault = decode_fault_word((7 << 3) | code).unwrap().unwrap();
            assert_eq!(
                (fault.kind, fault.invocation_id, fault.recovered),
                (kind, 7, false)
            );
        }
        assert_eq!(decode_fault_word(u64::MAX).unwrap(), None);
        for word in [
            0,
            6,
            7,
            (1_u64 << 63) | 1,
            (1_u64 << 63) | 2,
            (1_u64 << 63) | 5,
        ] {
            assert!(decode_fault_word(word).is_err());
        }
        // Every admitted checked map is one-dimensional with a u32 logical extent.
        // Neither fatal nor recovered codes may turn an impossible index into a result.
        for code in 1..=5 {
            for recovered in [0, 1_u64 << 63] {
                let word = ((u64::from(u32::MAX) + 1) << 3) | code | recovered;
                let result = decode_fault_word(word);
                assert!(
                    result.is_err(),
                    "noncanonical status {word:#x} accepted: {result:?}"
                );
            }
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
            Err(CudaOwnedDispatchError::GeometryOverflow)
        ));
        assert!(matches!(
            launch_grid(10, 0),
            Err(CudaOwnedDispatchError::InvalidBlockSize)
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
                .contains(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY)
        );
    }

    #[test]
    fn owned_dispatch_rejects_unchecked_f32_max() {
        let support = owned_dispatch_support();
        assert!(
            !support
                .dispatch_support
                .scalar_alu
                .direct
                .for_scalar(fusion_pcu::PcuScalarType::F32)
                .contains(PcuDispatchOpCaps::ALU_MAX)
        );
        assert!(
            !support
                .dispatch_support
                .instructions
                .direct
                .contains(PcuDispatchOpCaps::ALU_MAX)
        );
        assert!(
            !OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::F32)
                .contains(PcuDispatchOpCaps::ALU_MAX)
        );
        assert!(
            !OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_MAX)
        );
        assert!(
            !OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::F64)
                .contains(PcuDispatchOpCaps::ALU_MIN)
        );
        assert!(
            !OWNED_EXECUTORS[0]
                .support
                .dispatch_instructions
                .contains(PcuDispatchOpCaps::ALU_MIN)
        );
    }

    #[test]
    #[allow(clippy::cognitive_complexity)]
    fn owned_dispatch_advertises_only_supported_u32_alu_operations() {
        let supported = PcuDispatchOpCaps::ALU_CHECKED_INTEGER_BINARY;
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
        assert!(u16_support.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
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
        assert!(i16_support.contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM));
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
            OWNED_EXECUTORS[0]
                .support
                .dispatch_scalar_alu
                .for_scalar(fusion_pcu::PcuScalarType::I64)
                .contains(PcuDispatchOpCaps::ALU_CHECKED_DIV_REM)
        );
    }

    #[test]
    fn owned_dispatch_advertises_only_checked_low_float_binary_and_unary() {
        let expected = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY
            .union(PcuDispatchOpCaps::ALU_CHECKED_FLOAT_UNARY);
        for scalar in [
            fusion_pcu::PcuScalarType::F16,
            fusion_pcu::PcuScalarType::BF16,
            fusion_pcu::PcuScalarType::F8E4M3FN,
            fusion_pcu::PcuScalarType::F8E5M2,
        ] {
            assert_eq!(
                owned_dispatch_support()
                    .dispatch_support
                    .scalar_alu
                    .direct
                    .for_scalar(scalar),
                expected
            );
            assert_eq!(
                OWNED_EXECUTORS[0]
                    .support
                    .dispatch_scalar_alu
                    .for_scalar(scalar),
                expected
            );
        }
    }

    #[test]
    fn owned_dispatch_advertises_checked_float_binary_for_f32_and_f64() {
        let checked = PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY;
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
    ) -> Result<(), CudaOwnedDispatchError> {
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
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
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
            numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
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
        let f32_without_conversion =
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY.union(PcuDispatchOpCaps::ALU_MAX);
        let f64_without_conversion =
            PcuDispatchOpCaps::ALU_CHECKED_FLOAT_BINARY.union(PcuDispatchOpCaps::ALU_MAX);
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
                        Err(CudaOwnedDispatchError::UnsupportedRequirements)
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
                    Err(CudaOwnedDispatchError::UnsupportedRequirements)
                ));
                assert!(matches!(
                    checked_conversion_kernel_is_supported(conversion, grid, &missing_feature),
                    Err(CudaOwnedDispatchError::UnsupportedRequirements)
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

#[path = "execution.rs"]
mod execution;
#[rustfmt::skip]
pub use execution::{
    CudaExecutionStep,
    CudaOwnedExecution,
    CudaOwnedExecutionError,
    CudaOwnedExecutionNode,
    CudaOwnedExecutionOperation,
    CudaOwnedExecutionTwoSlot,
    CudaTwoSlotExecutionStep,


};

#[cfg(test)]
#[path = "checked_integer_tests.rs"]
mod checked_integer_tests;

#[cfg(test)]
#[path = "checked_float_tests.rs"]
mod checked_float_tests;

#[cfg(test)]
#[path = "checked_conversion_tests.rs"]
mod checked_conversion_tests;

#[path = "fault_word.rs"]
mod fault_word;
#[allow(clippy::redundant_pub_crate)]
// Keep the state machine internal even if this module is exposed later.
pub(crate) use fault_word::FaultWordState;

#[cfg(test)]
#[path = "checked_unary_tests.rs"]
mod checked_unary_tests;

#[cfg(test)]
mod fault_extent_tests {
    use super::decode_fault_word;
    #[test]
    fn checked_fault_domain_uses_actual_logical_extent() {
        for extent in [4, 65] {
            for tag in 1..=5 {
                assert!(
                    super::decode_fault_word_in_extent((u64::from(extent - 1) << 3) | tag, extent)
                        .is_ok()
                );
                assert!(
                    super::decode_fault_word_in_extent((u64::from(extent) << 3) | tag, extent)
                        .is_err()
                );
                assert!(
                    super::decode_fault_word_in_extent(
                        ((u64::from(extent) + 7) << 3) | tag,
                        extent
                    )
                    .is_err()
                );
            }
            for tag in [3, 4] {
                assert!(
                    super::decode_fault_word_in_extent(
                        (1 << 63) | (u64::from(extent) << 3) | tag,
                        extent
                    )
                    .is_err()
                );
            }
            assert_eq!(
                super::decode_fault_word_in_extent(u64::MAX, extent).unwrap(),
                None
            );
        }
        assert!(decode_fault_word((u64::from(u32::MAX) << 3) | 3).is_ok());
    }
}

#[cfg(all(test, feature = "tensor"))]
#[path = "fault_law_tests.rs"]
mod scalar_fault_law_tests;
