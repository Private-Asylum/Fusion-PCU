//! Synchronous, bounded PCU Dispatch execution on one `CUDA` device.

#[rustfmt::skip]
use std::{
    error::Error,
    fmt,
    mem::size_of,
};

#[rustfmt::skip]
use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchKernelIr,
};

#[rustfmt::skip]
use crate::{
    DeviceBuffer,
    CudaCompileError,
    CudaError,
    CudaKernelArgument,
    CudaRuntime,
    CudaLowerError,
    compile_cuda_source,
    lower_dispatch_to_cuda_source,
};

/// An already allocated device buffer bound to a PCU resource slot.
pub struct CudaDispatchBinding<'a> {
    pub id: PcuBindingRef,
    pub buffer: &'a DeviceBuffer,
}

#[derive(Debug)]
pub enum CudaDispatchError {
    Lower(CudaLowerError),
    Compile(CudaCompileError),
    CudaRtc(crate::CudaRtcError),
    Cuda(CudaError),
    ExecutionFault(fusion_pcu::PcuExecutionFault),
    CompilerUnavailable,
    MissingBinding(PcuBindingRef),
    DuplicateBinding(PcuBindingRef),
    UnexpectedBinding(PcuBindingRef),
    BufferTooSmall {
        binding: PcuBindingRef,
        required: usize,
        available: usize,
    },
    InvalidBlockSize,
    GeometryOverflow,
}

impl fmt::Display for CudaDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lower(error) => write!(f, "PCU Dispatch cannot lower to CUDA: {error}"),
            Self::Compile(error) => write!(f, "CUDA code object compilation failed: {error}"),
            Self::CudaRtc(error) => write!(f, "CUDA runtime compilation failed: {error}"),
            Self::Cuda(error) => write!(f, "CUDA execution failed: {error}"),
            Self::ExecutionFault(fault) => write!(f, "CUDA numerical fault: {fault:?}"),
            Self::CompilerUnavailable => {
                f.write_str("no usable CUDA source compiler is available for this device")
            }
            Self::MissingBinding(id) => write!(f, "missing device binding {id:?}"),
            Self::DuplicateBinding(id) => write!(f, "duplicate device binding {id:?}"),
            Self::UnexpectedBinding(id) => write!(f, "unexpected device binding {id:?}"),
            Self::BufferTooSmall {
                binding,
                required,
                available,
            } => write!(
                f,
                "device binding {binding:?} needs {required} bytes but has {available}"
            ),
            Self::InvalidBlockSize => f.write_str("CUDA block size must be nonzero"),
            Self::GeometryOverflow => f.write_str("CUDA launch geometry or buffer size overflow"),
        }
    }
}

impl Error for CudaDispatchError {}

impl From<CudaLowerError> for CudaDispatchError {
    fn from(error: CudaLowerError) -> Self {
        Self::Lower(error)
    }
}
impl From<CudaCompileError> for CudaDispatchError {
    fn from(error: CudaCompileError) -> Self {
        Self::Compile(error)
    }
}
impl From<CudaError> for CudaDispatchError {
    fn from(error: CudaError) -> Self {
        Self::Cuda(error)
    }
}

/// Compile and execute the supported f32 PCU Dispatch subset, waiting for GPU completion.
///
/// All bindings must be device allocations sized for the kernel's logical invocation count.
/// The generated kernel bounds-checks rounded-up workgroups. This path deliberately completes
/// synchronously so caller-owned buffers cannot be dropped or modified during execution.
///
/// # Errors
///
/// Returns structured lowering, binding, compilation, launch, or completion errors.
pub fn execute_pcu_dispatch(
    runtime: &CudaRuntime,
    kernel: &PcuDispatchKernelIr<'_>,
    architecture: &str,
    block_size: u32,
    bindings: &[CudaDispatchBinding<'_>],
) -> Result<(), CudaDispatchError> {
    if !crate::admission::checked_numeric_contract(kernel)
        || kernel.bindings.iter().any(|binding| {
            binding.binding_type
                != fusion_pcu::PcuBindingType::Value(fusion_pcu::PcuValueType::f32())
        })
    {
        return Err(CudaLowerError::UnsupportedRequirements.into());
    }
    let source = lower_dispatch_to_cuda_source(kernel)?;
    if block_size == 0 {
        return Err(CudaDispatchError::InvalidBlockSize);
    }
    let invocations = kernel.entry.logical_shape[0];
    let required = usize::try_from(invocations)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or(CudaDispatchError::GeometryOverflow)?;

    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index].iter().any(|prior| prior.id == binding.id) {
            return Err(CudaDispatchError::DuplicateBinding(binding.id));
        }
        if !kernel.bindings.iter().any(|declared| {
            declared.set == binding.id.set && declared.binding == binding.id.binding
        }) {
            return Err(CudaDispatchError::UnexpectedBinding(binding.id));
        }
    }

    let mut ordered = Vec::with_capacity(kernel.bindings.len());
    for declared in kernel.bindings {
        let id = PcuBindingRef::new(declared.set, declared.binding);
        let buffer = bindings
            .iter()
            .find(|candidate| candidate.id == id)
            .ok_or(CudaDispatchError::MissingBinding(id))?
            .buffer;
        if buffer.len() < required {
            return Err(CudaDispatchError::BufferTooSmall {
                binding: id,
                required,
                available: buffer.len(),
            });
        }
        ordered.push(CudaKernelArgument::Buffer(buffer));
    }

    let fault_word = if crate::owned_dispatch::kernel_uses_checked_arithmetic(kernel) {
        let mut word = runtime.allocate(size_of::<u64>())?;
        word.copy_from(&u64::MAX.to_ne_bytes())?;
        Some(word)
    } else {
        None
    };
    if let Some(word) = &fault_word {
        ordered.push(CudaKernelArgument::Buffer(word));
    }

    let grid_x = launch_grid(invocations, block_size)?;
    let image = compile_cuda_source(&source, architecture)?;
    let module = runtime.load_module(&image)?;
    let function = module.function(c"fusion_kernel")?;
    let stream = runtime.create_stream()?;
    // SAFETY: the lowerer emits exactly one f32 pointer parameter per validated binding, in
    // declaration order. We checked each device allocation's length for all indexed accesses.
    // The completion is waited before returning, so the buffers remain borrowed through use.
    let mut completion =
        unsafe { function.launch(&stream, [grid_x, 1, 1], [block_size, 1, 1], 0, &ordered)? };
    completion.wait()?;
    if let Some(word) = &fault_word {
        let mut bytes = [0_u8; size_of::<u64>()];
        word.copy_to(&mut bytes)?;
        if let Some(fault) = crate::owned_dispatch::decode_fault_word(u64::from_ne_bytes(bytes))? {
            return Err(CudaDispatchError::ExecutionFault(fault));
        }
    }
    Ok(())
}

fn launch_grid(invocations: u32, block_size: u32) -> Result<u32, CudaDispatchError> {
    if block_size == 0 {
        return Err(CudaDispatchError::InvalidBlockSize);
    }
    let grid = u64::from(invocations).div_ceil(u64::from(block_size));
    // The generated kernel computes its invocation ID in u32, and CUDA requires each rounded
    // grid dimension to contain fewer than 2^32 work items. Reject instead of wrapping IDs.
    if grid * u64::from(block_size) > u64::from(u32::MAX) {
        return Err(CudaDispatchError::GeometryOverflow);
    }
    u32::try_from(grid).map_err(|_| CudaDispatchError::GeometryOverflow)
}

#[cfg(test)]
mod tests {
    #[rustfmt::skip]
    use super::{
        CudaDispatchError,
        launch_grid,
    };
    #[rustfmt::skip]
    use fusion_pcu::{
        F32MapBuilder,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuValueType,
    };

    #[test]
    fn typed_builder_program_lowers_to_cuda_without_device() {
        let bindings = [
            PcuBinding::value(
                Some("input"),
                0,
                0,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::ReadOnly,
                PcuValueType::f32(),
            ),
            PcuBinding::value(
                Some("output"),
                0,
                1,
                PcuBindingStorageClass::Storage,
                PcuBindingAccess::WriteOnly,
                PcuValueType::f32(),
            ),
        ];
        let (builder, input) = F32MapBuilder::<8>::new(7, "map", [65, 1, 1], &bindings)
            .load_f32(PcuBindingRef::new(0, 0))
            .unwrap();
        let (builder, bias) = builder.constant(0.5).unwrap();
        let (builder, sum) = builder.add(input, bias).unwrap();
        let builder = builder.store_f32(PcuBindingRef::new(0, 1), sum).unwrap();
        let source = crate::lower_dispatch_to_cuda_source(&builder.ir()).unwrap();
        assert!(source.contains("fusion_kernel"));
    }

    #[test]
    fn rejects_rounded_launch_that_would_wrap_invocation_id() {
        assert!(matches!(
            launch_grid(u32::MAX, 4),
            Err(CudaDispatchError::GeometryOverflow)
        ));
        assert_eq!(launch_grid(65, 64).unwrap(), 2);
    }
}
