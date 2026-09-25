//! Synchronous, bounded PCU Dispatch execution on one `ROCm` device.

use std::{
    error::Error,
    fmt,
    mem::size_of,
};

use fusion_pcu::{
    PcuBindingRef,
    PcuDispatchKernelIr,
};

use crate::{
    DeviceBuffer,
    HipCompileError,
    HipError,
    HipKernelArgument,
    HipRuntime,
    RocmLowerError,
    compile_hip_source,
    lower_dispatch_to_hip_source,
};

/// An already allocated device buffer bound to a PCU resource slot.
pub struct RocmDispatchBinding<'a> {
    pub id: PcuBindingRef,
    pub buffer: &'a DeviceBuffer,
}

#[derive(Debug)]
pub enum RocmDispatchError {
    Lower(RocmLowerError),
    Compile(HipCompileError),
    HipRtc(crate::HipRtcError),
    Hip(HipError),
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

impl fmt::Display for RocmDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lower(error) => write!(f, "PCU Dispatch cannot lower to ROCm: {error}"),
            Self::Compile(error) => write!(f, "ROCm code object compilation failed: {error}"),
            Self::HipRtc(error) => write!(f, "ROCm runtime compilation failed: {error}"),
            Self::Hip(error) => write!(f, "ROCm execution failed: {error}"),
            Self::CompilerUnavailable => {
                f.write_str("no usable ROCm source compiler is available for this device")
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
            Self::InvalidBlockSize => f.write_str("HIP block size must be nonzero"),
            Self::GeometryOverflow => f.write_str("HIP launch geometry or buffer size overflow"),
        }
    }
}

impl Error for RocmDispatchError {}

impl From<RocmLowerError> for RocmDispatchError {
    fn from(error: RocmLowerError) -> Self {
        Self::Lower(error)
    }
}
impl From<HipCompileError> for RocmDispatchError {
    fn from(error: HipCompileError) -> Self {
        Self::Compile(error)
    }
}
impl From<HipError> for RocmDispatchError {
    fn from(error: HipError) -> Self {
        Self::Hip(error)
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
    runtime: &HipRuntime,
    kernel: &PcuDispatchKernelIr<'_>,
    architecture: &str,
    block_size: u32,
    bindings: &[RocmDispatchBinding<'_>],
) -> Result<(), RocmDispatchError> {
    let source = lower_dispatch_to_hip_source(kernel)?;
    if block_size == 0 {
        return Err(RocmDispatchError::InvalidBlockSize);
    }
    let invocations = kernel.entry.logical_shape[0];
    let required = usize::try_from(invocations)
        .ok()
        .and_then(|count| count.checked_mul(size_of::<f32>()))
        .ok_or(RocmDispatchError::GeometryOverflow)?;

    for (index, binding) in bindings.iter().enumerate() {
        if bindings[..index].iter().any(|prior| prior.id == binding.id) {
            return Err(RocmDispatchError::DuplicateBinding(binding.id));
        }
        if !kernel.bindings.iter().any(|declared| {
            declared.set == binding.id.set && declared.binding == binding.id.binding
        }) {
            return Err(RocmDispatchError::UnexpectedBinding(binding.id));
        }
    }

    let mut ordered = Vec::with_capacity(kernel.bindings.len());
    for declared in kernel.bindings {
        let id = PcuBindingRef::new(declared.set, declared.binding);
        let buffer = bindings
            .iter()
            .find(|candidate| candidate.id == id)
            .ok_or(RocmDispatchError::MissingBinding(id))?
            .buffer;
        if buffer.len() < required {
            return Err(RocmDispatchError::BufferTooSmall {
                binding: id,
                required,
                available: buffer.len(),
            });
        }
        ordered.push(HipKernelArgument::Buffer(buffer));
    }

    let grid_x = launch_grid(invocations, block_size)?;
    let image = compile_hip_source(&source, architecture)?;
    let module = runtime.load_module(&image)?;
    let function = module.function(c"fusion_kernel")?;
    let stream = runtime.create_stream()?;
    // SAFETY: the lowerer emits exactly one f32 pointer parameter per validated binding, in
    // declaration order. We checked each device allocation's length for all indexed accesses.
    // The completion is waited before returning, so the buffers remain borrowed through use.
    let mut completion =
        unsafe { function.launch(&stream, [grid_x, 1, 1], [block_size, 1, 1], 0, &ordered)? };
    completion.wait()?;
    Ok(())
}

fn launch_grid(invocations: u32, block_size: u32) -> Result<u32, RocmDispatchError> {
    if block_size == 0 {
        return Err(RocmDispatchError::InvalidBlockSize);
    }
    let grid = u64::from(invocations).div_ceil(u64::from(block_size));
    // The generated kernel computes its invocation ID in u32, and HIP requires each rounded
    // grid dimension to contain fewer than 2^32 work items. Reject instead of wrapping IDs.
    if grid * u64::from(block_size) > u64::from(u32::MAX) {
        return Err(RocmDispatchError::GeometryOverflow);
    }
    u32::try_from(grid).map_err(|_| RocmDispatchError::GeometryOverflow)
}

#[cfg(test)]
mod tests {
    use super::{
        RocmDispatchError,
        launch_grid,
    };
    use fusion_pcu::{
        F32MapBuilder,
        PcuBinding,
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingStorageClass,
        PcuValueType,
    };

    #[test]
    fn typed_builder_program_lowers_to_hip_without_device() {
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
        let source = crate::lower_dispatch_to_hip_source(&builder.ir()).unwrap();
        assert!(source.contains("fusion_kernel"));
    }

    #[test]
    fn rejects_rounded_launch_that_would_wrap_invocation_id() {
        assert!(matches!(
            launch_grid(u32::MAX, 4),
            Err(RocmDispatchError::GeometryOverflow)
        ));
        assert_eq!(launch_grid(65, 64).unwrap(), 2);
    }
}
