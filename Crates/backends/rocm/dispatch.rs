//! Synchronous, bounded PCU Dispatch execution on one `ROCm` device.

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
    PcuOwnedBindingRequirement,
    PcuOwnedDispatchBindingError,
};

#[rustfmt::skip]
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
    Binding(PcuOwnedDispatchBindingError),
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
    ExecutionFault(fusion_pcu::PcuExecutionFault),
}

impl fmt::Display for RocmDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lower(error) => write!(f, "PCU Dispatch cannot lower to ROCm: {error}"),
            Self::Binding(error) => write!(f, "PCU Dispatch binding is invalid: {error:?}"),
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
            Self::ExecutionFault(fault) => write!(
                f,
                "PCU invocation {} faulted: {:?}",
                fault.invocation_id, fault.kind
            ),
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

/// Compile and execute a supported scalar PCU Dispatch kernel, waiting for GPU completion.
///
/// All bindings must be device allocations sized for the kernel's logical invocation count.
/// The generated kernel bounds-checks rounded-up workgroups. This path deliberately completes
/// synchronously so caller-owned buffers cannot be dropped or modified during execution.
///
/// # Errors
///
/// Returns structured lowering, binding, compilation, launch, or completion errors.
#[allow(clippy::too_many_lines)]
pub fn execute_pcu_dispatch(
    runtime: &HipRuntime,
    kernel: &PcuDispatchKernelIr<'_>,
    architecture: &str,
    block_size: u32,
    bindings: &[RocmDispatchBinding<'_>],
) -> Result<(), RocmDispatchError> {
    let source = lower_dispatch_to_hip_source(kernel)?;
    let fault_law = crate::owned_dispatch::checked_scalar_fault_law(kernel);
    if operations_use_checked_fault(kernel.ops) && fault_law.is_none() {
        return Err(RocmDispatchError::Lower(
            RocmLowerError::UnsupportedKernelInterface,
        ));
    }
    if block_size == 0 {
        return Err(RocmDispatchError::InvalidBlockSize);
    }
    let invocations = kernel.entry.logical_shape[0];
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

    let checked_fault = operations_use_checked_fault(kernel.ops);
    let mut ordered = Vec::with_capacity(kernel.bindings.len() + usize::from(checked_fault));
    for requirement in binding_requirements(kernel)? {
        let id = requirement.target;
        let buffer = bindings
            .iter()
            .find(|candidate| candidate.id == id)
            .ok_or(RocmDispatchError::MissingBinding(id))?
            .buffer;
        let required = usize::try_from(requirement.min_required_bytes)
            .map_err(|_| RocmDispatchError::GeometryOverflow)?;
        if buffer.len() < required {
            return Err(RocmDispatchError::BufferTooSmall {
                binding: id,
                required,
                available: buffer.len(),
            });
        }
        ordered.push(HipKernelArgument::Buffer(buffer));
    }
    let mut fault_word = if checked_fault {
        let mut buffer = runtime.allocate(size_of::<u64>())?;
        buffer.copy_from(&u64::MAX.to_ne_bytes())?;
        Some(buffer)
    } else {
        None
    };
    if let Some(fault_word) = fault_word.as_ref() {
        ordered.push(HipKernelArgument::Buffer(fault_word));
    }

    let grid_x = launch_grid(invocations, block_size)?;
    let image = compile_hip_source(&source, architecture)?;
    let module = runtime.load_module(&image)?;
    let function = module.function(c"fusion_kernel")?;
    let stream = runtime.create_stream()?;
    // SAFETY: the lowerer emits typed pointers for the projected actual resources, in original
    // declaration order. Core requirements checked complete direct/grid indexed spans.
    // The completion is waited before returning, so the buffers remain borrowed through use.
    let mut completion =
        unsafe { function.launch(&stream, [grid_x, 1, 1], [block_size, 1, 1], 0, &ordered)? };
    completion.wait()?;
    if let Some(fault_word) = fault_word.take() {
        let mut bytes = [0_u8; size_of::<u64>()];
        fault_word.copy_to(&mut bytes)?;
        if let Some(fault) = decode_fault_word_in_extent(
            u64::from_ne_bytes(bytes),
            crate::owned_dispatch::checked_fault_extent(kernel),
        )? {
            if fault_law.is_none_or(|law| !law.allows(fault.kind, fault.recovered)) {
                return Err(RocmDispatchError::Hip(HipError::InvalidExecutionFaultWord(
                    u64::from_ne_bytes(bytes),
                )));
            }
            return Err(RocmDispatchError::ExecutionFault(fault));
        }
    }
    Ok(())
}

fn decode_fault_word_in_extent(
    word: u64,
    extent: u32,
) -> Result<Option<fusion_pcu::PcuExecutionFault>, RocmDispatchError> {
    match decode_fault_word(word) {
        Ok(Some(fault)) if !fault.is_within_logical_extent(u64::from(extent)) => Err(
            RocmDispatchError::Hip(HipError::InvalidExecutionFaultWord(word)),
        ),
        result => result,
    }
}

// The legacy synchronous entry compiles on every call, so it freezes the same resource
// projection and exact indexed spans as owned preparation before touching any buffer.
fn binding_requirements(
    kernel: &PcuDispatchKernelIr<'_>,
) -> Result<Vec<PcuOwnedBindingRequirement>, RocmDispatchError> {
    let count = core::num::NonZeroU32::new(kernel.entry.logical_shape[0])
        .ok_or(RocmDispatchError::GeometryOverflow)?;
    let shape = fusion_pcu::PcuInvocationShape::invocations(count);
    let projection = crate::codegen::lower::map_binding_projection(kernel);
    kernel
        .bindings
        .iter()
        .filter(|binding| {
            projection.is_none_or(|schema| {
                schema.contains_output(binding.reference())
                    || schema.input_bindings().contains(&binding.reference())
            })
        })
        .map(|binding| {
            PcuOwnedBindingRequirement::from_verified_binding(kernel, binding.reference(), shape)
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(RocmDispatchError::Binding)
}

const fn decode_fault_word(
    word: u64,
) -> Result<Option<fusion_pcu::PcuExecutionFault>, RocmDispatchError> {
    if word == u64::MAX {
        return Ok(None);
    }
    let recovered = word & (1 << 63) != 0;
    let payload = word & !(1 << 63);
    // The same bounded one-dimensional map ABI is used by the legacy synchronous adapter.
    if payload >> 3 > 0xffff_ffff {
        return Err(RocmDispatchError::Hip(HipError::InvalidExecutionFaultWord(
            word,
        )));
    }
    let kind = match payload & 7 {
        1 => fusion_pcu::PcuExecutionFaultKind::DivideByZero,
        2 => fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow,
        3 => fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow,
        4 => fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow,
        5 => fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand,
        _ => {
            return Err(RocmDispatchError::Hip(HipError::InvalidExecutionFaultWord(
                word,
            )));
        }
    };
    if recovered
        && !matches!(
            kind,
            fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
                | fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
        )
    {
        return Err(RocmDispatchError::Hip(HipError::InvalidExecutionFaultWord(
            word,
        )));
    }
    Ok(Some(fusion_pcu::PcuExecutionFault {
        kind,
        invocation_id: payload >> 3,
        recovered,
    }))
}

fn operations_use_checked_fault(ops: &[fusion_pcu::PcuDispatchOp<'_>]) -> bool {
    ops.iter().any(|op| match op {
        fusion_pcu::PcuDispatchOp::Data(
            fusion_pcu::PcuDispatchDataOp::CheckedDivRem { .. }
            | fusion_pcu::PcuDispatchDataOp::CheckedIntegerBinary { .. }
            | fusion_pcu::PcuDispatchDataOp::CheckedFloatBinary { .. }
            | fusion_pcu::PcuDispatchDataOp::CheckedFloatUnary { .. }
            | fusion_pcu::PcuDispatchDataOp::CheckedFloatConvert { .. },
        ) => true,
        fusion_pcu::PcuDispatchOp::GridStrideLoop { body, .. } => {
            operations_use_checked_fault(body)
        }
        _ => false,
    })
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
    #[rustfmt::skip]
    use super::{
        RocmDispatchError,
        decode_fault_word,
        launch_grid,
        operations_use_checked_fault,
    };
    use fusion_pcu::PcuValueType;
    #[test]
    fn rejects_rounded_launch_that_would_wrap_invocation_id() {
        assert!(matches!(
            launch_grid(u32::MAX, 4),
            Err(RocmDispatchError::GeometryOverflow)
        ));
        assert_eq!(launch_grid(65, 64).unwrap(), 2);
    }

    #[test]
    fn direct_status_decoder_matches_checked_fault_invariants() {
        assert!(matches!(decode_fault_word(u64::MAX), Ok(None)));
        assert!(
            matches!(decode_fault_word((3_u64 << 3) | 1), Ok(Some(fault)) if fault.invocation_id == 3)
        );
        assert!(decode_fault_word((1 << 63) | (3_u64 << 3) | 1).is_err());
        assert!(decode_fault_word(6).is_err());
        for code in 1..=5 {
            for recovered in [0, 1_u64 << 63] {
                assert!(
                    decode_fault_word(((u64::from(u32::MAX) + 1) << 3) | code | recovered).is_err()
                );
            }
        }
    }

    #[test]
    fn checked_fault_detection_visits_nested_loops() {
        let fault = [fusion_pcu::PcuDispatchOp::Data(
            fusion_pcu::PcuDispatchDataOp::CheckedFloatUnary {
                value_type: PcuValueType::f32(),
                op: fusion_pcu::model::PcuDispatchFloatUnaryOp::Relu,
                underflow_policy: fusion_pcu::PcuFloatUnderflowPolicy::IeeeAfterRounding,
                range_policy: fusion_pcu::PcuRangePolicy::Reject,
                result: fusion_pcu::PcuDispatchValueId(2),
                value: fusion_pcu::PcuDispatchValueId(1),
            },
        )];
        let inner = [fusion_pcu::PcuDispatchOp::GridStrideLoop {
            extent: 1,
            body: &fault,
        }];
        let outer = [fusion_pcu::PcuDispatchOp::GridStrideLoop {
            extent: 1,
            body: &inner,
        }];
        assert!(operations_use_checked_fault(&outer));
    }
}

#[cfg(all(test, feature = "tensor"))]
#[path = "dispatch/domain_tests.rs"]
mod domain_tests;
