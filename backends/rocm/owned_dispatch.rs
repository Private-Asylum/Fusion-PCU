//! Owned, asynchronous PCU Dispatch adapter for one explicitly selected HIP device.
//!
//! This first adapter deliberately supports the existing scalar f32 source lowerer only. It
//! requires buffers allocated by this adapter's HIP runtime and retains exclusive buffer leases
//! until the HIP event proves that the kernel has stopped accessing them.

use std::{
    error::Error,
    fmt,
    mem::size_of,
};

use fusion_pcu::{
    PcuBaseContract,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuCompletionState,
    PcuDeviceIdentity,
    PcuDispatchFeatureCaps,
    PcuDispatchOpCaps,
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
    PcuObjectKind,
    PcuObjectRef,
    PcuOwnedDispatchBindingError,
    PcuPrimitiveCaps,
    PcuPrimitiveSupport,
    PcuSupport,
    PcuValueTypeCaps,
    validate_owned_dispatch_bindings,
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
    compile_hip_source,
    lower_dispatch_to_hip_source,
};

/// ROCm-owned dispatch setup or submission failure.
#[derive(Debug)]
pub enum RocmOwnedDispatchError {
    Hip(HipError),
    Lower(RocmLowerError),
    Compile(crate::HipCompileError),
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
    GeometryOverflow,
    Binding(PcuOwnedDispatchBindingError),
}

impl fmt::Display for RocmOwnedDispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hip(error) => error.fmt(f),
            Self::Lower(error) => write!(f, "PCU Dispatch cannot lower to ROCm: {error}"),
            Self::Compile(error) => write!(f, "ROCm code object compilation failed: {error}"),
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
            Self::GeometryOverflow => f.write_str("HIP launch geometry overflow"),
            Self::Binding(error) => write!(f, "invalid owned PCU binding: {error:?}"),
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
    architecture: String,
    block_size: u32,
}

impl RocmOwnedDispatchBackend {
    /// Validate and open one discovered device for owned asynchronous Dispatch.
    ///
    /// The HIP architecture remains an explicit caller choice because the stable HIP APIs used by
    /// discovery do not yet provide a safely versioned way to query its code-generation target.
    pub fn open(
        discovery: &RocmDiscovery,
        device: PcuObjectRef,
        architecture: impl Into<String>,
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
        let runtime = discovery.open_device(device)?;
        let actual = u32::try_from(runtime.device_info()?.index)
            .map_err(|_| RocmOwnedDispatchError::InvalidDeviceReference)?;
        if actual != identity.device_id() {
            return Err(RocmOwnedDispatchError::RuntimeDeviceMismatch {
                expected: identity.device_id(),
                actual,
            });
        }
        Ok(Self {
            runtime,
            device: identity,
            architecture: architecture.into(),
            block_size,
        })
    }

    /// Allocate a buffer in the exact HIP runtime session accepted by this adapter.
    pub fn allocate(&self, bytes: usize) -> Result<DeviceBuffer, HipError> {
        self.runtime.allocate(bytes)
    }

    /// Create a memory provider for this same opened HIP device and its caller-assigned pool.
    pub fn memory_provider(&self, pool: fusion_pcu::PcuMemoryPoolId) -> RocmMemoryProvider {
        self.runtime.memory_provider(pool)
    }

    /// Create binding metadata from this adapter's device identity and the allocation's true size.
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

    fn submit(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: Vec<PcuOwnedBinding<DeviceBuffer>>,
    ) -> Result<RocmOwnedCompletion, RocmOwnedDispatchError> {
        let kernel = submission.kernel;
        validate_owned_dispatch_bindings(kernel, submission.shape, self.device, &bindings)
            .map_err(RocmOwnedDispatchError::Binding)?;
        let source = lower_dispatch_to_hip_source(kernel)?;
        let logical_threads = submission.shape.thread_count().get();
        if kernel.entry.logical_shape != [logical_threads, 1, 1] {
            return Err(RocmOwnedDispatchError::Lower(
                RocmLowerError::InvalidKernelShape,
            ));
        }
        let required = usize::try_from(logical_threads)
            .ok()
            .and_then(|threads| threads.checked_mul(size_of::<f32>()))
            .ok_or(RocmOwnedDispatchError::GeometryOverflow)?;

        for binding in &bindings {
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
            if actual < required {
                return Err(RocmOwnedDispatchError::BufferTooSmall {
                    binding: binding.target,
                    required,
                    available: actual,
                });
            }
        }

        let grid_x = launch_grid(logical_threads, self.block_size)?;
        let image = compile_hip_source(&source, &self.architecture)?;
        let module = self.runtime.load_module(&image)?;
        let function = module.function(c"fusion_kernel")?;
        let stream = self.runtime.create_stream()?;
        let indices = binding_order(
            kernel
                .bindings
                .iter()
                .map(|binding| PcuBindingRef::new(binding.set, binding.binding)),
            &bindings,
        )
        .map_err(RocmOwnedDispatchError::Binding)?;
        let arguments = indices
            .into_iter()
            .map(|index| HipKernelArgument::Buffer(&bindings[index].resource))
            .collect::<Vec<_>>();

        // SAFETY: lowering validates the generated kernel ABI as exactly one f32 pointer per
        // declared binding in declaration order. Core admission validates full binding coverage,
        // device metadata, access, and type. This adapter additionally verifies actual allocation
        // length and HIP runtime identity, and HipKernel::launch acquires the shared exclusive
        // allocation gates and retains module, stream, and allocations through event completion.
        let hip = unsafe {
            function.launch(
                &stream,
                [grid_x, 1, 1],
                [self.block_size, 1, 1],
                0,
                &arguments,
            )?
        };
        Ok(RocmOwnedCompletion {
            hip: Some(hip),
            terminal: None,
        })
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

    fn device_identity(&self) -> PcuDeviceIdentity {
        self.device
    }

    fn submit_dispatch_owned_direct(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: Self::Bindings,
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<Self::Completion, Self::Error> {
        if !parameters.is_empty() {
            return Err(RocmOwnedDispatchError::Lower(
                RocmLowerError::UnsupportedKernelInterface,
            ));
        }
        self.submit(submission, bindings)
    }
}

/// Core completion contract over the HIP event-backed completion token.
pub struct RocmOwnedCompletion {
    hip: Option<HipCompletion>,
    terminal: Option<PcuCompletionOutcome>,
}

impl PcuOwnedCompletion for RocmOwnedCompletion {
    type Error = HipError;

    fn state(&self) -> Result<PcuCompletionState, Self::Error> {
        Ok(match self.terminal {
            Some(PcuCompletionOutcome::Succeeded) => PcuCompletionState::Succeeded,
            Some(PcuCompletionOutcome::Failed) => PcuCompletionState::Failed,
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
        self.hip.take();
        self.terminal = Some(PcuCompletionOutcome::Succeeded);
        Ok(PcuCompletionOutcome::Succeeded)
    }
}

fn launch_grid(threads: u32, block_size: u32) -> Result<u32, RocmOwnedDispatchError> {
    if block_size == 0 {
        return Err(RocmOwnedDispatchError::InvalidBlockSize);
    }
    let grid = u64::from(threads).div_ceil(u64::from(block_size));
    if grid * u64::from(block_size) > u64::from(u32::MAX) {
        return Err(RocmOwnedDispatchError::GeometryOverflow);
    }
    u32::try_from(grid).map_err(|_| RocmOwnedDispatchError::GeometryOverflow)
}

fn binding_order<R>(
    declared: impl Iterator<Item = PcuBindingRef>,
    provided: &[PcuOwnedBinding<R>],
) -> Result<Vec<usize>, PcuOwnedDispatchBindingError> {
    declared
        .map(|target| {
            provided
                .iter()
                .position(|binding| binding.target == target)
                .ok_or(PcuOwnedDispatchBindingError::Missing(target))
        })
        .collect()
}

fn owned_dispatch_support() -> PcuSupport {
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
        PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
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
            .union(PcuDispatchOpCaps::CONTROL_RETURN)
            .union(PcuDispatchOpCaps::BINDING_LOAD)
            .union(PcuDispatchOpCaps::BINDING_STORE),
        PcuDispatchOpCaps::empty(),
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
    .union(PcuDispatchOpCaps::CONTROL_RETURN)
    .union(PcuDispatchOpCaps::BINDING_LOAD)
    .union(PcuDispatchOpCaps::BINDING_STORE);

const OWNED_EXECUTORS: [PcuExecutorDescriptor; 1] = [PcuExecutorDescriptor {
    id: PcuExecutorId(0),
    name: "rocm-owned-dispatch",
    class: PcuExecutorClass::Compute,
    origin: PcuExecutorOrigin::TopologyBound,
    support: PcuExecutorSupport {
        primitives: PcuPrimitiveCaps::DISPATCH,
        dispatch_policy: PcuDispatchPolicyCaps::SERIAL
            .union(PcuDispatchPolicyCaps::ORDERED_SUBMISSION),
        value_types: PcuValueTypeCaps::FLOAT32.union(PcuValueTypeCaps::SCALAR_VALUES),
        dispatch_instructions: OWNED_DISPATCH_INSTRUCTIONS,
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
        RocmOwnedDispatchError,
        binding_order,
        launch_grid,
    };
    use fusion_pcu::{
        PcuBindingAccess,
        PcuBindingRef,
        PcuBindingType,
        PcuDeviceIdentity,
        PcuObjectKind,
        PcuObjectRef,
        PcuOwnedBinding,
        PcuProviderId,
        PcuValueType,
    };

    struct Noop;

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

        let order = binding_order([second, first].into_iter(), &provided).unwrap();
        assert_eq!(order, [1, 0]);
    }
}
