//! Optional PCU execution runners and runtime selection.

use core::fmt;
use std::boxed::Box;
use std::error::Error;
use std::string::String;
use std::vec::Vec;

use crate::backends::spirv::{
    lower_dispatch_to_spirv,
    PcuSpirvError,
    PcuSpirvFixedSink,
    PcuSpirvLoweringOptions,
};
use crate::dispatch::{
    validate_dispatch_submission,
    validate_invocation_bindings,
    validate_parameters,
};
use crate::{
    PcuDispatchSubmission,
    PcuError,
    PcuInvocationBinding,
    PcuInvocationBindings,
    PcuInvocationParameters,
    PcuKernelIrContract,
};

#[cfg(feature = "runner-vulkan")]
#[path = "vulkan/vulkan.rs"]
pub mod vulkan;

#[cfg(feature = "runner-vulkan")]
pub use vulkan::*;

const PCU_RUNTIME_SPIRV_WORD_CAPACITY: usize = 1024;
pub const PCU_SPIRV_BACKEND_ID: &str = "SPIR-V";

/// Abstract resource-addressing models that a runner may select for a resource class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuResourceAddressingModel {
    FixedDescriptors,
    DescriptorIndex,
    BufferDeviceAddress,
}

/// Factory registration for one PCU runner.
#[derive(Clone, Copy)]
pub struct PcuRunnerDescriptor {
    pub id: &'static str,
    pub priority: i32,
    pub probe: fn() -> Result<(), PcuRunnerError>,
    pub open: fn() -> Result<PcuRunnerHandle, PcuRunnerError>,
}

/// Mutable runner registry.
///
/// Compiled-in runners register themselves into the default registry, while future dynamically
/// loaded runners can use the same `register(...)` path.
#[derive(Default)]
pub struct PcuRunnerRegistry {
    descriptors: Vec<PcuRunnerDescriptor>,
}

impl PcuRunnerRegistry {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            descriptors: Vec::new(),
        }
    }

    #[must_use]
    pub fn with_compiled_runners() -> Self {
        let mut registry = Self::new();
        register_compiled_runners(&mut registry);
        registry
    }

    /// Registers one runner factory.
    ///
    /// # Errors
    ///
    /// Returns `RunnerAlreadyRegistered` when an id is already present.
    pub fn register(&mut self, descriptor: PcuRunnerDescriptor) -> Result<(), PcuRunnerError> {
        if self
            .descriptors
            .iter()
            .any(|existing| existing.id == descriptor.id)
        {
            return Err(PcuRunnerError::RunnerAlreadyRegistered { id: descriptor.id });
        }
        self.descriptors.push(descriptor);
        Ok(())
    }

    #[must_use]
    pub fn descriptors(&self) -> &[PcuRunnerDescriptor] {
        &self.descriptors
    }

    /// Opens the highest-priority supported runner.
    ///
    /// # Errors
    ///
    /// Returns a selection error if no compiled/registered runner is usable.
    pub fn open_auto(&self) -> Result<PcuRunnerHandle, PcuRunnerError> {
        if self.descriptors.is_empty() {
            return Err(PcuRunnerError::NoRegisteredRunners);
        }

        let mut selected: Option<PcuRunnerDescriptor> = None;
        for descriptor in self.descriptors.iter().copied() {
            if (descriptor.probe)().is_err() {
                continue;
            }

            match selected {
                Some(current) if current.priority >= descriptor.priority => {}
                _ => selected = Some(descriptor),
            }
        }

        let Some(descriptor) = selected else {
            return Err(PcuRunnerError::NoSupportedRunner);
        };
        (descriptor.open)()
    }

    /// Opens a runner by string id.
    ///
    /// # Errors
    ///
    /// Returns `RunnerNotRegistered` if the id is absent, or `RunnerUnavailable` if probing/opening
    /// fails.
    pub fn open_by_id(&self, id: &str) -> Result<PcuRunnerHandle, PcuRunnerError> {
        let Some(descriptor) = self
            .descriptors
            .iter()
            .copied()
            .find(|descriptor| descriptor.id == id)
        else {
            return Err(PcuRunnerError::RunnerNotRegistered { id: id.to_owned() });
        };

        if let Err(error) = (descriptor.probe)() {
            return Err(PcuRunnerError::RunnerUnavailable {
                id: descriptor.id,
                reason: error.to_string(),
            });
        }
        (descriptor.open)()
    }

    fn register_unchecked(&mut self, descriptor: PcuRunnerDescriptor) {
        self.descriptors.push(descriptor);
    }
}

fn register_compiled_runners(registry: &mut PcuRunnerRegistry) {
    #[cfg(feature = "runner-vulkan")]
    registry.register_unchecked(vulkan::PCU_VULKAN_RUNNER_DESCRIPTOR);
}

/// Opened PCU runtime selected from a runner registry.
pub struct PcuRuntime {
    runner: PcuRunnerHandle,
}

impl PcuRuntime {
    /// Opens the highest-priority supported compiled-in runner.
    ///
    /// # Errors
    ///
    /// Returns a selection or runner-open error when no runner can execute.
    pub fn auto() -> Result<Self, PcuRunnerError> {
        let registry = PcuRunnerRegistry::with_compiled_runners();
        Self::from_registry_auto(&registry)
    }

    /// Opens a compiled-in runner by id, for example `"Vulkan"`.
    ///
    /// # Errors
    ///
    /// Returns `RunnerNotRegistered` or `RunnerUnavailable` for the requested id.
    pub fn with_runner(id: &str) -> Result<Self, PcuRunnerError> {
        let registry = PcuRunnerRegistry::with_compiled_runners();
        Self::from_registry_runner(&registry, id)
    }

    /// Opens the highest-priority supported runner from a caller-provided registry.
    ///
    /// # Errors
    ///
    /// Returns a selection or runner-open error when no runner can execute.
    pub fn from_registry_auto(registry: &PcuRunnerRegistry) -> Result<Self, PcuRunnerError> {
        Ok(Self {
            runner: registry.open_auto()?,
        })
    }

    /// Opens a named runner from a caller-provided registry.
    ///
    /// # Errors
    ///
    /// Returns `RunnerNotRegistered` or `RunnerUnavailable` for the requested id.
    pub fn from_registry_runner(
        registry: &PcuRunnerRegistry,
        id: &str,
    ) -> Result<Self, PcuRunnerError> {
        Ok(Self {
            runner: registry.open_by_id(id)?,
        })
    }

    #[must_use]
    pub fn runner_id(&self) -> &'static str {
        self.runner.id()
    }

    #[must_use]
    pub fn runner_device_name(&self) -> Option<&str> {
        self.runner.device_name()
    }

    /// Lowers and submits one dispatch kernel through the selected runner.
    ///
    /// # Errors
    ///
    /// Returns structural admission, SPIR-V lowering, runner support, or execution failures.
    pub fn submit_dispatch(
        &self,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuInvocationBinding<'_>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuDispatchReport, PcuRunnerError> {
        validate_dispatch_submission(submission)
            .map_err(|error| PcuRunnerError::DispatchAdmission { error })?;
        validate_parameters(submission.kernel.signature(), parameters)
            .map_err(|error| PcuRunnerError::DispatchAdmission { error })?;
        validate_invocation_bindings(
            submission.kernel.signature(),
            PcuInvocationBindings { bindings },
        )
        .map_err(|error| PcuRunnerError::DispatchAdmission { error })?;

        let mut sink = PcuSpirvFixedSink::<PCU_RUNTIME_SPIRV_WORD_CAPACITY>::new();
        let info = lower_dispatch_to_spirv(
            submission.kernel,
            PcuSpirvLoweringOptions::minimal_shader(),
            &mut sink,
        )
        .map_err(|error| PcuRunnerError::BackendLowering {
            backend: PCU_SPIRV_BACKEND_ID,
            error,
        })?;
        let lowered = PcuLoweredSpirvDispatch {
            words: sink.as_slice(),
            word_count: info.word_count,
            bound: info.bound,
            local_size: submission.kernel.entry.logical_shape,
        };
        let execution = self
            .runner
            .submit_spirv_dispatch(&lowered, submission, bindings, parameters)?;

        Ok(PcuDispatchReport {
            runner_id: self.runner.id(),
            backend_id: PCU_SPIRV_BACKEND_ID,
            device_name: self.runner.device_name().map(str::to_owned),
            spirv_words: lowered.word_count,
            spirv_bound: lowered.bound,
            execution,
        })
    }
}

/// Type-erased opened runner.
pub struct PcuRunnerHandle {
    inner: Box<dyn PcuComputeRunner>,
}

impl fmt::Debug for PcuRunnerHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PcuRunnerHandle")
            .field("id", &self.id())
            .field("priority", &self.priority())
            .field("device_name", &self.device_name())
            .finish()
    }
}

impl PcuRunnerHandle {
    #[must_use]
    pub fn new(inner: Box<dyn PcuComputeRunner>) -> Self {
        Self { inner }
    }

    #[must_use]
    pub fn id(&self) -> &'static str {
        self.inner.id()
    }

    #[must_use]
    pub fn priority(&self) -> i32 {
        self.inner.priority()
    }

    #[must_use]
    pub fn device_name(&self) -> Option<&str> {
        self.inner.device_name()
    }

    fn submit_spirv_dispatch(
        &self,
        dispatch: &PcuLoweredSpirvDispatch<'_>,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuInvocationBinding<'_>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuRunnerExecutionReport, PcuRunnerError> {
        self.inner
            .submit_spirv_dispatch(dispatch, submission, bindings, parameters)
    }
}

/// Runner-side compute contract.
pub trait PcuComputeRunner {
    fn id(&self) -> &'static str;
    fn priority(&self) -> i32;

    fn device_name(&self) -> Option<&str> {
        None
    }

    /// Submits one lowered SPIR-V dispatch through this runner.
    ///
    /// # Errors
    ///
    /// Returns a runner support or execution error when the dispatch cannot be executed.
    fn submit_spirv_dispatch(
        &self,
        dispatch: &PcuLoweredSpirvDispatch<'_>,
        submission: PcuDispatchSubmission<'_>,
        bindings: &mut [PcuInvocationBinding<'_>],
        parameters: PcuInvocationParameters<'_>,
    ) -> Result<PcuRunnerExecutionReport, PcuRunnerError>;
}

/// Lowered SPIR-V dispatch handed to a SPIR-V-consuming runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PcuLoweredSpirvDispatch<'a> {
    pub words: &'a [u32],
    pub word_count: usize,
    pub bound: u32,
    pub local_size: [u32; 3],
}

/// Runner execution details for one dispatch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuRunnerExecutionReport {
    pub work_items: u32,
    pub dispatch_groups: [u32; 3],
    pub resource_model: PcuResourceAddressingModel,
}

/// Public runtime dispatch report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PcuDispatchReport {
    pub runner_id: &'static str,
    pub backend_id: &'static str,
    pub device_name: Option<String>,
    pub spirv_words: usize,
    pub spirv_bound: u32,
    pub execution: PcuRunnerExecutionReport,
}

#[derive(Debug)]
pub enum PcuRunnerError {
    RunnerAlreadyRegistered {
        id: &'static str,
    },
    RunnerNotRegistered {
        id: String,
    },
    RunnerUnavailable {
        id: &'static str,
        reason: String,
    },
    NoRegisteredRunners,
    NoSupportedRunner,
    DispatchAdmission {
        error: PcuError,
    },
    BackendLowering {
        backend: &'static str,
        error: PcuSpirvError,
    },
    DispatchUnsupportedByRunner {
        id: &'static str,
    },
    RunnerExecution {
        id: &'static str,
        reason: String,
    },
}

impl fmt::Display for PcuRunnerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RunnerAlreadyRegistered { id } => {
                write!(formatter, "PCU runner {id} already registered")
            }
            Self::RunnerNotRegistered { id } => {
                write!(formatter, "PCU runner {id} is not registered")
            }
            Self::RunnerUnavailable { id, reason } => {
                write!(formatter, "PCU runner {id} is unavailable: {reason}")
            }
            Self::NoRegisteredRunners => formatter.write_str("no PCU runners are registered"),
            Self::NoSupportedRunner => formatter.write_str("no supported PCU runner is available"),
            Self::DispatchAdmission { error } => {
                write!(formatter, "PCU dispatch admission failed: {error}")
            }
            Self::BackendLowering { backend, error } => {
                write!(formatter, "PCU backend {backend} lowering failed: {error}")
            }
            Self::DispatchUnsupportedByRunner { id } => {
                write!(formatter, "PCU runner {id} does not support this dispatch")
            }
            Self::RunnerExecution { id, reason } => {
                write!(formatter, "PCU runner {id} execution failed: {reason}")
            }
        }
    }
}

impl Error for PcuRunnerError {}

#[cfg(test)]
mod tests {
    use super::{
        PcuComputeRunner,
        PcuDispatchSubmission,
        PcuInvocationBinding,
        PcuInvocationParameters,
        PcuLoweredSpirvDispatch,
        PcuResourceAddressingModel,
        PcuRunnerDescriptor,
        PcuRunnerError,
        PcuRunnerExecutionReport,
        PcuRunnerHandle,
        PcuRunnerRegistry,
        PcuRuntime,
    };
    use std::boxed::Box;

    const DUMMY_ID: &str = "Dummy";

    struct DummyRunner;

    impl PcuComputeRunner for DummyRunner {
        fn id(&self) -> &'static str {
            DUMMY_ID
        }

        fn priority(&self) -> i32 {
            1
        }

        fn submit_spirv_dispatch(
            &self,
            _dispatch: &PcuLoweredSpirvDispatch<'_>,
            _submission: PcuDispatchSubmission<'_>,
            _bindings: &mut [PcuInvocationBinding<'_>],
            _parameters: PcuInvocationParameters<'_>,
        ) -> Result<PcuRunnerExecutionReport, PcuRunnerError> {
            Ok(PcuRunnerExecutionReport {
                work_items: 0,
                dispatch_groups: [0, 0, 0],
                resource_model: PcuResourceAddressingModel::FixedDescriptors,
            })
        }
    }

    fn dummy_probe() -> Result<(), PcuRunnerError> {
        Ok(())
    }

    fn dummy_open() -> Result<PcuRunnerHandle, PcuRunnerError> {
        Ok(PcuRunnerHandle::new(Box::new(DummyRunner)))
    }

    const DUMMY_DESCRIPTOR: PcuRunnerDescriptor = PcuRunnerDescriptor {
        id: DUMMY_ID,
        priority: 1,
        probe: dummy_probe,
        open: dummy_open,
    };

    #[test]
    fn empty_registry_reports_no_registered_runners() {
        let registry = PcuRunnerRegistry::new();

        let error = registry
            .open_auto()
            .expect_err("empty registry should fail");

        assert!(matches!(error, PcuRunnerError::NoRegisteredRunners));
    }

    #[test]
    fn missing_named_runner_reports_not_registered() {
        let registry = PcuRunnerRegistry::new();

        let error = registry
            .open_by_id("Vulkan")
            .expect_err("missing runner should fail");

        assert!(matches!(error, PcuRunnerError::RunnerNotRegistered { .. }));
    }

    #[test]
    fn duplicate_runner_registration_is_rejected() {
        let mut registry = PcuRunnerRegistry::new();

        registry
            .register(DUMMY_DESCRIPTOR)
            .expect("first registration should succeed");
        let error = registry
            .register(DUMMY_DESCRIPTOR)
            .expect_err("duplicate registration should fail");

        assert!(matches!(
            error,
            PcuRunnerError::RunnerAlreadyRegistered { id: DUMMY_ID }
        ));
    }

    #[test]
    fn runtime_can_open_registered_runner_by_id() {
        let mut registry = PcuRunnerRegistry::new();
        registry
            .register(DUMMY_DESCRIPTOR)
            .expect("dummy registration should succeed");

        let runtime = PcuRuntime::from_registry_runner(&registry, DUMMY_ID)
            .expect("dummy runner should open");

        assert_eq!(runtime.runner_id(), DUMMY_ID);
    }
}
