//! Ordinary Vulkan leaf ownership keeps the discovered logical root through escaped values.
#[rustfmt::skip]
use std::rc::Rc;
#[rustfmt::skip]
use fusion_pcu_core::{
    PcuDeviceActivation,
    PcuRuntimeDiscovery,
    PcuScalar,
    PcuObjectRef,
    PcuObjectKind,
    PcuProviderId,
    PcuProviderDescriptor,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuTargetDescriptor,
    PcuDeviceDescriptor,
    PcuDeviceClass,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanDiscovery,
    PcuVulkanTensorError,
};
#[rustfmt::skip]
use crate::global::{
    arguments::TensorBacking,
    tensor::{capture::PcuCapturedTensorProgram, PcuTensorInput, TensorInputKind},
    PcuArgumentError,
    PcuBackendChoice,
    PcuExecutionError,
    PcuExecutionPolicy,
    PcuTensor,
};
#[path = "roots/roots.rs"]
mod roots;
#[path = "typed/typed.rs"]
mod typed;

pub(super) struct Prepared {
    pub(super) root: Rc<PcuVulkanBackend>,
    program: typed::Typed,
    shape: Rc<[usize]>,
}
fn map_error(error: PcuVulkanTensorError) -> PcuExecutionError {
    match error {
        PcuVulkanTensorError::Graph(error) => PcuExecutionError::TensorBuild(error),
        PcuVulkanTensorError::Native(error) => PcuExecutionError::VulkanExecution(error),
        PcuVulkanTensorError::UnsupportedNode(_) | PcuVulkanTensorError::OutputCount(_) => {
            PcuExecutionError::InvalidTensorSourcePlan
        }
    }
}
impl Prepared {
    pub(super) fn prepare<T: PcuScalar, const N: usize>(
        built: &PcuCapturedTensorProgram,
        inputs: &[PcuTensorInput<'_, T>; N],
        options: PcuExecutionPolicy,
    ) -> Result<Self, PcuExecutionError> {
        typed::Typed::assess(&built.program, T::TYPE)?;
        if !matches!(
            options.backend,
            PcuBackendChoice::Vulkan | PcuBackendChoice::Automatic
        ) {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        let mut root = None;
        for &index in &built.input_indices {
            if let TensorInputKind::Resident(owner) = inputs[index].kind {
                #[allow(irrefutable_let_patterns)] // Vulkan-only has exactly one backing.
                let TensorBacking::Vulkan {
                    buffer,
                    root: resident,
                    ..
                } = &owner.backing
                else {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::UnsupportedResidentBorrow,
                    ));
                };
                owner
                    .validate_initialized()
                    .map_err(PcuExecutionError::Argument)?;
                if !resident.owns_buffer(buffer) {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::SessionMismatch,
                    ));
                }
                if root
                    .as_ref()
                    .is_some_and(|previous| !Rc::ptr_eq(previous, resident))
                {
                    return Err(PcuExecutionError::Argument(
                        PcuArgumentError::SessionMismatch,
                    ));
                }
                root = Some(Rc::clone(resident));
            }
        }
        let ordinal = options.device.unwrap_or(0);
        if root.as_ref().is_some_and(|root| {
            options.device.is_some_and(|ordinal| {
                root.device_identity()
                    .is_none_or(|identity| identity.device_id() != ordinal)
            })
        }) {
            return Err(PcuExecutionError::ResidentPolicyConflict);
        }
        let root = if root.is_some() {
            root
        } else {
            roots::retained(ordinal)?
        };
        let root = if let Some(root) = root {
            root
        } else {
            let root = Rc::new(open(ordinal)?);
            roots::remember(ordinal, &root)?;
            root
        };
        let program = typed::Typed::prepare(&root, &built.program, T::TYPE)?;
        let shape = program.shape();
        Ok(Self {
            root,
            program,
            shape,
        })
    }
    pub(super) fn execute<T: PcuScalar, const N: usize>(
        &mut self,
        inputs: &[PcuTensorInput<'_, T>; N],
        indices: &[usize],
    ) -> Result<PcuTensor<T>, PcuExecutionError> {
        let buffer = self.program.execute(inputs, indices)?;
        Ok(PcuTensor {
            backing: TensorBacking::Vulkan {
                buffer,
                shape: Rc::clone(&self.shape),
                root: Rc::clone(&self.root),
                validity: crate::global::arguments::ResidentValidity::Ready,
            },
        })
    }
}

fn open(ordinal: u32) -> Result<PcuVulkanBackend, PcuExecutionError> {
    let discovery = PcuVulkanDiscovery::discover().map_err(PcuExecutionError::VulkanExecution)?;
    let empty = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };
    let ready = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: ready,
    }];
    discovery
        .providers(&mut providers)
        .map_err(PcuExecutionError::VulkanExecution)?;
    let mut targets = [PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness: ready,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .map_err(PcuExecutionError::VulkanExecution)?;
    let count = discovery
        .devices(targets[0].reference, &mut [])
        .map_err(PcuExecutionError::VulkanExecution)?;
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: empty,
            target: empty,
            name: "",
            class: PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None
        };
        count
    ];
    discovery
        .devices(targets[0].reference, &mut devices)
        .map_err(PcuExecutionError::VulkanExecution)?;
    let descriptor = devices
        .iter()
        .find(|device| device.reference.id == ordinal)
        .ok_or(PcuExecutionError::ResidentPolicyConflict)?;
    discovery
        .open_device(descriptor.reference)
        .map_err(PcuExecutionError::VulkanExecution)
}
