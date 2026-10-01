//! Generation-scoped delegated runtime discovery and operation-specific cold offers.

#[rustfmt::skip]
use std::{
    path::Path,
    sync::atomic::{
        AtomicU64,
        Ordering,
    },
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCapabilitySnapshot,
    PcuCaps,
    PcuContextDescriptor,
    PcuContextKind,
    PcuDeviceActivation,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuDeviceIdentity,
    PcuExecutorClass,
    PcuExecutorDescriptor,
    PcuExecutorId,
    PcuExecutorOrigin,
    PcuExecutorSupport,
    PcuImplementationKind,
    PcuMemoryDomainDescriptor,
    PcuMemoryDomainKind,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuSupport,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use crate::{
    MlxDeviceFacts,
    MlxError,
    MlxRuntime,
    MlxSession,
};

const PROVIDER: PcuProviderId = PcuProviderId(0x4d4c_5831);
pub const EXECUTOR: PcuExecutorId = PcuExecutorId(0);
#[cfg(feature = "tensor")]
pub const MATMUL_REVISION: u64 = 0x0002_0020_0003_0001;
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// Cold MLX GPU inventory with separately admitted tensor offers.
///
/// SDK inspection may initialize its device/library caches, but
/// no PCU stream, tensor allocation or requested workload is created until activation.
/// Generic dispatch IR is unsupported; tensor implementations are offered separately.
pub struct MlxDiscovery {
    runtime: MlxRuntime,
    generation: u64,
    devices: Vec<MlxDeviceFacts>,
}

impl MlxDiscovery {
    /// Loads the explicitly selected pinned bridge and snapshots actual GPU facts.
    ///
    /// # Errors
    /// Returns unavailable/unsupported ABI/SDK/backend or native discovery errors.
    pub fn discover(bridge: impl AsRef<Path>) -> Result<Self, MlxError> {
        let runtime = MlxRuntime::load(bridge)?;
        let devices = runtime.devices()?;
        u32::try_from(devices.len()).map_err(|_| MlxError::InvalidExtent)?;
        let generation = GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| MlxError::Runtime("MLX discovery generation exhausted".into()))?;
        Ok(Self {
            runtime,
            generation,
            devices,
        })
    }

    /// Returns one inventory-bound device reference without activating a stream.
    ///
    /// # Errors
    /// Returns invalid device index.
    pub fn device_reference(&self, index: usize) -> Result<PcuObjectRef, MlxError> {
        if index >= self.devices.len() {
            return Err(MlxError::InvalidExtent);
        }
        Ok(self.reference(
            PcuObjectKind::Device,
            u32::try_from(index).map_err(|_| MlxError::InvalidExtent)?,
        ))
    }

    const fn reference(&self, kind: PcuObjectKind, id: u32) -> PcuObjectRef {
        PcuObjectRef {
            provider: PROVIDER,
            generation: self.generation,
            kind,
            id,
        }
    }
    fn validate(&self, reference: PcuObjectRef, kind: PcuObjectKind) -> Result<(), MlxError> {
        let length = if kind == PcuObjectKind::Target {
            1
        } else {
            self.devices.len()
        };
        if reference.provider != PROVIDER
            || reference.generation != self.generation
            || reference.kind != kind
            || reference.id as usize >= length
        {
            return Err(MlxError::InvalidRequest(
                "invalid or stale MLX reference".into(),
            ));
        }
        Ok(())
    }
    const fn readiness(&self) -> PcuProviderReadiness<'static> {
        PcuProviderReadiness {
            status: if self.devices.is_empty() {
                PcuProviderStatus::Unavailable
            } else {
                PcuProviderStatus::Ready
            },
            reason: None,
        }
    }
    fn support(&self) -> PcuSupport {
        let mut support = PcuSupport::unsupported();
        if !self.devices.is_empty() {
            support.caps = PcuCaps::ENUMERATE_EXECUTORS | PcuCaps::CLAIM_EXECUTOR;
            support.implementation = PcuImplementationKind::Native;
            support.executor_count = 1;
        }
        support
    }
}

impl PcuDeviceActivation for MlxDiscovery {
    type Session = MlxSession;
    type Error = MlxError;
    fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        let index = device.id as usize;
        let current = self.runtime.devices()?;
        if current.get(index) != self.devices.get(index) {
            return Err(MlxError::InvalidRequest(
                "MLX inventory changed after discovery".into(),
            ));
        }
        let identity = PcuDeviceIdentity::from_device_ref(device).ok_or(MlxError::InvalidExtent)?;
        self.runtime.open_gpu_with_identity(index, Some(identity))
    }
}

impl PcuRuntimeDiscovery for MlxDiscovery {
    type Error = MlxError;
    fn providers<'a>(
        &'a self,
        output: &mut [PcuProviderDescriptor<'a>],
    ) -> Result<usize, MlxError> {
        if let Some(slot) = output.first_mut() {
            *slot = PcuProviderDescriptor {
                id: PROVIDER,
                generation: self.generation,
                backend: "mlx",
                readiness: self.readiness(),
            };
        }
        Ok(1)
    }
    fn targets<'a>(
        &'a self,
        provider: PcuProviderId,
        generation: u64,
        output: &mut [PcuTargetDescriptor<'a>],
    ) -> Result<usize, MlxError> {
        self.validate(
            PcuObjectRef {
                provider,
                generation,
                kind: PcuObjectKind::Target,
                id: 0,
            },
            PcuObjectKind::Target,
        )?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuTargetDescriptor {
                reference: self.reference(PcuObjectKind::Target, 0),
                name: "MLX delegated GPU",
                readiness: self.readiness(),
            };
        }
        Ok(1)
    }
    fn devices<'a>(
        &'a self,
        target: PcuObjectRef,
        output: &mut [PcuDeviceDescriptor<'a>],
    ) -> Result<usize, MlxError> {
        self.validate(target, PcuObjectKind::Target)?;
        for (index, (slot, facts)) in output.iter_mut().zip(&self.devices).enumerate() {
            *slot = PcuDeviceDescriptor {
                reference: self.device_reference(index)?,
                target,
                name: &facts.name,
                class: PcuDeviceClass::Gpu,
                vendor: None,
                architecture: (!facts.architecture.is_empty())
                    .then_some(facts.architecture.as_str()),
                generation: None,
                location: None,
            };
        }
        Ok(self.devices.len())
    }
    fn contexts<'a>(
        &'a self,
        device: PcuObjectRef,
        output: &mut [PcuContextDescriptor<'a>],
    ) -> Result<usize, MlxError> {
        self.validate(device, PcuObjectKind::Device)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuContextDescriptor {
                reference: self.reference(PcuObjectKind::Context, device.id),
                device,
                name: "MLX explicit GPU stream",
                kind: PcuContextKind::Compute,
            };
        }
        Ok(1)
    }
    fn memory_domains<'a>(
        &'a self,
        context: PcuObjectRef,
        output: &mut [PcuMemoryDomainDescriptor<'a>],
    ) -> Result<usize, MlxError> {
        self.validate(context, PcuObjectKind::Context)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuMemoryDomainDescriptor {
                reference: self.reference(PcuObjectKind::MemoryDomain, context.id),
                context,
                name: "MLX delegated backing",
                kind: PcuMemoryDomainKind::Other,
                capacity_bytes: None,
            };
        }
        Ok(1)
    }
    fn target_capabilities(&self, target: PcuObjectRef) -> Result<PcuCapabilitySnapshot, MlxError> {
        self.validate(target, PcuObjectKind::Target)?;
        Ok(PcuCapabilitySnapshot {
            support: self.support(),
        })
    }
    fn device_capabilities(&self, device: PcuObjectRef) -> Result<PcuCapabilitySnapshot, MlxError> {
        self.validate(device, PcuObjectKind::Device)?;
        Ok(PcuCapabilitySnapshot {
            support: self.support(),
        })
    }
    fn executors(
        &self,
        object: PcuObjectRef,
        output: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, MlxError> {
        if object.kind == PcuObjectKind::MemoryDomain {
            return Err(MlxError::InvalidRequest(
                "memory domain is not an executor".into(),
            ));
        }
        self.validate(object, object.kind)?;
        if self.devices.is_empty() {
            return Ok(0);
        }
        if let Some(slot) = output.first_mut() {
            *slot = PcuExecutorDescriptor {
                id: EXECUTOR,
                name: "MLX delegated tensor executor",
                class: PcuExecutorClass::Adapter,
                origin: PcuExecutorOrigin::TopologyBound,
                support: PcuExecutorSupport::unsupported(),
            };
        }
        Ok(1)
    }
}

#[cfg(feature = "tensor")]
#[path = "offers/offers.rs"]
mod offers;
#[cfg(feature = "tensor")]
pub use offers::MlxMatmulRequest;
