//! Hosted discovery of one logical CPU host execution device.

#[rustfmt::skip]
use std::sync::atomic::{
    AtomicU64,
    Ordering,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuCapabilitySnapshot,
    PcuContextDescriptor,
    PcuContextKind,
    PcuDeviceActivation,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuExecutorDescriptor,
    PcuMemoryDomainDescriptor,
    PcuMemoryDomainKind,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use crate::{
    PcuCpuFeatures,
    PcuCpuHostBackend,
    PcuCpuProcessor,
};

#[path = "caps/caps.rs"]
mod caps;

/// CPU provider namespace, distinct from GPU providers.
pub const PCU_CPU_PROVIDER: PcuProviderId = PcuProviderId(0x4350_5531);
static GENERATION: AtomicU64 = AtomicU64::new(1);
const READY: PcuProviderReadiness<'static> = PcuProviderReadiness {
    status: PcuProviderStatus::Ready,
    reason: None,
};

/// Structured logical discovery reference failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuCpuDiscoveryError {
    GenerationExhausted,
    InvalidReference {
        reference: PcuObjectRef,
        expected_kind: PcuObjectKind,
    },
    UnsupportedObject(PcuObjectRef),
}

/// One logical synchronous host executor. Physical topology and memory capacity remain unknown.
#[derive(Debug)]
pub struct PcuCpuDiscovery {
    generation: u64,
    processor: PcuCpuProcessor,
}
impl PcuCpuDiscovery {
    /// Captures a fresh generation and runtime instruction facts without executing a kernel.
    ///
    /// # Errors
    /// Returns generation exhaustion rather than recycling stale object identities.
    pub fn discover() -> Result<Self, PcuCpuDiscoveryError> {
        let generation = GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| PcuCpuDiscoveryError::GenerationExhausted)?;
        Ok(Self {
            generation,
            processor: PcuCpuProcessor::detect(),
        })
    }
    #[must_use]
    pub const fn device_reference(&self) -> PcuObjectRef {
        self.reference(PcuObjectKind::Device)
    }
    #[must_use]
    pub const fn processor_features(&self) -> PcuCpuFeatures {
        self.processor.features()
    }
    const fn reference(&self, kind: PcuObjectKind) -> PcuObjectRef {
        PcuObjectRef {
            provider: PCU_CPU_PROVIDER,
            generation: self.generation,
            kind,
            id: 0,
        }
    }
    fn validate(
        &self,
        reference: PcuObjectRef,
        kind: PcuObjectKind,
    ) -> Result<(), PcuCpuDiscoveryError> {
        if reference != self.reference(kind) {
            return Err(PcuCpuDiscoveryError::InvalidReference {
                reference,
                expected_kind: kind,
            });
        }
        Ok(())
    }
}
impl PcuDeviceActivation for PcuCpuDiscovery {
    type Session = PcuCpuHostBackend;
    type Error = PcuCpuDiscoveryError;
    fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        // The selected instructions were detected in this snapshot; no speed claim is implied.
        Ok(
            PcuCpuHostBackend::new(self.processor, self.processor.widest())
                .expect("detected CPU implementation"),
        )
    }
}
impl PcuRuntimeDiscovery for PcuCpuDiscovery {
    type Error = PcuCpuDiscoveryError;
    fn providers<'a>(
        &'a self,
        output: &mut [PcuProviderDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        if let Some(slot) = output.first_mut() {
            *slot = PcuProviderDescriptor {
                id: PCU_CPU_PROVIDER,
                generation: self.generation,
                backend: "cpu",
                readiness: READY,
            };
        }
        Ok(1)
    }
    fn targets<'a>(
        &'a self,
        provider: PcuProviderId,
        generation: u64,
        output: &mut [PcuTargetDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
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
                reference: self.reference(PcuObjectKind::Target),
                name: "CPU host",
                readiness: READY,
            };
        }
        Ok(1)
    }
    fn devices<'a>(
        &'a self,
        target: PcuObjectRef,
        output: &mut [PcuDeviceDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        self.validate(target, PcuObjectKind::Target)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuDeviceDescriptor {
                reference: self.device_reference(),
                target,
                name: "Logical CPU host",
                class: PcuDeviceClass::Cpu,
                vendor: None,
                architecture: None,
                generation: None,
                location: None,
            };
        }
        Ok(1)
    }
    fn contexts<'a>(
        &'a self,
        device: PcuObjectRef,
        output: &mut [PcuContextDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuContextDescriptor {
                reference: self.reference(PcuObjectKind::Context),
                device,
                name: "Synchronous CPU host",
                kind: PcuContextKind::Compute,
            };
        }
        Ok(1)
    }
    fn memory_domains<'a>(
        &'a self,
        context: PcuObjectRef,
        output: &mut [PcuMemoryDomainDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        self.validate(context, PcuObjectKind::Context)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuMemoryDomainDescriptor {
                reference: self.reference(PcuObjectKind::MemoryDomain),
                context,
                name: "Borrowed host memory",
                kind: PcuMemoryDomainKind::HostVisible,
                capacity_bytes: None,
            };
        }
        Ok(1)
    }
    fn target_capabilities(
        &self,
        target: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error> {
        self.validate(target, PcuObjectKind::Target)?;
        Ok(PcuCapabilitySnapshot {
            support: caps::support(),
        })
    }
    fn device_capabilities(
        &self,
        device: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        Ok(PcuCapabilitySnapshot {
            support: caps::support(),
        })
    }
    fn executors(
        &self,
        object: PcuObjectRef,
        output: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, Self::Error> {
        self.validate(object, object.kind)?;
        if object.kind == PcuObjectKind::MemoryDomain {
            return Err(PcuCpuDiscoveryError::UnsupportedObject(object));
        }
        if let Some(slot) = output.first_mut() {
            *slot = caps::EXECUTORS[0];
        }
        Ok(1)
    }
}

#[cfg(test)]
#[path = "tests/tests.rs"]
mod tests;
