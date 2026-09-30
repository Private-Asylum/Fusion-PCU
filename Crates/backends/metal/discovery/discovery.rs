//! Cold Metal discovery with generation-scoped physical references.

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
    PcuDeviceFacts,
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
    PcuStableDeviceIdentity,
    PcuSupport,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use crate::{
    MetalDeviceFacts,
    MetalError,
    MetalSession,
};

const PROVIDER: PcuProviderId = PcuProviderId(0x4d54_4c31);
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// Cold physical discovery snapshot. No queues, buffers or executables are opened.
///
/// Capabilities describe the bounded owned checked-map profile. Exact IR structure and policies
/// are admitted during preparation; discovery never compiles or activates an executable.
pub struct MetalDiscovery {
    generation: u64,
    devices: Vec<MetalDeviceFacts>,
}
impl MetalDiscovery {
    /// Activates a generation-bound owned dispatch session after physical identity validation.
    ///
    /// # Errors
    /// Returns stale reference, changed inventory, or native activation failure.
    pub fn open_owned_device(
        &self,
        device: PcuObjectRef,
    ) -> Result<crate::MetalOwnedDispatchBackend, MetalError> {
        let session = self.open_device(device)?;
        let identity = fusion_pcu::PcuDeviceIdentity::from_device_ref(device)
            .ok_or(MetalError::Unsupported)?;
        Ok(crate::MetalOwnedDispatchBackend::new(session, identity))
    }
    /// Reads the current native physical inventory.
    ///
    /// # Errors
    /// Returns Unsupported off macOS or a native inventory failure.
    pub fn discover() -> Result<Self, MetalError> {
        let devices = MetalSession::discover()?;
        u32::try_from(devices.len()).map_err(|_| MetalError::InvalidExtent)?;
        let generation = GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| MetalError::Runtime("Metal discovery generation exhausted".into()))?;
        Ok(Self {
            generation,
            devices,
        })
    }
    pub(crate) const fn reference(&self, kind: PcuObjectKind, id: u32) -> PcuObjectRef {
        PcuObjectRef {
            provider: PROVIDER,
            generation: self.generation,
            kind,
            id,
        }
    }
    fn validate(&self, reference: PcuObjectRef, kind: PcuObjectKind) -> Result<(), MetalError> {
        let bound = if kind == PcuObjectKind::Target {
            1
        } else {
            self.devices.len()
        };
        if reference.provider != PROVIDER
            || reference.generation != self.generation
            || reference.kind != kind
            || reference.id as usize >= bound
        {
            return Err(MetalError::Runtime(
                "invalid or stale Metal discovery reference".into(),
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
}
impl PcuDeviceActivation for MetalDiscovery {
    type Session = MetalSession;
    type Error = MetalError;
    fn open_device(&self, device: PcuObjectRef) -> Result<MetalSession, MetalError> {
        self.validate(device, PcuObjectKind::Device)?;
        let session = MetalSession::open(device.id as usize)?;
        if session.facts().registry_id != self.devices[device.id as usize].registry_id {
            return Err(MetalError::Runtime(
                "Metal inventory changed after discovery".into(),
            ));
        }
        Ok(session)
    }
}
impl PcuRuntimeDiscovery for MetalDiscovery {
    type Error = MetalError;
    fn providers<'a>(
        &'a self,
        output: &mut [PcuProviderDescriptor<'a>],
    ) -> Result<usize, MetalError> {
        if let Some(slot) = output.first_mut() {
            *slot = PcuProviderDescriptor {
                id: PROVIDER,
                generation: self.generation,
                backend: "metal",
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
    ) -> Result<usize, MetalError> {
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
                name: "Metal",
                readiness: self.readiness(),
            };
        }
        Ok(1)
    }
    fn devices<'a>(
        &'a self,
        target: PcuObjectRef,
        output: &mut [PcuDeviceDescriptor<'a>],
    ) -> Result<usize, MetalError> {
        self.validate(target, PcuObjectKind::Target)?;
        for (index, (slot, facts)) in output.iter_mut().zip(&self.devices).enumerate() {
            *slot = PcuDeviceDescriptor {
                reference: self.reference(
                    PcuObjectKind::Device,
                    u32::try_from(index).map_err(|_| MetalError::InvalidExtent)?,
                ),
                target,
                name: &facts.name,
                class: PcuDeviceClass::Gpu,
                vendor: None,
                architecture: None,
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
    ) -> Result<usize, MetalError> {
        self.validate(device, PcuObjectKind::Device)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuContextDescriptor {
                reference: self.reference(PcuObjectKind::Context, device.id),
                device,
                name: "Metal checked-map queue",
                kind: PcuContextKind::Compute,
            };
        }
        Ok(1)
    }
    fn memory_domains<'a>(
        &'a self,
        context: PcuObjectRef,
        output: &mut [PcuMemoryDomainDescriptor<'a>],
    ) -> Result<usize, MetalError> {
        self.validate(context, PcuObjectKind::Context)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuMemoryDomainDescriptor {
                reference: self.reference(PcuObjectKind::MemoryDomain, context.id),
                context,
                name: "Metal shared storage",
                kind: if self.devices[context.id as usize].unified_memory {
                    PcuMemoryDomainKind::Unified
                } else {
                    PcuMemoryDomainKind::HostVisible
                },
                capacity_bytes: None,
            };
        }
        Ok(1)
    }
    fn target_capabilities(
        &self,
        target: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, MetalError> {
        self.validate(target, PcuObjectKind::Target)?;
        Ok(PcuCapabilitySnapshot {
            support: if self.devices.is_empty() {
                PcuSupport::unsupported()
            } else {
                crate::owned_dispatch::discovery_support()
            },
        })
    }
    fn device_capabilities(
        &self,
        device: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, MetalError> {
        self.validate(device, PcuObjectKind::Device)?;
        Ok(PcuCapabilitySnapshot {
            support: crate::owned_dispatch::discovery_support(),
        })
    }
    fn device_facts(&self, device: PcuObjectRef) -> Result<PcuDeviceFacts, MetalError> {
        self.validate(device, PcuObjectKind::Device)?;
        let facts = &self.devices[device.id as usize];
        Ok(PcuDeviceFacts {
            stable_identity: Some(
                PcuStableDeviceIdentity::new("metal.registryID", &facts.registry_id.to_le_bytes())
                    .map_err(|_| MetalError::InvalidExtent)?,
            ),
            ..PcuDeviceFacts::default()
        })
    }
    fn executors(
        &self,
        object: PcuObjectRef,
        output: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, MetalError> {
        match object.kind {
            PcuObjectKind::Target | PcuObjectKind::Device | PcuObjectKind::Context => {
                self.validate(object, object.kind)?;
            }
            PcuObjectKind::MemoryDomain => return Err(MetalError::Unsupported),
        }
        if self.devices.is_empty() {
            return Ok(0);
        }
        let descriptors = crate::owned_dispatch::executor_descriptors();
        for (slot, descriptor) in output.iter_mut().zip(descriptors) {
            *slot = *descriptor;
        }
        Ok(descriptors.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn snapshot_rejects_stale_wrong_provider_kind_and_index() {
        let discovery = MetalDiscovery {
            generation: 17,
            devices: vec![MetalDeviceFacts {
                name: "fixture".into(),
                registry_id: 42,
                unified_memory: true,
                max_buffer_bytes: 1024,
            }],
        };
        let valid = discovery.reference(PcuObjectKind::Device, 0);
        assert!(
            discovery
                .device_facts(valid)
                .unwrap()
                .stable_identity
                .is_some()
        );
        for wrong in [
            PcuObjectRef {
                generation: 16,
                ..valid
            },
            PcuObjectRef {
                provider: PcuProviderId(0),
                ..valid
            },
            PcuObjectRef {
                kind: PcuObjectKind::Context,
                ..valid
            },
            PcuObjectRef { id: 1, ..valid },
        ] {
            assert!(discovery.device_capabilities(wrong).is_err());
            assert!(discovery.open_device(wrong).is_err());
        }
        assert_eq!(discovery.providers(&mut []).unwrap(), 1);
        assert_eq!(
            discovery
                .devices(discovery.reference(PcuObjectKind::Target, 0), &mut [])
                .unwrap(),
            1
        );
        assert_eq!(discovery.executors(valid, &mut []).unwrap(), 1);
        assert_eq!(discovery.contexts(valid, &mut []).unwrap(), 1);
        assert_eq!(
            discovery
                .memory_domains(discovery.reference(PcuObjectKind::Context, 0), &mut [])
                .unwrap(),
            1
        );
    }
}
