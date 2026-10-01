//! Generation-scoped, bounded physical Vulkan discovery and explicit device activation.

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
    PcuDeviceDescriptor,
    PcuDeviceFacts,
    PcuDeviceIdentity,
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
    PcuSupport,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use crate::{
    ffi::{
        VulkanDevice,
        VulkanNativeDevice,
    },
    PcuVulkanBackend,
    PcuVulkanDescriptorHeapBudget,
    PcuVulkanError,
    PcuVulkanCaps,
};
#[path = "caps/caps.rs"]
mod caps;

const PROVIDER: PcuProviderId = PcuProviderId(0x564b_4c31);
static GENERATION: AtomicU64 = AtomicU64::new(1);

/// A cold physical snapshot. Discovery activates no logical device, queue or allocation.
pub struct PcuVulkanDiscovery {
    generation: u64,
    devices: Vec<VulkanNativeDevice>,
}

impl PcuVulkanDiscovery {
    /// Queries the native compute inventory. CPU software devices are excluded.
    ///
    /// # Errors
    /// Returns loader/inventory or generation exhaustion errors.
    pub fn discover() -> Result<Self, PcuVulkanError> {
        let devices = VulkanNativeDevice::discover()?;
        let generation = GENERATION
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| PcuVulkanError::DiscoveryGenerationExhausted)?;
        Ok(Self {
            generation,
            devices,
        })
    }

    /// Returns native query facts without claiming the optional features are enabled.
    ///
    /// # Errors
    /// Rejects stale or non-device discovery references.
    pub fn native_caps(&self, device: PcuObjectRef) -> Result<PcuVulkanCaps, PcuVulkanError> {
        self.validate(device, PcuObjectKind::Device)?;
        Ok(self.devices[device.id as usize].caps)
    }

    const fn reference(&self, kind: PcuObjectKind, id: u32) -> PcuObjectRef {
        PcuObjectRef {
            provider: PROVIDER,
            generation: self.generation,
            kind,
            id,
        }
    }

    fn validate(&self, object: PcuObjectRef, kind: PcuObjectKind) -> Result<(), PcuVulkanError> {
        let bound = if kind == PcuObjectKind::Target {
            1
        } else {
            self.devices.len()
        };
        if object.provider != PROVIDER
            || object.generation != self.generation
            || object.kind != kind
            || object.id as usize >= bound
        {
            return Err(PcuVulkanError::InvalidDiscoveryReference);
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

impl PcuDeviceActivation for PcuVulkanDiscovery {
    type Session = PcuVulkanBackend;
    type Error = PcuVulkanError;

    fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        let identity = PcuDeviceIdentity::from_device_ref(device)
            .ok_or(PcuVulkanError::InvalidDiscoveryReference)?;
        let native = VulkanDevice::open_native(
            PcuVulkanDescriptorHeapBudget::empty(),
            Some(&self.devices[device.id as usize]),
        )?;
        Ok(PcuVulkanBackend::discovered(native, identity))
    }
}

impl PcuRuntimeDiscovery for PcuVulkanDiscovery {
    type Error = PcuVulkanError;

    fn providers<'a>(
        &'a self,
        output: &mut [PcuProviderDescriptor<'a>],
    ) -> Result<usize, Self::Error> {
        if let Some(slot) = output.first_mut() {
            *slot = PcuProviderDescriptor {
                id: PROVIDER,
                generation: self.generation,
                backend: "vulkan",
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
                reference: self.reference(PcuObjectKind::Target, 0),
                name: "Vulkan compute",
                readiness: self.readiness(),
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
        for (id, (slot, device)) in output.iter_mut().zip(&self.devices).enumerate() {
            *slot = PcuDeviceDescriptor {
                reference: self.reference(
                    PcuObjectKind::Device,
                    u32::try_from(id).map_err(|_| PcuVulkanError::InvalidDiscoveryReference)?,
                ),
                target,
                name: &device.name,
                class: device.class,
                vendor: device.vendor,
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
    ) -> Result<usize, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        if let Some(slot) = output.first_mut() {
            *slot = PcuContextDescriptor {
                reference: self.reference(PcuObjectKind::Context, device.id),
                device,
                name: "Vulkan prepared bit-map compute queue",
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
                reference: self.reference(PcuObjectKind::MemoryDomain, context.id),
                context,
                name: "Vulkan host-visible coherent storage",
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
            support: if self.devices.is_empty() {
                PcuSupport::unsupported()
            } else {
                caps::support(self.devices.iter().any(|device| device.caps.shader_float64))
            },
        })
    }

    fn device_capabilities(
        &self,
        device: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        Ok(PcuCapabilitySnapshot {
            support: caps::support(self.devices[device.id as usize].caps.shader_float64),
        })
    }

    fn device_facts(&self, device: PcuObjectRef) -> Result<PcuDeviceFacts, Self::Error> {
        self.validate(device, PcuObjectKind::Device)?;
        Ok(self.devices[device.id as usize].facts)
    }

    fn executors(
        &self,
        object: PcuObjectRef,
        output: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, Self::Error> {
        if object.kind == PcuObjectKind::MemoryDomain {
            return Err(PcuVulkanError::InvalidDiscoveryReference);
        }
        self.validate(object, object.kind)?;
        if self.devices.is_empty() {
            return Ok(0);
        }
        if let Some(slot) = output.first_mut() {
            let float64 = if object.kind == PcuObjectKind::Target {
                self.devices.iter().any(|device| device.caps.shader_float64)
            } else {
                self.devices[object.id as usize].caps.shader_float64
            };
            *slot = caps::executor(float64);
        }
        Ok(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot_is_bounded_and_all_references_are_validated() {
        let discovery = PcuVulkanDiscovery {
            generation: 7,
            devices: Vec::new(),
        };
        assert_eq!(discovery.providers(&mut []).unwrap(), 1);
        assert_eq!(discovery.targets(PROVIDER, 7, &mut []).unwrap(), 1);
        assert_eq!(
            discovery
                .devices(discovery.reference(PcuObjectKind::Target, 0), &mut [])
                .unwrap(),
            0
        );
        for reference in [
            PcuObjectRef {
                provider: PcuProviderId(3),
                ..discovery.reference(PcuObjectKind::Target, 0)
            },
            PcuObjectRef {
                generation: 8,
                ..discovery.reference(PcuObjectKind::Target, 0)
            },
            discovery.reference(PcuObjectKind::Device, 0),
            discovery.reference(PcuObjectKind::Target, 1),
        ] {
            assert!(discovery.target_capabilities(reference).is_err());
        }
        assert!(
            discovery
                .open_device(discovery.reference(PcuObjectKind::Device, 0))
                .is_err()
        );
    }
}
