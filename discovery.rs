//! Bounded, backend-neutral runtime discovery contracts.
//!
//! Discovery is deliberately separate from backend preference and selection. Implementations
//! describe what is present; callers provide bounded storage and decide what to use.

use crate::{
    PcuExecutorDescriptor,
    PcuSupport,
};

/// Stable identifier for a provider within one process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuProviderId(pub u32);

/// Discovered object category, preventing equal numeric ids for different categories from aliasing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuObjectKind {
    Target,
    Device,
    Context,
    MemoryDomain,
}

/// Identity of a discovered object. `id` is only meaningful for its provider and generation.
/// A generation change invalidates references from the previous enumeration snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuObjectRef {
    pub provider: PcuProviderId,
    pub generation: u64,
    pub kind: PcuObjectKind,
    pub id: u32,
}

/// Opens an explicitly selected, discovered device into a backend-owned session or lease.
///
/// Implementations must validate the reference's provider, generation, kind, and identifier
/// against their current discovery state before opening anything. In particular, references
/// whose `kind` is not [`PcuObjectKind::Device`] and references from an older generation must
/// be rejected. Callers choose the device themselves; this contract defines no preference or
/// automatic selection policy.
///
/// `Session` is intentionally unconstrained: backends may return an owned, non-`Copy` resource
/// whose drop releases the opened device/session.
pub trait PcuDeviceActivation {
    /// Backend-specific opened session or lease.
    type Session;

    /// Backend-specific activation error.
    type Error;

    /// Opens the device named by one explicit discovery reference.
    fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error>;
}

/// Current provider availability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuProviderStatus {
    Ready,
    Degraded,
    Unavailable,
}

/// Honest status and optional human-readable explanation for one provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuProviderReadiness<'a> {
    pub status: PcuProviderStatus,
    pub reason: Option<&'a str>,
}

/// Runtime-discovered provider descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuProviderDescriptor<'a> {
    pub id: PcuProviderId,
    pub generation: u64,
    pub backend: &'a str,
    pub readiness: PcuProviderReadiness<'a>,
}

/// Runtime-discovered backend target (for example, an API instance or accelerator adapter).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuTargetDescriptor<'a> {
    pub reference: PcuObjectRef,
    pub name: &'a str,
    pub readiness: PcuProviderReadiness<'a>,
}

/// Broad device category; it makes no claim about relative performance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDeviceClass {
    Cpu,
    Gpu,
    Accelerator,
    Io,
    Other,
}

/// Optional device location supplied by the discovering backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuDeviceLocation<'a> {
    PciBusId(&'a str),
    Platform(&'a str),
    Remote(&'a str),
}

/// Runtime-discovered device descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuDeviceDescriptor<'a> {
    pub reference: PcuObjectRef,
    pub target: PcuObjectRef,
    pub name: &'a str,
    pub class: PcuDeviceClass,
    /// Reported hardware vendor, when the provider can identify it.
    pub vendor: Option<&'a str>,
    /// Backend-native architecture identifier, when available.
    pub architecture: Option<&'a str>,
    /// Hardware generation, when reported directly rather than guessed from a model name.
    pub generation: Option<&'a str>,
    /// Backend-reported location for correlating with host topology or display facts.
    pub location: Option<PcuDeviceLocation<'a>>,
}

/// Broad execution-context category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuContextKind {
    General,
    Compute,
    Transfer,
    Other,
}

/// Runtime-discovered execution context descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuContextDescriptor<'a> {
    pub reference: PcuObjectRef,
    pub device: PcuObjectRef,
    pub name: &'a str,
    pub kind: PcuContextKind,
}

/// Broad memory-domain category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PcuMemoryDomainKind {
    DeviceLocal,
    HostVisible,
    Unified,
    Other,
}

/// Runtime-discovered memory domain. `capacity_bytes` is absent when unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuMemoryDomainDescriptor<'a> {
    pub reference: PcuObjectRef,
    pub context: PcuObjectRef,
    pub name: &'a str,
    pub kind: PcuMemoryDomainKind,
    pub capacity_bytes: Option<u64>,
}

/// Capability snapshot for one discovered target or device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuCapabilitySnapshot {
    pub support: PcuSupport,
}

/// Runtime discovery contract with bounded caller-owned enumeration buffers.
///
/// Enumeration methods return the total number of required slots, even when the supplied
/// buffer is too small; implementations fill the available prefix. Names are borrowed from the
/// provider and therefore need no allocation in the caller. Object references are valid only
/// within their provider and generation.
pub trait PcuRuntimeDiscovery {
    /// Provider-specific error type, preserving backend diagnostics.
    type Error;

    /// Enumerates available providers.
    fn providers<'a>(
        &'a self,
        output: &mut [PcuProviderDescriptor<'a>],
    ) -> Result<usize, Self::Error>;

    /// Enumerates backend targets for one provider generation.
    fn targets<'a>(
        &'a self,
        provider: PcuProviderId,
        generation: u64,
        output: &mut [PcuTargetDescriptor<'a>],
    ) -> Result<usize, Self::Error>;

    /// Enumerates devices for one target.
    fn devices<'a>(
        &'a self,
        target: PcuObjectRef,
        output: &mut [PcuDeviceDescriptor<'a>],
    ) -> Result<usize, Self::Error>;

    /// Enumerates contexts for one device.
    fn contexts<'a>(
        &'a self,
        device: PcuObjectRef,
        output: &mut [PcuContextDescriptor<'a>],
    ) -> Result<usize, Self::Error>;

    /// Enumerates memory domains for one context.
    fn memory_domains<'a>(
        &'a self,
        context: PcuObjectRef,
        output: &mut [PcuMemoryDomainDescriptor<'a>],
    ) -> Result<usize, Self::Error>;

    /// Queries target-level capabilities.
    fn target_capabilities(
        &self,
        target: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error>;

    /// Queries device-specific capabilities and limits.
    fn device_capabilities(
        &self,
        device: PcuObjectRef,
    ) -> Result<PcuCapabilitySnapshot, Self::Error>;

    /// Copies executor descriptors into caller storage and returns the required total count.
    fn executors(
        &self,
        object: PcuObjectRef,
        output: &mut [PcuExecutorDescriptor],
    ) -> Result<usize, Self::Error>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PcuError;

    struct ActivationMock;

    #[derive(Debug)]
    struct MockSession {
        device: PcuObjectRef,
    }

    impl Drop for MockSession {
        fn drop(&mut self) {}
    }

    impl PcuDeviceActivation for ActivationMock {
        type Session = MockSession;
        type Error = PcuError;

        fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
            if device.kind != PcuObjectKind::Device {
                return Err(PcuError::invalid());
            }
            if device.provider != PcuProviderId(3) || device.generation != 7 || device.id != 2 {
                return Err(PcuError::state_conflict());
            }
            Ok(MockSession { device })
        }
    }

    #[test]
    fn activation_requires_current_device_reference_and_returns_owned_session() {
        let activation = ActivationMock;
        let device = PcuObjectRef {
            provider: PcuProviderId(3),
            generation: 7,
            kind: PcuObjectKind::Device,
            id: 2,
        };
        let session = activation.open_device(device).unwrap();
        assert_eq!(session.device, device);
        // Moving the session is supported without imposing Copy or Clone on backend leases.
        let _moved_session = session;

        let stale = PcuObjectRef {
            generation: 6,
            ..device
        };
        assert_eq!(
            activation.open_device(stale).unwrap_err().kind(),
            crate::PcuErrorKind::StateConflict
        );
        let wrong_kind = PcuObjectRef {
            kind: PcuObjectKind::Context,
            ..device
        };
        assert_eq!(
            activation.open_device(wrong_kind).unwrap_err().kind(),
            crate::PcuErrorKind::Invalid
        );
    }

    struct Mock;

    impl PcuRuntimeDiscovery for Mock {
        type Error = PcuError;
        fn providers<'a>(
            &'a self,
            output: &mut [PcuProviderDescriptor<'a>],
        ) -> Result<usize, PcuError> {
            let count = 1;
            if let Some(slot) = output.first_mut() {
                *slot = PcuProviderDescriptor {
                    id: PcuProviderId(3),
                    generation: 7,
                    backend: "mock",
                    readiness: PcuProviderReadiness {
                        status: PcuProviderStatus::Ready,
                        reason: None,
                    },
                };
            }
            Ok(count)
        }
        fn targets<'a>(
            &'a self,
            provider: PcuProviderId,
            generation: u64,
            output: &mut [PcuTargetDescriptor<'a>],
        ) -> Result<usize, PcuError> {
            if provider != PcuProviderId(3) || generation != 7 {
                return Err(PcuError::invalid());
            }
            if let Some(slot) = output.first_mut() {
                *slot = PcuTargetDescriptor {
                    reference: PcuObjectRef {
                        provider,
                        generation,
                        kind: PcuObjectKind::Target,
                        id: 0,
                    },
                    name: "target",
                    readiness: PcuProviderReadiness {
                        status: PcuProviderStatus::Ready,
                        reason: None,
                    },
                };
            }
            Ok(1)
        }
        fn devices<'a>(
            &'a self,
            target: PcuObjectRef,
            output: &mut [PcuDeviceDescriptor<'a>],
        ) -> Result<usize, PcuError> {
            if target.generation != 7 {
                return Err(PcuError::invalid());
            }
            if let Some(slot) = output.first_mut() {
                *slot = PcuDeviceDescriptor {
                    reference: PcuObjectRef {
                        id: 2,
                        kind: PcuObjectKind::Device,
                        ..target
                    },
                    target,
                    name: "device",
                    class: PcuDeviceClass::Gpu,
                    vendor: Some("mock vendor"),
                    architecture: None,
                    generation: None,
                    location: None,
                };
            }
            Ok(1)
        }
        fn contexts<'a>(
            &'a self,
            _: PcuObjectRef,
            _: &mut [PcuContextDescriptor<'a>],
        ) -> Result<usize, PcuError> {
            Ok(0)
        }
        fn memory_domains<'a>(
            &'a self,
            _: PcuObjectRef,
            _: &mut [PcuMemoryDomainDescriptor<'a>],
        ) -> Result<usize, PcuError> {
            Ok(0)
        }
        fn target_capabilities(&self, _: PcuObjectRef) -> Result<PcuCapabilitySnapshot, PcuError> {
            Ok(PcuCapabilitySnapshot {
                support: PcuSupport::unsupported(),
            })
        }
        fn device_capabilities(&self, _: PcuObjectRef) -> Result<PcuCapabilitySnapshot, PcuError> {
            self.target_capabilities(PcuObjectRef {
                provider: PcuProviderId(3),
                generation: 7,
                kind: PcuObjectKind::Target,
                id: 0,
            })
        }
        fn executors(
            &self,
            _: PcuObjectRef,
            _: &mut [PcuExecutorDescriptor],
        ) -> Result<usize, PcuError> {
            Ok(0)
        }
    }

    #[test]
    fn enumeration_reports_required_capacity_and_scopes_refs_to_generation() {
        let mock = Mock;
        assert_eq!(mock.providers(&mut []).unwrap(), 1);
        let mut targets = [PcuTargetDescriptor {
            reference: PcuObjectRef {
                provider: PcuProviderId(0),
                generation: 0,
                kind: PcuObjectKind::Target,
                id: 0,
            },
            name: "",
            readiness: PcuProviderReadiness {
                status: PcuProviderStatus::Unavailable,
                reason: None,
            },
        }];
        assert_eq!(mock.targets(PcuProviderId(3), 7, &mut targets).unwrap(), 1);
        let mut devices = [];
        assert_eq!(mock.devices(targets[0].reference, &mut devices).unwrap(), 1);
        assert_eq!(
            mock.targets(PcuProviderId(3), 6, &mut targets)
                .unwrap_err()
                .kind(),
            crate::PcuErrorKind::Invalid
        );
    }
}
