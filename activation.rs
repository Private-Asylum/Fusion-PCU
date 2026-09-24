//! Fixed-capacity routing from discovered devices to statically linked activation backends.
//!
//! The router stores provider identities only. It does not erase or own backend sessions: the
//! caller uses the returned [`PcuBackendToken`] to select a concrete backend and keeps that
//! backend's typed [`crate::PcuDeviceActivation::Session`] in a consumer-owned enum or value.
//! This keeps heterogeneous session ownership explicit without allocation or a core-wide enum.

use crate::{
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderId,
};

/// Explicit route to one statically linked backend registered in a router.
///
/// `slot` is the registration order within the router. Consumers can match it against their
/// own typed backend collection; `provider` remains available for diagnostics and validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PcuBackendToken {
    provider: PcuProviderId,
    slot: usize,
}

impl PcuBackendToken {
    pub const fn provider(self) -> PcuProviderId {
        self.provider
    }

    pub const fn slot(self) -> usize {
        self.slot
    }
}

/// Why a provider could not be registered or a discovered device could not be routed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcuActivationRouteError {
    DuplicateProvider(PcuProviderId),
    CapacityExhausted(PcuProviderId),
    UnknownProvider(PcuProviderId),
    NotADevice(PcuObjectKind),
}

/// Bounded, allocation-free routing table for statically linked activation backends.
///
/// Register providers in the same order as the consumer's typed backend mapping. Device
/// references carry a provider ID, so [`route`](Self::route) returns the corresponding slot.
/// The selected backend must still validate generation and device identity in `open_device`,
/// as required by [`crate::PcuDeviceActivation`].
pub struct PcuRuntimeActivationRouter<const N: usize> {
    providers: [Option<PcuProviderId>; N],
    len: usize,
}

impl<const N: usize> PcuRuntimeActivationRouter<N> {
    pub const fn new() -> Self {
        Self {
            providers: [None; N],
            len: 0,
        }
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Registers a provider and returns the token its typed consumer mapping should use.
    pub fn register(
        &mut self,
        provider: PcuProviderId,
    ) -> Result<PcuBackendToken, PcuActivationRouteError> {
        if self.providers[..self.len].contains(&Some(provider)) {
            return Err(PcuActivationRouteError::DuplicateProvider(provider));
        }
        if self.len == N {
            return Err(PcuActivationRouteError::CapacityExhausted(provider));
        }
        let slot = self.len;
        self.providers[slot] = Some(provider);
        self.len += 1;
        Ok(PcuBackendToken { provider, slot })
    }

    /// Routes a discovered device reference to its registered statically linked backend.
    ///
    /// This checks the object category and provider registration. The selected backend remains
    /// responsible for validating the generation and exact device ID before activation.
    pub fn route(&self, device: PcuObjectRef) -> Result<PcuBackendToken, PcuActivationRouteError> {
        if device.kind != PcuObjectKind::Device {
            return Err(PcuActivationRouteError::NotADevice(device.kind));
        }
        let slot = self.providers[..self.len]
            .iter()
            .position(|provider| *provider == Some(device.provider))
            .ok_or(PcuActivationRouteError::UnknownProvider(device.provider))?;
        Ok(PcuBackendToken {
            provider: device.provider,
            slot,
        })
    }
}

impl<const N: usize> Default for PcuRuntimeActivationRouter<N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PcuDeviceActivation;

    fn device(provider: u32, kind: PcuObjectKind) -> PcuObjectRef {
        PcuObjectRef {
            provider: PcuProviderId(provider),
            generation: 7,
            kind,
            id: 42,
        }
    }

    #[test]
    fn routes_devices_by_provider_to_consumer_owned_slots() {
        let mut router = PcuRuntimeActivationRouter::<2>::new();
        let first = router.register(PcuProviderId(10)).unwrap();
        let second = router.register(PcuProviderId(20)).unwrap();

        assert_eq!(first.slot(), 0);
        assert_eq!(first.provider(), PcuProviderId(10));
        assert_eq!(router.route(device(20, PcuObjectKind::Device)), Ok(second));
    }

    #[test]
    fn rejects_non_devices_unknown_and_duplicate_or_over_capacity_providers() {
        let mut router = PcuRuntimeActivationRouter::<1>::new();
        router.register(PcuProviderId(10)).unwrap();

        assert_eq!(
            router.route(device(10, PcuObjectKind::Target)),
            Err(PcuActivationRouteError::NotADevice(PcuObjectKind::Target))
        );
        assert_eq!(
            router.route(device(20, PcuObjectKind::Device)),
            Err(PcuActivationRouteError::UnknownProvider(PcuProviderId(20)))
        );
        assert_eq!(
            router.register(PcuProviderId(10)),
            Err(PcuActivationRouteError::DuplicateProvider(PcuProviderId(
                10
            )))
        );
        assert_eq!(
            router.register(PcuProviderId(20)),
            Err(PcuActivationRouteError::CapacityExhausted(PcuProviderId(
                20
            )))
        );
    }

    struct CpuBackend;
    struct CpuSession {
        device_id: u32,
    }

    impl PcuDeviceActivation for CpuBackend {
        type Session = CpuSession;
        type Error = &'static str;

        fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
            validate_device(device, PcuProviderId(10), 7)?;
            Ok(CpuSession {
                device_id: device.id,
            })
        }
    }

    struct GpuBackend;
    struct GpuSession {
        device_id: u32,
        queue_count: u8,
    }

    impl PcuDeviceActivation for GpuBackend {
        type Session = GpuSession;
        type Error = &'static str;

        fn open_device(&self, device: PcuObjectRef) -> Result<Self::Session, Self::Error> {
            validate_device(device, PcuProviderId(20), 12)?;
            Ok(GpuSession {
                device_id: device.id,
                queue_count: 4,
            })
        }
    }

    fn validate_device(
        device: PcuObjectRef,
        expected_provider: PcuProviderId,
        current_generation: u64,
    ) -> Result<(), &'static str> {
        if device.kind != PcuObjectKind::Device || device.provider != expected_provider {
            return Err("wrong device");
        }
        if device.generation != current_generation {
            return Err("stale generation");
        }
        Ok(())
    }

    enum ConsumerSession {
        Cpu(CpuSession),
        Gpu(GpuSession),
    }

    fn activate(
        token: PcuBackendToken,
        device: PcuObjectRef,
        cpu: &CpuBackend,
        gpu: &GpuBackend,
    ) -> Result<ConsumerSession, &'static str> {
        match token.slot() {
            0 => cpu.open_device(device).map(ConsumerSession::Cpu),
            1 => gpu.open_device(device).map(ConsumerSession::Gpu),
            _ => Err("unknown backend slot"),
        }
    }

    #[test]
    fn consumer_dispatches_into_its_typed_session_enum_and_backend_rejects_stale_refs() {
        let mut router = PcuRuntimeActivationRouter::<2>::new();
        router.register(PcuProviderId(10)).unwrap();
        router.register(PcuProviderId(20)).unwrap();
        let cpu = CpuBackend;
        let gpu = GpuBackend;

        let gpu_device = PcuObjectRef {
            provider: PcuProviderId(20),
            generation: 12,
            kind: PcuObjectKind::Device,
            id: 99,
        };
        let token = router.route(gpu_device).unwrap();
        let session = activate(token, gpu_device, &cpu, &gpu).unwrap();
        match session {
            ConsumerSession::Gpu(session) => {
                assert_eq!(session.device_id, 99);
                assert_eq!(session.queue_count, 4);
            }
            ConsumerSession::Cpu(_) => panic!("router selected the wrong typed backend"),
        }

        let cpu_device = PcuObjectRef {
            provider: PcuProviderId(10),
            generation: 7,
            kind: PcuObjectKind::Device,
            id: 31,
        };
        match activate(router.route(cpu_device).unwrap(), cpu_device, &cpu, &gpu).unwrap() {
            ConsumerSession::Cpu(session) => assert_eq!(session.device_id, 31),
            ConsumerSession::Gpu(_) => panic!("router selected the wrong typed backend"),
        }

        let stale_device = PcuObjectRef {
            generation: 11,
            ..gpu_device
        };
        let stale_token = router.route(stale_device).unwrap();
        assert_eq!(
            activate(stale_token, stale_device, &cpu, &gpu).err(),
            Some("stale generation")
        );
    }
}
