//! Selects an actual physical device and retains its stable UUID for independent native peers.
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanDiscovery};
#[rustfmt::skip]
use pcu_facade::{PcuDeviceClass,PcuDeviceDescriptor,PcuObjectKind,PcuObjectRef,PcuProviderDescriptor,PcuProviderId,PcuProviderReadiness,PcuProviderStatus,PcuRuntimeDiscovery,PcuStableDeviceIdentity,PcuTargetDescriptor};
pub fn selected() -> (PcuVulkanBackend, PcuStableDeviceIdentity) {
    let discovery = PcuVulkanDiscovery::discover().unwrap();
    let empty = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Target,
        id: 0,
    };
    let readiness = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness,
    }];
    discovery.providers(&mut providers).unwrap();
    let mut targets = [PcuTargetDescriptor {
        reference: empty,
        name: "",
        readiness,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [PcuDeviceDescriptor {
        reference: empty,
        target: empty,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }];
    assert!(
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap()
            > 0
    );
    let selected = devices[0].reference;
    (
        PcuVulkanBackend::open(&discovery, selected).unwrap(),
        discovery
            .device_facts(selected)
            .unwrap()
            .stable_identity
            .unwrap(),
    )
}
