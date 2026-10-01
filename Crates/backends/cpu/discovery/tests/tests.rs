use super::*;

#[test]
fn logical_snapshot_validates_identity_and_reports_unknown_physical_facts() {
    let discovery = PcuCpuDiscovery::discover().unwrap();
    let device = discovery.device_reference();
    assert_eq!(discovery.providers(&mut []).unwrap(), 1);
    assert_eq!(
        discovery
            .targets(device.provider, device.generation, &mut [])
            .unwrap(),
        1
    );
    assert_eq!(
        discovery
            .devices(discovery.reference(PcuObjectKind::Target), &mut [])
            .unwrap(),
        1
    );
    assert_eq!(discovery.contexts(device, &mut []).unwrap(), 1);
    assert_eq!(
        discovery
            .memory_domains(discovery.reference(PcuObjectKind::Context), &mut [])
            .unwrap(),
        1
    );
    assert_eq!(discovery.executors(device, &mut []).unwrap(), 1);
    assert_eq!(
        discovery.device_facts(device).unwrap(),
        fusion_pcu::PcuDeviceFacts::default()
    );
    discovery.open_device(device).unwrap();
    let mut domains = [PcuMemoryDomainDescriptor {
        reference: device,
        context: device,
        name: "fixture",
        kind: PcuMemoryDomainKind::HostVisible,
        capacity_bytes: Some(1),
    }];
    discovery
        .memory_domains(discovery.reference(PcuObjectKind::Context), &mut domains)
        .unwrap();
    assert_eq!(domains[0].capacity_bytes, None);
    for invalid in [
        PcuObjectRef {
            generation: device.generation + 1,
            ..device
        },
        PcuObjectRef {
            provider: PcuProviderId(0),
            ..device
        },
        PcuObjectRef {
            kind: PcuObjectKind::Context,
            ..device
        },
        PcuObjectRef { id: 1, ..device },
    ] {
        assert!(discovery.open_device(invalid).is_err());
        assert!(discovery.device_facts(invalid).is_err());
        assert!(discovery.device_capabilities(invalid).is_err());
    }
    let second = PcuCpuDiscovery::discover().unwrap();
    assert_ne!(device.generation, second.device_reference().generation);
    assert!(second.open_device(device).is_err());
    assert!(
        discovery
            .executors(discovery.reference(PcuObjectKind::MemoryDomain), &mut [])
            .is_err()
    );
    let support = discovery.device_capabilities(device).unwrap().support;
    assert_eq!(support.executor_count, 1);
    assert_eq!(
        caps::EXECUTORS[0].origin,
        fusion_pcu::PcuExecutorOrigin::Synthetic
    );
}
