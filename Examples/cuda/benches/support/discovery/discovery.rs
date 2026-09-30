//! Hardware fixture shared by the native-driver and PCU benchmark routes.
#[rustfmt::skip]
use fusion_pcu_cuda::{
    CudaDiscovery,
    CudaOwnedDispatchBackend,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
};

pub fn selected_device() -> (CudaDiscovery, CudaOwnedDispatchBackend) {
    let discovery = CudaDiscovery::new();
    let invalid = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Device,
        id: 0,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(discovery.providers(&mut providers).unwrap(), 1);
    let mut targets = [PcuTargetDescriptor {
        reference: invalid,
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    assert_eq!(
        discovery
            .targets(providers[0].id, providers[0].generation, &mut targets)
            .unwrap(),
        1
    );
    let count = discovery.devices(targets[0].reference, &mut []).unwrap();
    assert!(count > 0, "test requires a visible CUDA device");
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: invalid,
            target: invalid,
            name: "",
            class: PcuDeviceClass::Other,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        };
        count
    ];
    discovery
        .devices(targets[0].reference, &mut devices)
        .unwrap();
    let selected = devices[0].reference;
    let info = discovery.device_info(selected).unwrap();
    eprintln!(
        "selected CUDA device: ordinal={}, name={}, architecture={:?}, block_size=256",
        selected.id, info.name, info.architecture
    );
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Cuda,
        device: Some(selected.id),
        block_size: 256,
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })
    .unwrap();
    let session = CudaOwnedDispatchBackend::open(&discovery, selected, 256)
        .expect("open selected CUDA device");
    (discovery, session)
}
