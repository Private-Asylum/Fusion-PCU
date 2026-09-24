//! Inspect ROCm through PCU's consumer-facing discovery registry.

use fusion_pcu::{
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscoveryRegistry,
    PcuTargetDescriptor,
};
use fusion_pcu_rocm::RocmDiscovery;

fn main() {
    if let Err(error) = run() {
        eprintln!("ROCm discovery failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let rocm = RocmDiscovery::new();
    let mut registry = PcuRuntimeDiscoveryRegistry::<4>::new();
    registry.register(&rocm).map_err(report)?;

    let provider_count = registry.providers(&mut []).map_err(report)?;
    let mut providers = vec![empty_provider(); provider_count];
    registry.providers(&mut providers).map_err(report)?;
    for provider in providers {
        println!(
            "backend {}: {:?}{}",
            provider.backend,
            provider.readiness.status,
            provider
                .readiness
                .reason
                .map(|reason| format!(" ({reason})"))
                .unwrap_or_default()
        );
        let target_count = registry
            .targets(provider.id, provider.generation, &mut [])
            .map_err(report)?;
        let mut targets = vec![empty_target(); target_count];
        registry
            .targets(provider.id, provider.generation, &mut targets)
            .map_err(report)?;
        for target in targets {
            println!("  target {}: {:?}", target.name, target.readiness.status);
            let count = registry
                .devices(target.reference, &mut [])
                .map_err(report)?;
            let mut devices = vec![empty_device(); count];
            registry
                .devices(target.reference, &mut devices)
                .map_err(report)?;
            for device in devices {
                let caps = registry
                    .device_capabilities(device.reference)
                    .map_err(report)?;
                println!(
                    "    device {}: vendor={:?} architecture={:?} generation={:?} location={:?} caps={:#x}",
                    device.name,
                    device.vendor,
                    device.architecture,
                    device.generation,
                    device.location,
                    caps.support.caps.bits()
                );
            }
        }
    }
    Ok(())
}

fn report(error: fusion_pcu::PcuRegistryError) -> String {
    format!(
        "provider {:?} {:?}: {}{}",
        error.provider,
        error.operation,
        error.message(),
        if error.truncated { "…" } else { "" }
    )
}

const EMPTY_REF: PcuObjectRef = PcuObjectRef {
    provider: PcuProviderId(0),
    generation: 0,
    kind: PcuObjectKind::Target,
    id: 0,
};

fn empty_provider<'a>() -> PcuProviderDescriptor<'a> {
    PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }
}

fn empty_target<'a>() -> PcuTargetDescriptor<'a> {
    PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }
}

fn empty_device<'a>() -> PcuDeviceDescriptor<'a> {
    PcuDeviceDescriptor {
        reference: EMPTY_REF,
        target: EMPTY_REF,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }
}
