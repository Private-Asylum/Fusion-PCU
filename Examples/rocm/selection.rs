//! Example-owned runtime policy for ranking discovered `ROCm` devices.

use std::error::Error;

use fusion_pcu::{
    PcuCaps,
    PcuContextDescriptor,
    PcuContextKind,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuMemoryDomainDescriptor,
    PcuMemoryDomainKind,
    PcuMemoryPoolId,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuTargetDescriptor,
};
use fusion_pcu_rocm::RocmDiscovery;

#[derive(Clone, Debug)]
#[allow(dead_code)] // A shared module is compiled separately by consumers with different fields.
pub struct Candidate {
    pub device: PcuObjectRef,
    pub pool: PcuMemoryPoolId,
    pub architecture: Option<String>,
    pub total_memory: u64,
    pub name: String,
}

pub fn preferred_device() -> Result<Option<u32>, Box<dyn Error>> {
    std::env::var("FUSION_ROCM_DEVICE")
        .ok()
        .map(|value| value.parse::<u32>().map_err(Into::into))
        .transpose()
}

#[allow(dead_code)] // The standalone smoke probe selects candidates without opening a PCU session.
pub fn open_ranked(
    discovery: &RocmDiscovery,
    candidates: Vec<Candidate>,
    block_size: u32,
) -> Result<(fusion_pcu_rocm::RocmOwnedDispatchBackend, Candidate), Box<dyn Error>> {
    let mut failures = Vec::new();
    for candidate in candidates {
        match fusion_pcu_rocm::RocmOwnedDispatchBackend::open(
            discovery,
            candidate.device,
            block_size,
        ) {
            Ok(session) => return Ok((session, candidate)),
            Err(error) => failures.push(format!("device {}: {error}", candidate.device.id)),
        }
    }
    Err(format!(
        "no capable ROCm device could be opened: {}",
        failures.join("; ")
    )
    .into())
}

/// Rank viable devices by physical memory, then stable runtime index.
///
/// A caller can supply `preferred` at runtime to require one exact device. This example policy
/// never silently substitutes a different device for an explicit request.
///
/// # Errors
///
/// Returns the discovery error or a summary when no candidate satisfies the requested profile.
pub fn ranked_devices(
    discovery: &RocmDiscovery,
    preferred: Option<u32>,
    require_dispatch: bool,
) -> Result<Vec<Candidate>, Box<dyn Error>> {
    let mut providers = [empty_provider()];
    discovery.providers(&mut providers)?;
    let mut targets = [empty_target()];
    discovery.targets(providers[0].id, providers[0].generation, &mut targets)?;
    let count = discovery.devices(targets[0].reference, &mut [])?;
    let mut devices = vec![empty_device(); count];
    discovery.devices(targets[0].reference, &mut devices)?;

    if count == 0 {
        return Err("no ROCm devices were discovered".into());
    }
    let mut ranked = Vec::new();
    let mut rejected = Vec::new();
    for device in devices {
        if preferred.is_some_and(|id| id != device.reference.id) {
            continue;
        }
        if require_dispatch {
            let caps = match discovery.device_capabilities(device.reference) {
                Ok(caps) => caps,
                Err(error) => {
                    rejected.push(format!(
                        "device {} capability query failed: {error}",
                        device.reference.id
                    ));
                    continue;
                }
            };
            let executor_count = match discovery.executors(device.reference, &mut []) {
                Ok(count) => count,
                Err(error) => {
                    rejected.push(format!(
                        "device {} executor query failed: {error}",
                        device.reference.id
                    ));
                    continue;
                }
            };
            if !caps.support.caps.contains(PcuCaps::COMPUTE_DISPATCH) || executor_count == 0 {
                rejected.push(format!(
                    "device {} lacks a ready compute executor",
                    device.reference.id
                ));
                continue;
            }
        }
        let mut contexts = [empty_context()];
        if !matches!(discovery.contexts(device.reference, &mut contexts), Ok(count) if count > 0) {
            rejected.push(format!(
                "device {} has no compute context",
                device.reference.id
            ));
            continue;
        }
        let mut domains = [empty_domain()];
        if !matches!(discovery.memory_domains(contexts[0].reference, &mut domains), Ok(count) if count > 0)
        {
            rejected.push(format!(
                "device {} has no memory domain",
                device.reference.id
            ));
            continue;
        }
        ranked.push(Candidate {
            device: device.reference,
            pool: PcuMemoryPoolId(domains[0].reference.id),
            architecture: device.architecture.map(str::to_owned),
            total_memory: domains[0].capacity_bytes.unwrap_or(0),
            name: device.name.to_owned(),
        });
    }
    ranked.sort_by(|a, b| {
        b.total_memory
            .cmp(&a.total_memory)
            .then_with(|| a.device.id.cmp(&b.device.id))
    });
    if ranked.is_empty() {
        let request = preferred.map_or_else(
            || "any ROCm device".to_owned(),
            |id| format!("ROCm device {id}"),
        );
        return Err(format!("no capable {request}: {}", rejected.join("; ")).into());
    }
    Ok(ranked)
}

const EMPTY_REF: PcuObjectRef = PcuObjectRef {
    provider: PcuProviderId(0),
    generation: 0,
    kind: PcuObjectKind::Target,
    id: 0,
};

const fn empty_provider<'a>() -> PcuProviderDescriptor<'a> {
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

const fn empty_target<'a>() -> PcuTargetDescriptor<'a> {
    PcuTargetDescriptor {
        reference: EMPTY_REF,
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }
}

const fn empty_device<'a>() -> PcuDeviceDescriptor<'a> {
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

const fn empty_context<'a>() -> PcuContextDescriptor<'a> {
    PcuContextDescriptor {
        reference: EMPTY_REF,
        device: EMPTY_REF,
        name: "",
        kind: PcuContextKind::Other,
    }
}

const fn empty_domain<'a>() -> PcuMemoryDomainDescriptor<'a> {
    PcuMemoryDomainDescriptor {
        reference: EMPTY_REF,
        context: EMPTY_REF,
        name: "",
        kind: PcuMemoryDomainKind::Other,
        capacity_bytes: None,
    }
}
