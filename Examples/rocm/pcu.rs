//! Explicit `ROCm` selection followed by backend-neutral PCU memory and Dispatch contracts.

use std::process::ExitCode;
use std::num::NonZeroU32;

use fusion_pcu::{
    PcuContextDescriptor,
    PcuContextKind,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuMemoryDomainDescriptor,
    PcuMemoryDomainKind,
    PcuMemoryPoolId,
    PcuMemoryAccess,
    PcuMemoryAdmissionPolicy,
    PcuMemoryAllocationRequest,
    PcuMemoryHostAccess,
    PcuMemoryProvider,
    PcuMemoryRatio,
    PcuMemoryReservationLedger,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscoveryRegistry,
    PcuTargetDescriptor,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuInvocationParameters,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuOwnedDispatchMemorySession,
    PcuPreparedOwnedDispatch,
    PcuValueType,
    allocate_with_policy,
};
use fusion_pcu_rocm::{
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};
use fusion_pcu_macros::pcu_dispatch;

#[pcu_dispatch(invocations = 250)]
fn grid_stride_add<const N: usize>(input: &[f32], output: &mut [f32]) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < N {
        output[id] = ((input[id] + 2.5) * 2.0) / 2.0 - 1.0;
        id += stride;
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR fusion-rocm-pcu: {error}");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)] // Keep discovery, resource, and completion lifetimes visible end to end.
fn run() -> Result<(), Box<dyn std::error::Error>> {
    const COUNT: usize = 2048;
    const INVOCATIONS: u32 = 250;
    let rocm = RocmDiscovery::new();
    let available = discovered_devices(&rocm)?;
    let preferred = std::env::var("FUSION_ROCM_DEVICE")
        .ok()
        .map(|index| index.parse::<i32>())
        .transpose()?;
    let candidates = rank_devices(&available, preferred)?;
    let bindings = grid_stride_add_bindings();
    let builder = grid_stride_add::<COUNT>(&bindings)?;
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(INVOCATIONS).expect("nonzero")),
    };
    let mut selection_failures = Vec::new();
    let mut opened = None;
    for candidate in candidates {
        match RocmOwnedDispatchBackend::open(&rocm, candidate.reference, 64) {
            Ok(session) => {
                match session.prepare_dispatch_owned(submission, PcuInvocationParameters::empty()) {
                    Ok(prepared) => {
                        opened = Some((session, candidate.clone(), prepared));
                        break;
                    }
                    Err(error) => selection_failures.push(format!(
                        "device {} dispatch preparation: {error:?}",
                        candidate.reference.id
                    )),
                }
            }
            Err(error) => {
                selection_failures.push(format!("device {} open: {error}", candidate.reference.id));
            }
        }
    }
    let (session, selected, prepared) = opened.ok_or_else(|| {
        format!(
            "no ROCm device could prepare the dispatch: {}",
            selection_failures.join("; ")
        )
    })?;
    println!(
        "PCU selected ROCm device {} ({} bytes physical memory)",
        selected.reference.id, selected.total_memory
    );
    let input: Vec<f32> = (0..COUNT)
        .map(|index| f32::from(u16::try_from(index).expect("example index fits in u16")) + 1.25)
        .collect();
    let mut input_bytes = Vec::with_capacity(COUNT * 4);
    for value in &input {
        input_bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let pool_id = selected.pool;
    let mut memory_provider = PcuOwnedDispatchMemorySession::memory_provider(&session, pool_id);
    let mut memory_ledger = PcuMemoryReservationLedger::<2>::new();
    let memory_policy = PcuMemoryAdmissionPolicy {
        system_used: Some(PcuMemoryRatio::new(95, 100)),
        process_used: None,
    };
    let memory_request = PcuMemoryAllocationRequest {
        pool: pool_id,
        size_bytes: input_bytes.len() as u64,
        alignment_bytes: 4,
        access: PcuMemoryAccess::ReadWrite,
        host_access: PcuMemoryHostAccess::TransferOnly,
        require_device_local: false,
    };
    let mut gpu_input = allocate_with_policy(
        &mut memory_provider,
        &mut memory_ledger,
        memory_request,
        memory_policy,
    )
    .map_err(|error| format!("PCU input memory admission: {error:?}"))?;
    let gpu_output = allocate_with_policy(
        &mut memory_provider,
        &mut memory_ledger,
        memory_request,
        memory_policy,
    )
    .map_err(|error| format!("PCU output memory admission: {error:?}"))?;
    memory_provider
        .transfer_to(gpu_input.resource_mut(), 0, &input_bytes)
        .map_err(|error| format!("PCU input transfer: {error:?}"))?;

    for _ in 0..2 {
        let owned_bindings = vec![
            session.bind(
                PcuBindingRef::new(0, 0),
                PcuBindingAccess::ReadOnly,
                PcuBindingType::Value(PcuValueType::f32()),
                gpu_input.resource(),
            )?,
            session.bind(
                PcuBindingRef::new(0, 1),
                PcuBindingAccess::ReadWrite,
                PcuBindingType::Value(PcuValueType::f32()),
                gpu_output.resource(),
            )?,
        ];
        let mut completion = prepared
            .submit_owned(owned_bindings)
            .map_err(|error| format!("PCU prepared dispatch submission: {error:?}"))?;
        if completion.wait()? != PcuCompletionOutcome::Succeeded {
            return Err("PCU dispatch did not succeed".into());
        }
    }

    let mut output_bytes = vec![0_u8; input_bytes.len()];
    memory_provider
        .transfer_from(gpu_output.resource(), 0, &mut output_bytes)
        .map_err(|error| format!("PCU output transfer: {error:?}"))?;
    for (index, chunk) in output_bytes.chunks_exact(4).enumerate() {
        let actual = f32::from_ne_bytes(chunk.try_into()?);
        let expected = input[index] + 1.5;
        if actual.to_bits() != expected.to_bits() {
            return Err(format!("output[{index}] = {actual}, expected {expected}").into());
        }
    }
    println!(
        "PCU prepared grid-stride dispatch on explicitly selected ROCm backend passed twice: {INVOCATIONS} invocations covered {COUNT} values"
    );
    gpu_input
        .release(&mut memory_ledger)
        .map_err(|error| format!("PCU input accounting release: {error:?}"))?;
    gpu_output
        .release(&mut memory_ledger)
        .map_err(|error| format!("PCU output accounting release: {error:?}"))?;
    Ok(())
}

#[derive(Clone)]
struct Candidate {
    reference: PcuObjectRef,
    total_memory: u64,
    pool: PcuMemoryPoolId,
}

fn rank_devices(
    devices: &[Candidate],
    preferred: Option<i32>,
) -> Result<Vec<&Candidate>, Box<dyn std::error::Error>> {
    if let Some(index) = preferred {
        devices
            .iter()
            .find(|device| i32::try_from(device.reference.id).ok() == Some(index))
            .map(|device| vec![device])
            .ok_or_else(|| format!("requested ROCm device index {index} is unavailable").into())
    } else {
        if devices.is_empty() {
            return Err("no ROCm devices are available".into());
        }
        let mut ranked: Vec<_> = devices.iter().collect();
        ranked.sort_by(|a, b| {
            b.total_memory
                .cmp(&a.total_memory)
                .then_with(|| a.reference.id.cmp(&b.reference.id))
        });
        Ok(ranked)
    }
}

fn discovered_devices(rocm: &RocmDiscovery) -> Result<Vec<Candidate>, Box<dyn std::error::Error>> {
    let mut registry = PcuRuntimeDiscoveryRegistry::<1>::new();
    registry.register(rocm).map_err(registry_error)?;
    let mut providers = [empty_provider()];
    registry.providers(&mut providers).map_err(registry_error)?;
    let provider = providers[0];
    let mut targets = [empty_target()];
    registry
        .targets(provider.id, provider.generation, &mut targets)
        .map_err(registry_error)?;
    let target = targets[0];
    let count = registry
        .devices(target.reference, &mut [])
        .map_err(registry_error)?;
    let mut devices = vec![empty_device(); count];
    registry
        .devices(target.reference, &mut devices)
        .map_err(registry_error)?;
    let mut candidates = Vec::with_capacity(count);
    for device in devices {
        let mut contexts = [empty_context()];
        registry
            .contexts(device.reference, &mut contexts)
            .map_err(registry_error)?;
        let mut domains = [empty_domain()];
        registry
            .memory_domains(contexts[0].reference, &mut domains)
            .map_err(registry_error)?;
        candidates.push(Candidate {
            reference: device.reference,
            total_memory: domains[0].capacity_bytes.unwrap_or(0),
            pool: PcuMemoryPoolId(domains[0].reference.id),
        });
    }
    Ok(candidates)
}

fn registry_error(error: fusion_pcu::PcuRegistryError) -> Box<dyn std::error::Error> {
    format!("PCU discovery {:?}: {}", error.operation, error.message()).into()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn device(index: u32, total_memory: u64) -> Candidate {
        Candidate {
            reference: PcuObjectRef {
                provider: PcuProviderId(1),
                generation: 1,
                kind: PcuObjectKind::Device,
                id: index,
            },
            total_memory,
            pool: PcuMemoryPoolId(index),
        }
    }

    #[test]
    fn consumer_policy_prefers_capacity_then_stable_index() {
        let devices = [device(3, 24), device(1, 24), device(0, 12)];
        assert_eq!(
            rank_devices(&devices, None)
                .unwrap()
                .iter()
                .map(|device| device.reference.id)
                .collect::<Vec<_>>(),
            [1, 3, 0]
        );
        assert_eq!(rank_devices(&devices, Some(0)).unwrap()[0].reference.id, 0);
        assert!(rank_devices(&devices, Some(9)).is_err());
        assert!(rank_devices(&[], None).is_err());
    }
}
