//! Hardware proof for PCU Dispatch -> HIP source -> HSACO -> `ROCm` execution.

use std::process::ExitCode;
use std::num::NonZeroU32;

use fusion_pcu::{
    F32MapBuilder,
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
    PcuMemoryUsage,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscoveryRegistry,
    PcuTargetDescriptor,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDispatchSubmission,
    PcuInvocationParameters,
    PcuInvocationShape,
    PcuOwnedCompletion,
    PcuOwnedDispatchBackend,
    PcuValueType,
    allocate_with_policy,
};
use fusion_pcu_rocm::{
    HipRuntime,
    Rocblas,
    RocmDiscovery,
    RocmDispatchBinding,
    RocmOwnedDispatchBackend,
    HipError,
    execute_pcu_dispatch,
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ERROR fusion-rocm-pcu: {error}");
            ExitCode::FAILURE
        }
    }
}

#[allow(clippy::too_many_lines)] // This hardware smoke test keeps resource lifetimes visible end to end.
fn run() -> Result<(), Box<dyn std::error::Error>> {
    const COUNT: usize = 65;
    let architecture = std::env::var("FUSION_ROCM_ARCH").unwrap_or_else(|_| "gfx1030".into());
    let rocm = RocmDiscovery::new();
    let available = discovered_devices(&rocm)?;
    let preferred = std::env::var("FUSION_ROCM_DEVICE")
        .ok()
        .map(|index| index.parse::<i32>())
        .transpose()?;
    let candidates = rank_devices(&available, preferred)?;
    let mut open_failures = Vec::new();
    let mut opened = None;
    for candidate in candidates {
        let attempt = rocm
            .open_device(candidate.reference)
            .map_err(|error| error.to_string());
        match attempt {
            Ok(runtime) => {
                opened = Some((runtime, candidate.clone()));
                break;
            }
            Err(error) => open_failures.push(format!("device {}: {error}", candidate.reference.id)),
        }
    }
    let (runtime, selected) = opened.ok_or_else(|| {
        format!(
            "no ROCm device could be opened: {}",
            open_failures.join("; ")
        )
    })?;
    let device = runtime.device_info()?;
    println!(
        "ROCm device: {} (index {}, {} bytes physical memory)",
        device.name, selected.reference.id, selected.total_memory
    );
    let pool = runtime.memory_pool_snapshot(PcuMemoryPoolId(selected.reference.id))?;
    if let (Some(capacity), PcuMemoryUsage::Known(used)) =
        (pool.capacity_bytes, pool.system_used_bytes)
    {
        println!("ROCm memory pool: {used}/{capacity} bytes system-used");
    }

    let input: Vec<f32> = (0..COUNT)
        .map(|index| f32::from(u16::try_from(index).expect("smoke-test index fits in u16")))
        .collect();
    let mut input_bytes = Vec::with_capacity(COUNT * 4);
    for value in &input {
        input_bytes.extend_from_slice(&value.to_ne_bytes());
    }
    let pool_id = PcuMemoryPoolId(selected.reference.id);
    let mut memory_provider = runtime.memory_provider(pool_id);
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

    let bindings = [
        PcuBinding::value(
            Some("input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::f32(),
        ),
        PcuBinding::value(
            Some("output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::WriteOnly,
            PcuValueType::f32(),
        ),
    ];
    let (builder, input_value) = F32MapBuilder::<8>::new(1, "add_one", [65, 1, 1], &bindings)
        .load_f32(PcuBindingRef::new(0, 0))?;
    let (builder, one) = builder.constant(1.0)?;
    let (builder, result) = builder.add(input_value, one)?;
    let builder = builder.store_f32(PcuBindingRef::new(0, 1), result)?;
    let kernel = builder.ir();
    execute_pcu_dispatch(
        &runtime,
        &kernel,
        &architecture,
        64,
        &[
            RocmDispatchBinding {
                id: PcuBindingRef::new(0, 0),
                buffer: gpu_input.resource().device_buffer(),
            },
            RocmDispatchBinding {
                id: PcuBindingRef::new(0, 1),
                buffer: gpu_output.resource().device_buffer(),
            },
        ],
    )?;

    let mut output_bytes = vec![0_u8; input_bytes.len()];
    memory_provider
        .transfer_from(gpu_output.resource(), 0, &mut output_bytes)
        .map_err(|error| format!("PCU output transfer: {error:?}"))?;
    for (index, chunk) in output_bytes.chunks_exact(4).enumerate() {
        let actual = f32::from_ne_bytes(chunk.try_into()?);
        let expected = input[index] + 1.0;
        if (actual - expected).abs() > f32::EPSILON {
            return Err(format!("output[{index}] = {actual}, expected {expected}").into());
        }
    }
    println!("PCU -> ROCm dispatch passed for {COUNT} values");
    gpu_input
        .release(&mut memory_ledger)
        .map_err(|error| format!("PCU input accounting release: {error:?}"))?;
    gpu_output
        .release(&mut memory_ledger)
        .map_err(|error| format!("PCU output accounting release: {error:?}"))?;
    let async_backend =
        RocmOwnedDispatchBackend::open(&rocm, selected.reference, &architecture, 64)?;
    let mut async_memory_provider = async_backend.memory_provider(pool_id);
    let mut async_memory_ledger = PcuMemoryReservationLedger::<2>::new();
    let mut async_input = allocate_with_policy(
        &mut async_memory_provider,
        &mut async_memory_ledger,
        memory_request,
        memory_policy,
    )
    .map_err(|error| format!("PCU async input memory admission: {error:?}"))?;
    let async_output = allocate_with_policy(
        &mut async_memory_provider,
        &mut async_memory_ledger,
        memory_request,
        memory_policy,
    )
    .map_err(|error| format!("PCU async output memory admission: {error:?}"))?;
    async_memory_provider
        .transfer_to(async_input.resource_mut(), 0, &input_bytes)
        .map_err(|error| format!("PCU async input transfer: {error:?}"))?;
    let async_readback = async_output.resource().device_buffer().clone();
    let async_bindings = vec![
        async_backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::f32()),
            async_output.resource().device_buffer().clone(),
        )?,
        async_backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::f32()),
            async_input.resource().device_buffer().clone(),
        )?,
    ];
    let mut completion = async_backend
        .submit_dispatch_owned(
            PcuDispatchSubmission {
                kernel: &kernel,
                shape: PcuInvocationShape::threads(NonZeroU32::new(65).unwrap()),
            },
            async_bindings,
            PcuInvocationParameters::empty(),
        )
        .map_err(|error| format!("PCU owned async submission: {error:?}"))?;
    let mut async_output_bytes = vec![0_u8; input_bytes.len()];
    if !matches!(
        async_readback.copy_to(&mut async_output_bytes),
        Err(HipError::Busy)
    ) {
        return Err("owned async output was accessible before completion wait".into());
    }
    if completion.wait()? != PcuCompletionOutcome::Succeeded {
        return Err("owned async dispatch did not succeed".into());
    }
    async_memory_provider
        .transfer_from(async_output.resource(), 0, &mut async_output_bytes)
        .map_err(|error| format!("PCU async output transfer: {error:?}"))?;
    if async_output_bytes != output_bytes {
        return Err("owned async dispatch output differs from synchronous reference".into());
    }
    drop(async_readback);
    drop(completion);
    async_input
        .release(&mut async_memory_ledger)
        .map_err(|error| format!("PCU async input accounting release: {error:?}"))?;
    async_output
        .release(&mut async_memory_ledger)
        .map_err(|error| format!("PCU async output accounting release: {error:?}"))?;
    if async_memory_ledger.reserved_bytes(pool_id) != Some(0) {
        return Err("owned async admission did not release its memory reservation".into());
    }
    let replacement = allocate_with_policy(
        &mut async_memory_provider,
        &mut async_memory_ledger,
        memory_request,
        memory_policy,
    )
    .map_err(|error| format!("PCU async memory reuse admission: {error:?}"))?;
    replacement
        .release(&mut async_memory_ledger)
        .map_err(|error| format!("PCU async memory reuse release: {error:?}"))?;
    println!("PCU -> ROCm owned async dispatch passed for {COUNT} values");
    verify_rocblas(&runtime)?;
    Ok(())
}

#[derive(Clone)]
struct Candidate {
    reference: PcuObjectRef,
    total_memory: u64,
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

fn verify_rocblas(runtime: &HipRuntime) -> Result<(), Box<dyn std::error::Error>> {
    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect()
    }

    // Column-major A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]].
    let a_data = bytes(&[1.0, 3.0, 2.0, 4.0]);
    let b_data = bytes(&[5.0, 7.0, 6.0, 8.0]);
    let mut a = runtime.allocate(a_data.len())?;
    let mut b = runtime.allocate(b_data.len())?;
    let c = runtime.allocate(4 * size_of::<f32>())?;
    a.copy_from(&a_data)?;
    b.copy_from(&b_data)?;
    Rocblas::new(runtime)?.sgemm(false, false, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &c, 2)?;
    let mut result = vec![0_u8; 4 * size_of::<f32>()];
    c.copy_to(&mut result)?;
    let actual: Vec<f32> = result
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect();
    let expected = [19.0, 43.0, 22.0, 50.0];
    if actual != expected {
        return Err(
            format!("Rust rocBLAS SGEMM returned {actual:?}, expected {expected:?}").into(),
        );
    }
    println!("Rust rocBLAS SGEMM and output verification passed");
    Ok(())
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
