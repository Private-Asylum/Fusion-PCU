//! Compare repeated prepared PCU submissions with repeated direct HIP launches.

use std::{
    error::Error,
    num::NonZeroU32,
    time::{
        Duration,
        Instant,
    },
};

use fusion_pcu::{
    F32MapBuilder,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuCompletionOutcome,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuObjectKind,
    PcuObjectRef,
    PcuOwnedCompletion,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscoveryRegistry,
    PcuTargetDescriptor,
    PcuValueType,
};
use fusion_pcu_rocm::{
    HipCompletion,
    HipKernel,
    HipKernelArgument,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    compile_hip_source,
    lower_dispatch_to_hip_source,
};

const ELEMENTS: usize = 65;
const BLOCK_SIZE: u32 = 64;
const WARMUPS: usize = 32;
const SAMPLES: usize = 512;

#[allow(clippy::too_many_lines)] // Keep benchmark setup, paired sampling, and verification together.
fn main() -> Result<(), Box<dyn Error>> {
    let architecture = std::env::var("FUSION_ROCM_ARCH").unwrap_or_else(|_| "gfx1030".into());
    let discovery = RocmDiscovery::new();
    let device = select_device(&discovery)?;
    let direct_runtime = discovery.open_device(device)?;
    let prepared_backend =
        RocmOwnedDispatchBackend::open(&discovery, device, &architecture, BLOCK_SIZE)?;

    let input: Vec<f32> = (0..ELEMENTS)
        .map(|value| f32::from(u16::try_from(value).expect("benchmark index fits u16")))
        .collect();
    let input_bytes = encode_f32(&input);
    let buffer_bytes = input_bytes.len();
    let mut prepared_input = prepared_backend.allocate(buffer_bytes)?;
    let prepared_output = prepared_backend.allocate(buffer_bytes)?;
    prepared_input.copy_from(&input_bytes)?;
    let mut direct_input = direct_runtime.allocate(buffer_bytes)?;
    let direct_output = direct_runtime.allocate(buffer_bytes)?;
    direct_input.copy_from(&input_bytes)?;

    let declarations = [
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
    let (builder, value) =
        F32MapBuilder::<8>::new(1, "dispatch_benchmark", [65, 1, 1], &declarations)
            .load_f32(PcuBindingRef::new(0, 0))?;
    let (builder, one) = builder.constant(1.0)?;
    let (builder, result) = builder.add(value, one)?;
    let builder = builder.store_f32(PcuBindingRef::new(0, 1), result)?;
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(65).expect("65 is nonzero")),
    };

    let prepared_bindings = vec![
        prepared_backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::f32()),
            prepared_input.clone(),
        )?,
        prepared_backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::f32()),
            prepared_output.clone(),
        )?,
    ];

    let prepare_started = Instant::now();
    let prepared = prepared_backend.prepare_dispatch(submission)?;
    let cold_prepare = prepare_started.elapsed();

    let source = lower_dispatch_to_hip_source(&kernel)?;
    let direct_image = compile_hip_source(&source, &architecture)?;
    let direct_module = direct_runtime.load_module(&direct_image)?;
    let direct_function = direct_module.function(c"fusion_kernel")?;
    let direct_stream = direct_runtime.create_stream()?;
    let direct_arguments = [
        HipKernelArgument::Buffer(&direct_input),
        HipKernelArgument::Buffer(&direct_output),
    ];

    println!(
        "Device: {}; same 65-element f32 kernel, block=64/grid=2, and per-launch wait; separate resident buffer pairs",
        discovery.device_info(device)?.name,
    );
    println!(
        "Cold prepared PCU prepare (lower + compile + load + resolve + stream): {:.3} ms",
        cold_prepare.as_secs_f64() * 1_000.0,
    );
    println!("Direct HIP compile/module/function/stream setup completed outside warm timing");

    for iteration in 0..WARMUPS {
        run_pair(
            iteration,
            &prepared,
            &prepared_bindings,
            &direct_function,
            &direct_stream,
            &direct_arguments,
        )?;
    }

    let mut prepared_samples = Vec::with_capacity(SAMPLES);
    let mut direct_samples = Vec::with_capacity(SAMPLES);
    for iteration in 0..SAMPLES {
        let (prepared_sample, direct_sample) = if iteration.is_multiple_of(2) {
            (
                run_prepared(&prepared, &prepared_bindings)?,
                run_direct(&direct_function, &direct_stream, &direct_arguments)?,
            )
        } else {
            let direct_sample = run_direct(&direct_function, &direct_stream, &direct_arguments)?;
            let prepared_sample = run_prepared(&prepared, &prepared_bindings)?;
            (prepared_sample, direct_sample)
        };
        prepared_samples.push(prepared_sample);
        direct_samples.push(direct_sample);
    }

    print_samples("Prepared PCU (bindings prebuilt)", &prepared_samples);
    print_samples("Direct HIP", &direct_samples);
    println!(
        "Host call return includes validation and per-launch argument/event work; it is not GPU execution duration. Completion wait is host-side synchronization latency."
    );

    let mut prepared_bytes = vec![0; buffer_bytes];
    let mut direct_bytes = vec![0; buffer_bytes];
    prepared_output.copy_to(&mut prepared_bytes)?;
    direct_output.copy_to(&mut direct_bytes)?;
    verify_output("prepared", &prepared_bytes, &input)?;
    verify_output("direct", &direct_bytes, &input)?;
    println!("Both paths produced the expected 65 values.");
    Ok(())
}

fn run_pair(
    iteration: usize,
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch<'_>,
    prepared_bindings: &[fusion_pcu::PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    direct_function: &HipKernel,
    direct_stream: &fusion_pcu_rocm::HipStreamHandle,
    direct_arguments: &[HipKernelArgument<'_>],
) -> Result<(), Box<dyn Error>> {
    if iteration.is_multiple_of(2) {
        run_prepared(prepared, prepared_bindings)?;
        run_direct(direct_function, direct_stream, direct_arguments)?;
    } else {
        run_direct(direct_function, direct_stream, direct_arguments)?;
        run_prepared(prepared, prepared_bindings)?;
    }
    Ok(())
}

fn run_prepared(
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch<'_>,
    bindings: &[fusion_pcu::PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
) -> Result<Sample, Box<dyn Error>> {
    let total_started = Instant::now();
    let launch_started = Instant::now();
    let mut completion = prepared.submit(bindings)?;
    let launch_return = launch_started.elapsed();
    let wait_started = Instant::now();
    if completion.wait()? != PcuCompletionOutcome::Succeeded {
        return Err("prepared dispatch did not succeed".into());
    }
    Ok(Sample {
        launch_return,
        wait: wait_started.elapsed(),
        total: total_started.elapsed(),
    })
}

fn run_direct(
    function: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    arguments: &[HipKernelArgument<'_>],
) -> Result<Sample, Box<dyn Error>> {
    let total_started = Instant::now();
    let launch_started = Instant::now();
    let mut completion = direct_launch(function, stream, arguments)?;
    let launch_return = launch_started.elapsed();
    let wait_started = Instant::now();
    completion.wait()?;
    Ok(Sample {
        launch_return,
        wait: wait_started.elapsed(),
        total: total_started.elapsed(),
    })
}

#[allow(unsafe_code)]
fn direct_launch(
    function: &HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    arguments: &[HipKernelArgument<'_>],
) -> Result<HipCompletion, fusion_pcu_rocm::HipError> {
    // SAFETY: `source` comes from lowering the exact f32 map IR used here, the argument order is
    // its declared input/output order, both buffers have 65 * sizeof(f32) bytes, and the caller
    // synchronizes the returned completion before reusing either allocation.
    unsafe { function.launch(stream, [2, 1, 1], [BLOCK_SIZE, 1, 1], 0, arguments) }
}

#[derive(Clone, Copy)]
struct Sample {
    launch_return: Duration,
    wait: Duration,
    total: Duration,
}

fn print_samples(label: &str, samples: &[Sample]) {
    print_metric(label, "host call return", samples, |sample| {
        sample.launch_return
    });
    print_metric(label, "completion wait", samples, |sample| sample.wait);
    print_metric(label, "end-to-end total", samples, |sample| sample.total);
}

fn print_metric(
    label: &str,
    metric: &str,
    samples: &[Sample],
    value: impl Fn(&Sample) -> Duration,
) {
    let mut values = samples.iter().map(value).collect::<Vec<_>>();
    values.sort_unstable();
    let p50 = percentile(&values, 50);
    let p95 = percentile(&values, 95);
    println!(
        "{label} {metric}: p50={:.3} us p95={:.3} us ({} runs)",
        p50.as_secs_f64() * 1_000_000.0,
        p95.as_secs_f64() * 1_000_000.0,
        values.len(),
    );
}

fn percentile(samples: &[Duration], percentile: usize) -> Duration {
    let index = samples
        .len()
        .saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1);
    samples[index]
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn verify_output(label: &str, bytes: &[u8], input: &[f32]) -> Result<(), Box<dyn Error>> {
    for (index, chunk) in bytes.chunks_exact(size_of::<f32>()).enumerate() {
        let actual = f32::from_ne_bytes(chunk.try_into()?);
        let expected = input[index] + 1.0;
        if (actual - expected).abs() > f32::EPSILON {
            return Err(format!("{label} output[{index}]={actual}; expected {expected}").into());
        }
    }
    Ok(())
}

fn select_device(discovery: &RocmDiscovery) -> Result<PcuObjectRef, Box<dyn Error>> {
    let mut registry = PcuRuntimeDiscoveryRegistry::<1>::new();
    registry.register(discovery).map_err(registry_error)?;
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    registry.providers(&mut providers).map_err(registry_error)?;
    let mut targets = [PcuTargetDescriptor {
        reference: empty_ref(PcuObjectKind::Target),
        name: "",
        readiness: PcuProviderReadiness {
            status: PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    registry
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .map_err(registry_error)?;
    let count = registry
        .devices(targets[0].reference, &mut [])
        .map_err(registry_error)?;
    if count == 0 {
        return Err("no ROCm devices are available".into());
    }
    let mut devices = vec![
        PcuDeviceDescriptor {
            reference: empty_ref(PcuObjectKind::Device),
            target: targets[0].reference,
            name: "",
            class: PcuDeviceClass::Gpu,
            vendor: None,
            architecture: None,
            generation: None,
            location: None,
        };
        count
    ];
    registry
        .devices(targets[0].reference, &mut devices)
        .map_err(registry_error)?;
    let preferred = std::env::var("FUSION_ROCM_DEVICE")
        .ok()
        .map(|value| value.parse::<u32>())
        .transpose()?;
    devices
        .iter()
        .find(|device| preferred.is_none_or(|index| device.reference.id == index))
        .map(|device| device.reference)
        .ok_or_else(|| "requested ROCm device is unavailable".into())
}

const fn empty_ref(kind: PcuObjectKind) -> PcuObjectRef {
    PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind,
        id: 0,
    }
}

fn registry_error(error: fusion_pcu::PcuRegistryError) -> Box<dyn Error> {
    format!("PCU discovery {:?}: {}", error.operation, error.message()).into()
}
