//! Compare repeated prepared PCU submissions with repeated direct HIP launches.

#[path = "support/dispatch.rs"]
mod dispatch_support;
mod support;

use std::{
    error::Error,
    hint::black_box,
    num::NonZeroU32,
    time::Duration,
};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};

use fusion_pcu::{
    F32MapBuilder,
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuValueType,
};
use fusion_pcu_rocm::{
    HipKernelArgument,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    compile_hip_source,
};
use dispatch_support::{
    BLOCK_SIZE,
    READBACK_SAMPLES,
    encode_f32,
    print_duration_samples,
    print_samples,
    run_direct,
    run_prepared,
    timed_copy,
    verify_output,
};

fn dispatch_benchmarks(criterion: &mut Criterion) {
    if let Err(error) = run_benchmarks(criterion) {
        panic!("dispatch benchmark setup failed: {error}");
    }
}

#[allow(clippy::too_many_lines)]
fn run_benchmarks(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err("no ROCm device has an architecture for the native hipcc comparison".into());
    }
    let (prepared_backend, selected) =
        support::selection::open_ranked(&discovery, candidates, BLOCK_SIZE)?;
    let device = selected.device;
    let architecture = selected
        .architecture
        .ok_or("selected Dispatch device has no HIP architecture")?;
    let direct_runtime = discovery.open_device(device)?;
    println!(
        "Device: {}; architecture={architecture}; block={BLOCK_SIZE}; Criterion paired per-launch wait",
        discovery.device_info(device)?.name
    );
    run_case(
        criterion,
        "orchestration-heavy",
        65,
        &architecture,
        &prepared_backend,
        &direct_runtime,
    )?;
    run_case(
        criterion,
        "workload-heavy",
        1 << 20,
        &architecture,
        &prepared_backend,
        &direct_runtime,
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_case(
    criterion: &mut Criterion,
    name: &str,
    elements: usize,
    architecture: &str,
    prepared_backend: &RocmOwnedDispatchBackend,
    direct_runtime: &fusion_pcu_rocm::HipRuntime,
) -> Result<(), Box<dyn Error>> {
    let invocations = u32::try_from(elements)?;
    let grid = invocations.div_ceil(BLOCK_SIZE);
    let input: Vec<f32> = (0..elements)
        .map(|value| f32::from(u16::try_from(value % 1024).expect("sample fits u16")))
        .collect();
    let input_bytes = encode_f32(&input);
    let mut prepared_input = prepared_backend.allocate(input_bytes.len())?;
    let prepared_output = prepared_backend.allocate(input_bytes.len())?;
    prepared_input.copy_from(&input_bytes)?;
    let mut direct_input = direct_runtime.allocate(input_bytes.len())?;
    let direct_output = direct_runtime.allocate(input_bytes.len())?;
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
        F32MapBuilder::<8>::new(1, "dispatch_benchmark", [invocations, 1, 1], &declarations)
            .load_f32(PcuBindingRef::new(0, 0))?;
    let (builder, one) = builder.constant(1.0)?;
    let (builder, result) = builder.add(value, one)?;
    let builder = builder.store_f32(PcuBindingRef::new(0, 1), result)?;
    let kernel = builder.ir();
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(invocations).expect("nonzero elements"),
        ),
    };
    let prepared = support::cold_once("PCU cold prepare", || {
        prepared_backend.prepare_dispatch(submission)
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\n\nextern \"C\" __global__ void native_add(const float* input, float* output) {{\n    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;\n    if (id < {elements}u) output[id] = input[id] + 1.0f;\n}}\n"
    );
    let direct_image = compile_hip_source(&source, architecture)?;
    let direct_module = direct_runtime.load_module(&direct_image)?;
    let direct_function = direct_module.function(c"native_add")?;
    let direct_stream = direct_runtime.create_stream()?;
    let direct_arguments = [
        HipKernelArgument::Buffer(&direct_input),
        HipKernelArgument::Buffer(&direct_output),
    ];

    println!(
        "\nCase {name}: {elements} f32 elements, block={BLOCK_SIZE}, grid={grid}, {} MiB resident input + output per path",
        input_bytes.len() * 2 / (1024 * 1024)
    );

    {
        let mut group = criterion.benchmark_group(name);
        group.throughput(Throughput::Elements(elements as u64));
        for (label, pcu_target) in [("Prepared PCU", true), ("Direct HIP", false)] {
            let mut observed = Vec::new();
            group.bench_function(BenchmarkId::new(label, elements), |bencher| {
                bencher.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for iteration in 0..iterations {
                        let (pcu, native) = if iteration.is_multiple_of(2) {
                            (
                                run_prepared(
                                    prepared_backend,
                                    &prepared,
                                    &prepared_input,
                                    &prepared_output,
                                )
                                .expect("prepared dispatch failed"),
                                run_direct(
                                    &direct_function,
                                    &direct_stream,
                                    &direct_arguments,
                                    grid,
                                )
                                .expect("direct HIP dispatch failed"),
                            )
                        } else {
                            let native = run_direct(
                                &direct_function,
                                &direct_stream,
                                &direct_arguments,
                                grid,
                            )
                            .expect("direct HIP dispatch failed");
                            let pcu = run_prepared(
                                prepared_backend,
                                &prepared,
                                &prepared_input,
                                &prepared_output,
                            )
                            .expect("prepared dispatch failed");
                            (pcu, native)
                        };
                        elapsed += if pcu_target { pcu.total } else { native.total };
                        observed.push(black_box(if pcu_target { pcu } else { native }));
                        black_box((&pcu, &native));
                    }
                    elapsed
                });
            });
            print_samples(label, &observed);
        }
        group.finish();
    }
    let mut prepared_bytes = vec![0; input_bytes.len()];
    let mut direct_bytes = vec![0; input_bytes.len()];
    // Repeated, paired, alternating reads avoid treating one fixed-order copy as a benchmark.
    prepared_output.copy_to(&mut prepared_bytes)?;
    direct_output.copy_to(&mut direct_bytes)?;
    let mut pcu_reads = Vec::with_capacity(READBACK_SAMPLES);
    let mut native_reads = Vec::with_capacity(READBACK_SAMPLES);
    for iteration in 0..READBACK_SAMPLES {
        if iteration.is_multiple_of(2) {
            pcu_reads.push(timed_copy(&prepared_output, &mut prepared_bytes)?);
            native_reads.push(timed_copy(&direct_output, &mut direct_bytes)?);
        } else {
            native_reads.push(timed_copy(&direct_output, &mut direct_bytes)?);
            pcu_reads.push(timed_copy(&prepared_output, &mut prepared_bytes)?);
        }
    }
    verify_output("prepared", &prepared_bytes, &input)?;
    verify_output("direct", &direct_bytes, &input)?;
    println!(
        "Verified both outputs; paired blocking readback outside dispatch samples ({READBACK_SAMPLES} samples):"
    );
    print_duration_samples("PCU readback", &pcu_reads);
    print_duration_samples("HIP readback", &native_reads);
    Ok(())
}

criterion_group! {
    name = benches;
    config = support::criterion_config();
    targets = dispatch_benchmarks
}
criterion_main!(benches);
