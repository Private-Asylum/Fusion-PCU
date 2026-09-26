//! Compare typed u32 copy through prepared PCU with a native HIP copy kernel.

#[path = "support/dispatch.rs"]
#[allow(dead_code)] // Shared dispatch support also contains the f32 benchmark's helpers.
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
    PcuBindingAccess,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuValueType,
};
use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::{
    HipKernelArgument,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    compile_hip_source,
};
use dispatch_support::{
    BLOCK_SIZE,
    READBACK_SAMPLES,
    encode_u32,
    print_duration_samples,
    print_samples,
    run_direct,
    run_prepared_typed,
    timed_copy,
    verify_u32_copy,
};

#[pcu(invocations = N)]
fn copy_u32<const N: usize>(input: &[u32], output: &mut [u32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

#[pcu(invocations = 250)]
fn grid_stride_copy_u32<const N: usize>(input: &[u32], output: &mut [u32]) {
    let mut id = pcu::context::global_invocation_id();
    let stride = pcu::context::invocation_count();
    while id < N {
        output[id] = input[id];
        id += stride;
    }
}

fn dispatch_benchmarks(criterion: &mut Criterion) {
    if let Err(error) = run_benchmarks(criterion) {
        panic!("u32 dispatch benchmark setup failed: {error}");
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
    verify_grid_stride_copy(&prepared_backend)?;
    println!(
        "Device: {}; architecture={architecture}; block={BLOCK_SIZE}; Criterion paired per-launch wait",
        discovery.device_info(device)?.name
    );
    run_case(
        criterion,
        "u32-copy-orchestration-heavy",
        65,
        65,
        &architecture,
        &prepared_backend,
        &direct_runtime,
    )?;
    run_case(
        criterion,
        "u32-copy-workload-heavy",
        1 << 20,
        1 << 20,
        &architecture,
        &prepared_backend,
        &direct_runtime,
    )?;
    run_case(
        criterion,
        "u32-copy-grid-stride",
        2048,
        250,
        &architecture,
        &prepared_backend,
        &direct_runtime,
    )?;
    Ok(())
}

fn verify_grid_stride_copy(backend: &RocmOwnedDispatchBackend) -> Result<(), Box<dyn Error>> {
    let values = (0..2048_u32)
        .map(|value| value ^ 0xA5A5_5A5A)
        .collect::<Vec<_>>();
    let bytes = encode_u32(&values);
    let mut input = backend.allocate(bytes.len())?;
    let output = backend.allocate(bytes.len())?;
    input.copy_from(&bytes)?;
    let bindings = grid_stride_copy_u32_bindings();
    let builder = grid_stride_copy_u32::<2048>(&bindings)?;
    let kernel = builder.ir();
    let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(250).expect("nonzero")),
    })?;
    run_prepared_typed(
        backend,
        &prepared,
        &input,
        &output,
        PcuValueType::u32(),
        PcuBindingAccess::ReadWrite,
    )?;
    let mut actual = vec![0_u8; bytes.len()];
    output.copy_to(&mut actual)?;
    verify_u32_copy("grid-stride", &actual, &values)?;
    println!("u32 grid-stride copy verified: 250 invocations covered 2048 elements");
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_case(
    criterion: &mut Criterion,
    name: &str,
    elements: usize,
    logical_invocations: usize,
    architecture: &str,
    prepared_backend: &RocmOwnedDispatchBackend,
    direct_runtime: &fusion_pcu_rocm::HipRuntime,
) -> Result<(), Box<dyn Error>> {
    let invocations = u32::try_from(logical_invocations)?;
    let grid = invocations.div_ceil(BLOCK_SIZE);
    let input = (0..elements)
        .map(|index| u32::try_from(index).expect("benchmark shape fits u32") ^ 0xA5A5_5A5A)
        .collect::<Vec<_>>();
    let input_bytes = encode_u32(&input);
    let mut prepared_input = prepared_backend.allocate(input_bytes.len())?;
    let prepared_output = prepared_backend.allocate(input_bytes.len())?;
    prepared_input.copy_from(&input_bytes)?;
    let mut direct_input = direct_runtime.allocate(input_bytes.len())?;
    let direct_output = direct_runtime.allocate(input_bytes.len())?;
    direct_input.copy_from(&input_bytes)?;

    let bindings = copy_u32_bindings();
    let direct_builder = if elements == 65 {
        copy_u32::<65>(&bindings)?
    } else {
        copy_u32::<{ 1 << 20 }>(&bindings)?
    };
    let grid_stride_builder = grid_stride_copy_u32::<2048>(&bindings)?;
    let kernel = if logical_invocations == 250 {
        grid_stride_builder.ir()
    } else {
        direct_builder.ir()
    };
    let submission = PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(invocations).expect("nonzero elements"),
        ),
    };
    let prepared = support::cold_once("PCU cold prepare", || {
        prepared_backend.prepare_dispatch(submission)
    })?;
    let source = if logical_invocations == 250 {
        format!(
            "#include <hip/hip_runtime.h>\n\nextern \"C\" __global__ void native_u32_copy(const unsigned int* input, unsigned int* output) {{\n    const unsigned int base = blockIdx.x * blockDim.x + threadIdx.x;\n    if (base >= {logical_invocations}u) return;\n    for (unsigned int id = base; id < {elements}u; id += {logical_invocations}u) output[id] = input[id];\n}}\n"
        )
    } else {
        format!(
            "#include <hip/hip_runtime.h>\n\nextern \"C\" __global__ void native_u32_copy(const unsigned int* input, unsigned int* output) {{\n    const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;\n    if (id < {elements}u) output[id] = input[id];\n}}\n"
        )
    };
    let direct_image = compile_hip_source(&source, architecture)?;
    let direct_module = direct_runtime.load_module(&direct_image)?;
    let direct_function = direct_module.function(c"native_u32_copy")?;
    let direct_stream = direct_runtime.create_stream()?;
    let direct_arguments = [
        HipKernelArgument::Buffer(&direct_input),
        HipKernelArgument::Buffer(&direct_output),
    ];

    println!(
        "\nCase {name}: {elements} u32 elements, logical invocations={logical_invocations}, block={BLOCK_SIZE}, grid={grid}, {} MiB resident input + output per path",
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
                                run_prepared_typed(
                                    prepared_backend,
                                    &prepared,
                                    &prepared_input,
                                    &prepared_output,
                                    PcuValueType::u32(),
                                    PcuBindingAccess::ReadWrite,
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
                            let pcu = run_prepared_typed(
                                prepared_backend,
                                &prepared,
                                &prepared_input,
                                &prepared_output,
                                PcuValueType::u32(),
                                PcuBindingAccess::ReadWrite,
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
    verify_u32_copy("prepared", &prepared_bytes, &input)?;
    verify_u32_copy("direct", &direct_bytes, &input)?;
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
