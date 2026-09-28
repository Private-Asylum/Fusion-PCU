//! Matched fresh output allocation, synchronous launch and release; transfers outside timing.

#[path = "alloc.rs"]
#[allow(dead_code)] // Other benchmark consumers use the phased counter helper too.
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    rc::Rc,
    time::{
        Duration,
        Instant,
    },
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    PcuDeviceTensor,
    PcuMemoryPoolId,
};
#[rustfmt::skip]
use fusion_pcu::dialect::tensor::{
    Graph,
    TensorArithmeticCapability,
    TensorArithmeticRewritePolicy,
    TensorPointwiseGroupingPolicy,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    HipKernelArgument,
    HipRuntime,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
    compile_hip_source,
};

#[fusion_pcu::pcu]
fn source_identity(
    input: &[f32],
) -> Result<fusion_pcu::PcuTensor<f32>, fusion_pcu::PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu]
fn source_relu(input: &[f32]) -> Result<fusion_pcu::PcuTensor<f32>, fusion_pcu::PcuExecutionError> {
    pcu::relu(input)
}

#[fusion_pcu::pcu(invocations: N)]
fn source_refresh<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

fn refresh_source_input(
    elements: usize,
    input: &[f32],
    output: &mut fusion_pcu::PcuTensor<f32>,
) -> Result<(), fusion_pcu::PcuExecutionError> {
    match elements {
        65 => source_refresh::<65>(input, output),
        1_048_576 => source_refresh::<1_048_576>(input, output),
        _ => unreachable!("benchmark declares both shapes"),
    }
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) = crate::support::selection::open_ranked(&discovery, candidates, 256)?;
    let backend = Rc::new(backend);
    let runtime = discovery.open_device(selected.device)?;
    let architecture = selected
        .architecture
        .as_deref()
        .ok_or("missing architecture")?;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    println!("Owned-program benchmark device: {}", selected.name);
    for elements in [65, 1_048_576] {
        run_case(
            criterion,
            &backend,
            &runtime,
            selected.pool,
            architecture,
            elements,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)] // Keeps the matched setup and timing boundaries reviewable together.
#[allow(clippy::significant_drop_tightening)] // Criterion finish consumes the group after both routes.
#[allow(clippy::branches_sharing_code)] // Route selection stays outside each measured boundary.
#[allow(unsafe_code)] // Native HIP launch ABI is verified against the typed PCU operation.
fn run_case(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &HipRuntime,
    pool: PcuMemoryPoolId,
    architecture: &str,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = root.assessor();
    let mut graph = Graph::default();
    let input_id = graph.input([elements])?;
    let output_id = graph.relu(input_id)?;
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = crate::support::cold_once("owned PCU program preparation", || {
        assessor.prepare_owned_program(program)
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_relu(const float* input, float* output) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) output[id] = input[id] > 0.0f ? input[id] : 0.0f; }}\n"
    );
    let image = crate::support::cold_once("native owned-output kernel compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(c"native_relu")?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(256);
    let mut host = vec![0.0_f32; elements];
    let mut observed = vec![0.0_f32; elements];
    fill(&mut host, 0);
    let owner = PcuDeviceTensor::new([elements], backend.upload_buffer(pool, &host)?)?;
    let mut buffer = Some(owner.into_buffer());
    let mut native_input = runtime.allocate(elements * size_of::<f32>())?;
    let mut memory = backend.memory_provider(pool);
    // Warm compilation and validate the first genuinely owned escape before timing.
    let owner = PcuDeviceTensor::new([elements], buffer.take().expect("resident input owner"))?;
    let outputs = crate::support::cold_once(
        "owned PCU first execution including dispatch compilation",
        || {
            assessor.execute_owned_program_outputs(
                &prepared,
                &[(input_id, &owner)],
                pool,
                &mut memory,
            )
        },
    )?;
    backend.download_buffer(pool, outputs[0].1.buffer(), &mut observed)?;
    verify(&host, &observed);
    drop(outputs);
    let capture = alloc::AllocationCapture::start();
    let outputs = root.assessor().execute_owned_program_outputs(
        &prepared,
        &[(input_id, &owner)],
        pool,
        &mut memory,
    )?;
    drop(outputs);
    let pcu_counts = alloc::AllocationCapture::finish();
    drop(capture);
    native_input.copy_from(bytemuck::cast_slice(&host))?;
    let capture = alloc::AllocationCapture::start();
    let output = runtime.allocate(elements * size_of::<f32>())?;
    let arguments = [
        HipKernelArgument::Buffer(&native_input),
        HipKernelArgument::Buffer(&output),
    ];
    // SAFETY: The native ABI and extents are identical to the timed native route.
    let mut completion =
        unsafe { kernel.launch(&stream, [grid, 1, 1], [256, 1, 1], 0, &arguments) }?;
    completion.wait()?;
    drop(completion);
    drop(output);
    let native_counts = alloc::AllocationCapture::finish();
    drop(capture);
    println!(
        "Owned fresh-output Rust heap census {elements}: PCU {} allocs/{} reallocs/{} requested B; native {}/{}/{}. Excludes input refresh, readback, and native driver allocations.",
        pcu_counts.alloc_calls,
        pcu_counts.realloc_calls,
        pcu_counts.requested_bytes,
        native_counts.alloc_calls,
        native_counts.realloc_calls,
        native_counts.requested_bytes,
    );
    let source_input = source_identity(&host)?;
    let source_output = source_relu(&source_input)?;
    source_output.read_into(&mut observed)?;
    verify(&host, &observed);
    drop(source_output);
    let capture = alloc::AllocationCapture::start();
    let source_output = source_relu(&source_input)?;
    drop(source_output);
    let source_counts = alloc::AllocationCapture::finish();
    drop(capture);
    println!(
        "Owned source Rust heap census {elements}: {} allocs/{} reallocs/{} requested B. Excludes source ownership setup, refresh, readback and driver allocations.",
        source_counts.alloc_calls, source_counts.realloc_calls, source_counts.requested_bytes
    );
    drop(source_input);
    buffer = Some(owner.into_buffer());

    // Balance all six route orders so shared clock/thermal drift cannot masquerade as wrapper cost.
    let mut pcu_times = [Duration::ZERO; 36];
    let mut native_times = [Duration::ZERO; 36];
    let mut source_times = [Duration::ZERO; 36];
    let mut source_ratios = [0.0_f64; 36];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut paired_ratios = [0.0_f64; 36];
    for sample in 0..36 {
        fill(&mut host, u64::try_from(sample + 1)?);
        backend.refresh_buffer(pool, buffer.as_mut().expect("resident input owner"), &host)?;
        native_input.copy_from(bytemuck::cast_slice(&host))?;
        let owner = PcuDeviceTensor::new([elements], buffer.take().expect("resident input owner"))?;
        let source_input = source_identity(&host)?;
        for route in orders[sample % orders.len()] {
            if route == 0 {
                let start = Instant::now();
                let outputs = root.assessor().execute_owned_program_outputs(
                    &prepared,
                    &[(input_id, &owner)],
                    pool,
                    &mut memory,
                )?;
                let execution = start.elapsed();
                backend.download_buffer(pool, outputs[0].1.buffer(), &mut observed)?;
                verify(&host, &observed);
                let release = Instant::now();
                drop(black_box(outputs));
                pcu_times[sample] = execution + release.elapsed();
            } else if route == 1 {
                let start = Instant::now();
                let output = runtime.allocate(elements * size_of::<f32>())?;
                let arguments = [
                    HipKernelArgument::Buffer(&native_input),
                    HipKernelArgument::Buffer(&output),
                ];
                // SAFETY: The paired route uses the same typed ABI and full extents as Criterion.
                let mut completion =
                    unsafe { kernel.launch(&stream, [grid, 1, 1], [256, 1, 1], 0, &arguments) }?;
                completion.wait()?;
                drop(completion);
                let execution = start.elapsed();
                output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
                verify(&host, &observed);
                let release = Instant::now();
                drop(black_box(output));
                native_times[sample] = execution + release.elapsed();
            } else {
                let start = Instant::now();
                let output = source_relu(&source_input)?;
                let execution = start.elapsed();
                output.read_into(&mut observed)?;
                verify(&host, &observed);
                let release = Instant::now();
                drop(black_box(output));
                source_times[sample] = execution + release.elapsed();
            }
        }
        paired_ratios[sample] =
            pcu_times[sample].as_secs_f64() / native_times[sample].as_secs_f64();
        source_ratios[sample] =
            source_times[sample].as_secs_f64() / native_times[sample].as_secs_f64();
        drop(source_input);
        buffer = Some(owner.into_buffer());
    }
    pcu_times.sort_unstable();
    native_times.sort_unstable();
    paired_ratios.sort_unstable_by(f64::total_cmp);
    source_times.sort_unstable();
    source_ratios.sort_unstable_by(f64::total_cmp);
    println!(
        "Owned fresh-output paired diagnostic {elements}, 36 balanced CPU-verified triples: raw PCU {:?}, source PCU {:?}, native {:?}, raw/native paired ratio {:.4}, source/native paired ratio {:.4}. Diagnostic medians, not Criterion intervals.",
        (pcu_times[17] + pcu_times[18]) / 2,
        (source_times[17] + source_times[18]) / 2,
        (native_times[17] + native_times[18]) / 2,
        paired_ratios[17].midpoint(paired_ratios[18]),
        source_ratios[17].midpoint(source_ratios[18]),
    );

    let mut group = criterion.benchmark_group("owned_program_fresh_output");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    let mut job = 0_u64;
    group.bench_function(BenchmarkId::new("pcu", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                backend
                    .refresh_buffer(pool, buffer.as_mut().expect("resident input owner"), &host)
                    .expect("refresh resident input");
                let owner =
                    PcuDeviceTensor::new([elements], buffer.take().expect("resident input owner"))
                        .expect("shape");
                let start = Instant::now();
                let outputs = root
                    .assessor()
                    .execute_owned_program_outputs(
                        &prepared,
                        &[(input_id, &owner)],
                        pool,
                        &mut memory,
                    )
                    .expect("owned execution");
                total += start.elapsed();
                backend
                    .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
                    .expect("verify readback");
                verify(&host, &observed);
                let start = Instant::now();
                drop(black_box(outputs));
                total += start.elapsed();
                buffer = Some(owner.into_buffer());
            }
            total
        });
    });
    group.bench_function(BenchmarkId::new("native", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                native_input
                    .copy_from(bytemuck::cast_slice(&host))
                    .expect("refresh native input");
                let start = Instant::now();
                let output = runtime
                    .allocate(elements * size_of::<f32>())
                    .expect("fresh native output");
                let arguments = [
                    HipKernelArgument::Buffer(&native_input),
                    HipKernelArgument::Buffer(&output),
                ];
                // SAFETY: Both pointers cover elements f32 values and match native_relu's ABI.
                let mut completion =
                    unsafe { kernel.launch(&stream, [grid, 1, 1], [256, 1, 1], 0, &arguments) }
                        .expect("native launch");
                completion.wait().expect("native terminal completion");
                drop(completion);
                total += start.elapsed();
                output
                    .copy_to(bytemuck::cast_slice_mut(&mut observed))
                    .expect("verify native readback");
                verify(&host, &observed);
                let start = Instant::now();
                drop(black_box(output));
                total += start.elapsed();
            }
            total
        });
    });
    group.bench_function(BenchmarkId::new("source", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                // Establish a current resident input outside the output execution boundary.
                // This includes real identity staging/copy; it is not cached pointer content.
                let input = source_identity(&host).expect("current owned source input");
                let start = Instant::now();
                let output = source_relu(&input).expect("same-name owned source execution");
                total += start.elapsed();
                output.read_into(&mut observed).expect("source readback");
                verify(&host, &observed);
                let start = Instant::now();
                drop(black_box(output));
                total += start.elapsed();
                drop(input);
            }
            total
        });
    });
    // Control the untimed allocation cadence: retain the input owner, refresh its contents
    // through the public same-name kernel API, and time the same fresh escaping output.
    // Refresh submission/transfer remains outside timing, as for raw PCU/native refresh.
    let mut retained_input = source_identity(&host)?;
    crate::support::cold_once("source retained-input refresh preparation", || {
        refresh_source_input(elements, &host, &mut retained_input)
    })?;
    group.bench_function(
        BenchmarkId::new("source_retained_input", elements),
        |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    fill(&mut host, job);
                    refresh_source_input(elements, &host, &mut retained_input)
                        .expect("refresh retained source input");
                    let start = Instant::now();
                    let output =
                        source_relu(&retained_input).expect("retained-input source execution");
                    total += start.elapsed();
                    output
                        .read_into(&mut observed)
                        .expect("source control readback");
                    verify(&host, &observed);
                    let start = Instant::now();
                    drop(black_box(output));
                    total += start.elapsed();
                }
                total
            });
        },
    );
    group.finish();
    Ok(())
}

fn fill(values: &mut [f32], job: u64) {
    let phase =
        f32::from_bits(0x3f00_0000 | (u32::try_from(job & 0x007f_ffff).expect("phase fits")));
    for (index, value) in values.iter_mut().enumerate() {
        let base = f32::from(u16::try_from(index % 1024).expect("sample fits"));
        *value = if index % 2 == 0 {
            base + phase
        } else {
            -base - phase
        };
    }
}

fn verify(input: &[f32], observed: &[f32]) {
    assert_eq!(input.len(), observed.len());
    assert!(
        input.iter().zip(observed).all(|(&input, &output)| {
            output.to_bits() == (if input > 0.0 { input } else { 0.0_f32 }).to_bits()
        }),
        "owned-program output differs from independent CPU oracle"
    );
}
