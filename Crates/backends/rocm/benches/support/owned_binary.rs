//! Matched fresh-output binary add through source, owned graph, and native HIP routes.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    mem::size_of,
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
    PcuTensor,
    PcuExecutionError,
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

const BLOCK: u32 = 256;

#[fusion_pcu::pcu]
fn add(lhs: &[f32], rhs: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::identity(lhs + rhs)?)
}

#[fusion_pcu::pcu]
fn identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu(invocations = N)]
fn refresh<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

fn refresh_input(
    elements: usize,
    input: &[f32],
    output: &mut PcuTensor<f32>,
) -> Result<(), PcuExecutionError> {
    match elements {
        65 => refresh::<65>(input, output),
        1_048_576 => refresh::<1_048_576>(input, output),
        _ => unreachable!("benchmark declares both shapes"),
    }
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect();
    let (backend, selected) =
        crate::support::selection::open_ranked(&discovery, candidates, BLOCK)?;
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
    println!("Owned binary-add benchmark device: {}", selected.name);
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

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(unsafe_code)] // Native HIP launch uses the same typed shape and addition as PCU.
fn run_case(
    criterion: &mut Criterion,
    backend: &Rc<RocmOwnedDispatchBackend>,
    runtime: &HipRuntime,
    pool: PcuMemoryPoolId,
    architecture: &str,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let assessor_root = RocmOwnedTensorAssessor::new(Rc::clone(backend))?;
    let assessor = assessor_root.assessor();
    let mut graph = Graph::default();
    let lhs_id = graph.input([elements], fusion_pcu::PcuScalarType::F32)?;
    let rhs_id = graph.input([elements], fusion_pcu::PcuScalarType::F32)?;
    let output_id = graph.add(lhs_id, rhs_id)?;
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = crate::support::cold_once("owned binary PCU program preparation", || {
        assessor.prepare_owned_program(program)
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_add(const float* lhs, const float* rhs, float* output) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) output[id] = lhs[id] + rhs[id]; }}\n"
    );
    let image = crate::support::cold_once("native binary-add HIP compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(c"native_add")?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(BLOCK);

    let mut lhs = vec![0.0_f32; elements];
    let mut rhs = vec![0.0_f32; elements];
    let mut observed = vec![0.0_f32; elements];
    fill(&mut lhs, &mut rhs, 0);
    let mut lhs_buffer = Some(backend.upload_buffer(pool, &lhs)?);
    let mut rhs_buffer = Some(backend.upload_buffer(pool, &rhs)?);
    let mut native_lhs = runtime.allocate(elements * size_of::<f32>())?;
    let mut native_rhs = runtime.allocate(elements * size_of::<f32>())?;
    native_lhs.copy_from(bytemuck::cast_slice(&lhs))?;
    native_rhs.copy_from(bytemuck::cast_slice(&rhs))?;
    let mut memory = backend.memory_provider(pool);

    let lhs_owner = PcuDeviceTensor::new([elements], lhs_buffer.take().expect("left input owner"))?;
    let rhs_owner =
        PcuDeviceTensor::new([elements], rhs_buffer.take().expect("right input owner"))?;
    let outputs = crate::support::cold_once("owned binary PCU first dispatch", || {
        assessor.execute_owned_program_outputs(
            &prepared,
            &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
            pool,
            &mut memory,
        )
    })?;
    backend.download_buffer(pool, outputs[0].1.buffer(), &mut observed)?;
    verify(&lhs, &rhs, &observed);
    drop(outputs);

    let mut source_lhs =
        crate::support::cold_once("source left identity ownership", || identity(&lhs))?;
    let mut source_rhs =
        crate::support::cold_once("source right identity ownership", || identity(&rhs))?;
    crate::support::cold_once("source left retained-input refresh", || {
        refresh_input(elements, &lhs, &mut source_lhs)
    })?;
    crate::support::cold_once("source right retained-input refresh", || {
        refresh_input(elements, &rhs, &mut source_rhs)
    })?;
    let first = crate::support::cold_once("source binary-add first execution", || {
        add(&source_lhs, &source_rhs)
    })?;
    first.read_into(&mut observed)?;
    verify(&lhs, &rhs, &observed);
    drop(first);

    // Warm each route and measure heap activity outside Criterion's measurement window.
    let _capture = alloc::AllocationCapture::start();
    let outputs = assessor_root.assessor().execute_owned_program_outputs(
        &prepared,
        &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
        pool,
        &mut memory,
    )?;
    drop(outputs);
    let raw_allocs = alloc::AllocationCapture::finish();
    let _capture = alloc::AllocationCapture::start();
    let output = runtime.allocate(elements * size_of::<f32>())?;
    let args = [
        HipKernelArgument::Buffer(&native_lhs),
        HipKernelArgument::Buffer(&native_rhs),
        HipKernelArgument::Buffer(&output),
    ];
    // SAFETY: Three buffers each cover `elements` f32 values, matching native_add's signature.
    let mut completion = unsafe { kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
    completion.wait()?;
    drop(completion);
    drop(output);
    let native_allocs = alloc::AllocationCapture::finish();
    let capture = alloc::AllocationCapture::start();
    let output = add(&source_lhs, &source_rhs)?;
    drop(output);
    let source_allocs = alloc::AllocationCapture::finish();
    drop(capture);
    println!(
        "Owned binary-add warm Rust heap census {elements}: raw PCU {}/{}/{} B, source {}/{}/{} B, native {}/{}/{} B allocations/reallocations/requested; excludes transfers, readback, and driver allocations.",
        raw_allocs.alloc_calls,
        raw_allocs.realloc_calls,
        raw_allocs.requested_bytes,
        source_allocs.alloc_calls,
        source_allocs.realloc_calls,
        source_allocs.requested_bytes,
        native_allocs.alloc_calls,
        native_allocs.realloc_calls,
        native_allocs.requested_bytes,
    );
    lhs_buffer = Some(lhs_owner.into_buffer());
    rhs_buffer = Some(rhs_owner.into_buffer());

    let mut raw_times = [Duration::ZERO; 36];
    let mut source_times = [Duration::ZERO; 36];
    let mut native_times = [Duration::ZERO; 36];
    let orders = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ];
    let mut raw_ratios = [0.0_f64; 36];
    let mut source_ratios = [0.0_f64; 36];
    for sample in 0..36 {
        fill(&mut lhs, &mut rhs, u64::try_from(sample + 1)?);
        backend.refresh_buffer(pool, lhs_buffer.as_mut().expect("left input owner"), &lhs)?;
        backend.refresh_buffer(pool, rhs_buffer.as_mut().expect("right input owner"), &rhs)?;
        native_lhs.copy_from(bytemuck::cast_slice(&lhs))?;
        native_rhs.copy_from(bytemuck::cast_slice(&rhs))?;
        refresh_input(elements, &lhs, &mut source_lhs)?;
        refresh_input(elements, &rhs, &mut source_rhs)?;
        for route in orders[sample % orders.len()] {
            if route == 0 {
                let lhs_owner =
                    PcuDeviceTensor::new([elements], lhs_buffer.take().expect("left input owner"))?;
                let rhs_owner = PcuDeviceTensor::new(
                    [elements],
                    rhs_buffer.take().expect("right input owner"),
                )?;
                let start = Instant::now();
                let outputs = assessor_root.assessor().execute_owned_program_outputs(
                    &prepared,
                    &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
                    pool,
                    &mut memory,
                )?;
                let execution = start.elapsed();
                backend.download_buffer(pool, outputs[0].1.buffer(), &mut observed)?;
                verify(&lhs, &rhs, &observed);
                let release = Instant::now();
                drop(black_box(outputs));
                raw_times[sample] = execution + release.elapsed();
                lhs_buffer = Some(lhs_owner.into_buffer());
                rhs_buffer = Some(rhs_owner.into_buffer());
            } else if route == 1 {
                let start = Instant::now();
                let output = add(&source_lhs, &source_rhs)?;
                let execution = start.elapsed();
                output.read_into(&mut observed)?;
                verify(&lhs, &rhs, &observed);
                let release = Instant::now();
                drop(black_box(output));
                source_times[sample] = execution + release.elapsed();
            } else {
                let start = Instant::now();
                let output = runtime.allocate(elements * size_of::<f32>())?;
                let args = [
                    HipKernelArgument::Buffer(&native_lhs),
                    HipKernelArgument::Buffer(&native_rhs),
                    HipKernelArgument::Buffer(&output),
                ];
                // SAFETY: Same shape and ABI as the validated native add kernel.
                let mut completion =
                    unsafe { kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }?;
                completion.wait()?;
                drop(completion);
                let execution = start.elapsed();
                output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
                verify(&lhs, &rhs, &observed);
                let release = Instant::now();
                drop(black_box(output));
                native_times[sample] = execution + release.elapsed();
            }
        }
        raw_ratios[sample] = raw_times[sample].as_secs_f64() / native_times[sample].as_secs_f64();
        source_ratios[sample] =
            source_times[sample].as_secs_f64() / native_times[sample].as_secs_f64();
    }
    raw_times.sort_unstable();
    source_times.sort_unstable();
    native_times.sort_unstable();
    raw_ratios.sort_unstable_by(f64::total_cmp);
    source_ratios.sort_unstable_by(f64::total_cmp);
    println!(
        "Owned binary-add paired diagnostic {elements}, 36 balanced CPU-verified triples: raw PCU {:?}, source {:?}, native {:?}; paired raw/native ratio {:.4}, source/native ratio {:.4}. Diagnostic medians, not Criterion intervals.",
        midpoint(raw_times[17], raw_times[18]),
        midpoint(source_times[17], source_times[18]),
        midpoint(native_times[17], native_times[18]),
        midpoint_f64(raw_ratios[17], raw_ratios[18]),
        midpoint_f64(source_ratios[17], source_ratios[18]),
    );

    let mut group = criterion.benchmark_group("owned_binary_fresh_output");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    let mut job = 36_u64;
    group.bench_function(BenchmarkId::new("raw_pcu", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut lhs, &mut rhs, job);
                backend
                    .refresh_buffer(pool, lhs_buffer.as_mut().expect("left input"), &lhs)
                    .expect("refresh left input");
                backend
                    .refresh_buffer(pool, rhs_buffer.as_mut().expect("right input"), &rhs)
                    .expect("refresh right input");
                let lhs_owner =
                    PcuDeviceTensor::new([elements], lhs_buffer.take().expect("left input"))
                        .expect("left input shape");
                let rhs_owner =
                    PcuDeviceTensor::new([elements], rhs_buffer.take().expect("right input"))
                        .expect("right input shape");
                let start = Instant::now();
                let outputs = assessor_root
                    .assessor()
                    .execute_owned_program_outputs(
                        &prepared,
                        &[(lhs_id, &lhs_owner), (rhs_id, &rhs_owner)],
                        pool,
                        &mut memory,
                    )
                    .expect("raw PCU add");
                total += start.elapsed();
                backend
                    .download_buffer(pool, outputs[0].1.buffer(), &mut observed)
                    .expect("raw PCU readback");
                verify(&lhs, &rhs, &observed);
                let start = Instant::now();
                drop(black_box(outputs));
                total += start.elapsed();
                lhs_buffer = Some(lhs_owner.into_buffer());
                rhs_buffer = Some(rhs_owner.into_buffer());
            }
            total
        });
    });
    group.bench_function(BenchmarkId::new("source", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut lhs, &mut rhs, job);
                refresh_input(elements, &lhs, &mut source_lhs).expect("refresh source left input");
                refresh_input(elements, &rhs, &mut source_rhs).expect("refresh source right input");
                let start = Instant::now();
                let output = add(&source_lhs, &source_rhs).expect("source add");
                total += start.elapsed();
                output.read_into(&mut observed).expect("source readback");
                verify(&lhs, &rhs, &observed);
                let start = Instant::now();
                drop(black_box(output));
                total += start.elapsed();
            }
            total
        });
    });
    group.bench_function(BenchmarkId::new("native_hip", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut lhs, &mut rhs, job);
                native_lhs
                    .copy_from(bytemuck::cast_slice(&lhs))
                    .expect("refresh native left input");
                native_rhs
                    .copy_from(bytemuck::cast_slice(&rhs))
                    .expect("refresh native right input");
                let start = Instant::now();
                let output = runtime
                    .allocate(elements * size_of::<f32>())
                    .expect("fresh native output");
                let args = [
                    HipKernelArgument::Buffer(&native_lhs),
                    HipKernelArgument::Buffer(&native_rhs),
                    HipKernelArgument::Buffer(&output),
                ];
                // SAFETY: Same shape and ABI as the validated native add kernel.
                let mut completion =
                    unsafe { kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &args) }
                        .expect("native add launch");
                completion.wait().expect("native add terminal completion");
                drop(completion);
                total += start.elapsed();
                output
                    .copy_to(bytemuck::cast_slice_mut(&mut observed))
                    .expect("native readback");
                verify(&lhs, &rhs, &observed);
                let start = Instant::now();
                drop(black_box(output));
                total += start.elapsed();
            }
            total
        });
    });
    group.finish();
    Ok(())
}

fn fill(lhs: &mut [f32], rhs: &mut [f32], job: u64) {
    for (index, (left, right)) in lhs.iter_mut().zip(rhs).enumerate() {
        let tag = f32::from(u16::try_from(index % 2048).expect("sample fits"));
        let phase =
            f32::from_bits(0x3f00_0000 | u32::try_from(job & 0x007f_ffff).expect("phase fits"));
        *left = if index % 2 == 0 {
            tag + phase
        } else {
            -tag - phase
        };
        *right = if index % 3 == 0 { phase } else { -phase };
    }
}

fn verify(lhs: &[f32], rhs: &[f32], observed: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), observed.len());
    assert!(
        lhs.iter()
            .zip(rhs)
            .zip(observed)
            .all(|((&left, &right), &actual)| { actual.to_bits() == (left + right).to_bits() }),
        "binary-add output differs from CPU f32 oracle"
    );
}

fn midpoint(lhs: Duration, rhs: Duration) -> Duration {
    Duration::from_secs_f64(lhs.as_secs_f64().midpoint(rhs.as_secs_f64()))
}

const fn midpoint_f64(lhs: f64, rhs: f64) -> f64 {
    lhs.midpoint(rhs)
}
