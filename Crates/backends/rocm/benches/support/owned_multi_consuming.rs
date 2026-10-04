//! Fresh-output wrapper baseline and two-owner consuming donor binary routes.

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
    PcuExecutionError,
    PcuMemoryPoolId,
    PcuTensor,
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
    DeviceBuffer,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
    compile_hip_source,
};

const BLOCK: u32 = 256;
const PAIRED_ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

#[fusion_pcu::pcu]
fn add_borrowed(
    lhs: &PcuTensor<f32>,
    rhs: &PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

// Ordinary Rust wrapper: the marked operation borrows both live owners, so it must
// publish fresh storage; the two inputs then drop inside this host call.
fn add_fresh_wrapper(
    lhs: PcuTensor<f32>,
    rhs: PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let output = add_borrowed(&lhs, &rhs)?;
    drop((lhs, rhs));
    Ok(output)
}

#[fusion_pcu::pcu]
fn add_owned(
    lhs: PcuTensor<f32>,
    rhs: PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

#[fusion_pcu::pcu]
fn seed(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

struct SampleInputs {
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    oracle: Vec<f32>,
    source: Option<(PcuTensor<f32>, PcuTensor<f32>)>,
    raw: Option<(
        PcuDeviceTensor<f32, RocmMemoryResource>,
        PcuDeviceTensor<f32, RocmMemoryResource>,
    )>,
    native: Option<(DeviceBuffer, DeviceBuffer)>,
}

impl SampleInputs {
    fn prepare(
        elements: usize,
        job: u64,
        backend: &RocmOwnedDispatchBackend,
        runtime: &HipRuntime,
        pool: PcuMemoryPoolId,
    ) -> Result<Self, Box<dyn Error>> {
        let mut lhs = vec![0.0_f32; elements];
        let mut rhs = vec![0.0_f32; elements];
        fill(&mut lhs, &mut rhs, job);
        let oracle = lhs
            .iter()
            .zip(&rhs)
            .map(|(&left, &right)| left + right)
            .collect();

        // All three routes receive distinct fresh backing initialized to the same sample values.
        // These allocations/transfers (and the source seeding calls) stay outside measured time.
        let source = (seed(&lhs)?, seed(&rhs)?);
        let raw = (
            PcuDeviceTensor::new([elements], backend.upload_buffer(pool, &lhs)?)?,
            PcuDeviceTensor::new([elements], backend.upload_buffer(pool, &rhs)?)?,
        );
        let byte_count = elements
            .checked_mul(size_of::<f32>())
            .ok_or("binary-add input size overflow")?;
        let mut native_lhs = runtime.allocate(byte_count)?;
        let mut native_rhs = runtime.allocate(byte_count)?;
        native_lhs.copy_from(bytemuck::cast_slice(&lhs))?;
        native_rhs.copy_from(bytemuck::cast_slice(&rhs))?;

        Ok(Self {
            lhs,
            rhs,
            oracle,
            source: Some(source),
            raw: Some(raw),
            native: Some((native_lhs, native_rhs)),
        })
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
        .ok_or("selected ROCm device has no architecture")?;
    fusion_pcu::global::configure(fusion_pcu::global::PcuExecutionPolicy {
        backend: fusion_pcu::global::PcuBackendChoice::Rocm,
        device: Some(selected.device.id),
        ..fusion_pcu::global::PcuExecutionPolicy::default()
    })?;
    println!("Owned multi-consuming benchmark device: {}", selected.name);
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
#[allow(unsafe_code)] // The native route uses an exact typed HIP kernel with matching extents.
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
    let prepared = crate::support::cold_once("multi-owner raw PCU preparation", || {
        assessor.prepare_owned_program(program)
    })?;
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_add(const float* lhs, const float* rhs, float* output) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) output[id] = lhs[id] + rhs[id]; }}\nextern \"C\" __global__ void native_add_inplace(float* lhs, const float* rhs) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) lhs[id] = lhs[id] + rhs[id]; }}\n"
    );
    let image = crate::support::cold_once("multi-owner native HIP compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(c"native_add")?;
    let inplace_kernel = module.function(c"native_add_inplace")?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(BLOCK);
    let output_bytes = elements
        .checked_mul(size_of::<f32>())
        .ok_or("binary-add output size overflow")?;
    let mut memory = backend.memory_provider(pool);
    let mut observed = vec![0.0_f32; elements];

    // Verify one operation per route before measurement. Every route creates separate fresh
    // inputs from identical CPU values, then the untimed readback checks the independent oracle.
    for route in 0..3 {
        let mut inputs = SampleInputs::prepare(elements, 0, backend, runtime, pool)?;
        let _ = run_route::<false>(
            route,
            &mut inputs,
            &assessor_root,
            &prepared,
            lhs_id,
            rhs_id,
            &mut memory,
            backend,
            runtime,
            &kernel,
            &stream,
            pool,
            grid,
            output_bytes,
            &mut observed,
            None,
        )?;
    }

    let census_inputs = SampleInputs::prepare(elements, 1, backend, runtime, pool)?;
    let census_raw = census(
        0,
        census_inputs,
        &assessor_root,
        &prepared,
        lhs_id,
        rhs_id,
        &mut memory,
        backend,
        runtime,
        &kernel,
        &stream,
        pool,
        grid,
        output_bytes,
        &mut observed,
    )?;
    let census_source_inputs = SampleInputs::prepare(elements, 1, backend, runtime, pool)?;
    let census_source = census(
        1,
        census_source_inputs,
        &assessor_root,
        &prepared,
        lhs_id,
        rhs_id,
        &mut memory,
        backend,
        runtime,
        &kernel,
        &stream,
        pool,
        grid,
        output_bytes,
        &mut observed,
    )?;
    let census_native_inputs = SampleInputs::prepare(elements, 1, backend, runtime, pool)?;
    let census_native = census(
        2,
        census_native_inputs,
        &assessor_root,
        &prepared,
        lhs_id,
        rhs_id,
        &mut memory,
        backend,
        runtime,
        &kernel,
        &stream,
        pool,
        grid,
        output_bytes,
        &mut observed,
    )?;
    println!(
        "Owned multi-consuming fresh-output Rust heap census {elements}: raw PCU {}/{}/{} B, fresh-wrapper source {}/{}/{} B, native HIP {}/{}/{} B allocations/reallocations/requested. Input setup/readback are excluded; output release is included.",
        census_raw.0.alloc_calls,
        census_raw.0.realloc_calls,
        census_raw.0.requested_bytes,
        census_source.0.alloc_calls,
        census_source.0.realloc_calls,
        census_source.0.requested_bytes,
        census_native.0.alloc_calls,
        census_native.0.realloc_calls,
        census_native.0.requested_bytes,
    );

    let mut times = [[Duration::ZERO; 36]; 3];
    let mut ratios = [[0.0_f64; 36]; 2];
    for sample in 0..36 {
        let mut inputs =
            SampleInputs::prepare(elements, u64::try_from(sample + 2)?, backend, runtime, pool)?;
        for route in PAIRED_ORDERS[sample % PAIRED_ORDERS.len()] {
            times[route][sample] = run_route::<false>(
                route,
                &mut inputs,
                &assessor_root,
                &prepared,
                lhs_id,
                rhs_id,
                &mut memory,
                backend,
                runtime,
                &kernel,
                &stream,
                pool,
                grid,
                output_bytes,
                &mut observed,
                None,
            )?;
        }
        ratios[0][sample] = times[0][sample].as_secs_f64() / times[2][sample].as_secs_f64();
        ratios[1][sample] = times[1][sample].as_secs_f64() / times[2][sample].as_secs_f64();
    }
    for route_times in &mut times {
        route_times.sort_unstable();
    }
    for route_ratios in &mut ratios {
        route_ratios.sort_unstable_by(f64::total_cmp);
    }
    println!(
        "Owned multi-consuming paired diagnostic {elements}, 36 balanced verified triples: raw PCU {:?}, source {:?}, native HIP {:?}; paired raw/native {:.4}, source/native {:.4}. Diagnostic medians, not Criterion intervals.",
        midpoint(times[0][17], times[0][18]),
        midpoint(times[1][17], times[1][18]),
        midpoint(times[2][17], times[2][18]),
        midpoint_f64(ratios[0][17], ratios[0][18]),
        midpoint_f64(ratios[1][17], ratios[1][18]),
    );

    let mut group = criterion.benchmark_group("owned_multi_consuming_fresh_output");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    for route in 0..3 {
        let mut job = 40_u64;
        let (name, route_label) = match route {
            0 => ("raw_owned_graph", "raw PCU"),
            1 => ("source_fresh_wrapper", "fresh-output source wrapper"),
            _ => ("native_hip", "native HIP"),
        };
        group.bench_function(BenchmarkId::new(name, elements), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    let mut inputs = SampleInputs::prepare(elements, job, backend, runtime, pool)
                        .expect("prepare matched fresh route inputs");
                    total += run_route::<false>(
                        route,
                        &mut inputs,
                        &assessor_root,
                        &prepared,
                        lhs_id,
                        rhs_id,
                        &mut memory,
                        backend,
                        runtime,
                        &kernel,
                        &stream,
                        pool,
                        grid,
                        output_bytes,
                        &mut observed,
                        None,
                    )
                    .unwrap_or_else(|error| panic!("{route_label} execution failed: {error}"));
                }
                total
            });
        });
    }
    group.finish();

    run_consumed_group(
        criterion,
        backend,
        runtime,
        pool,
        &assessor_root,
        &prepared,
        lhs_id,
        rhs_id,
        &inplace_kernel,
        &stream,
        grid,
        &mut memory,
        elements,
        &mut observed,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(unsafe_code)] // The native route uses matching in-place Add and terminal completion.
fn run_consumed_group(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    pool: PcuMemoryPoolId,
    assessor_root: &RocmOwnedTensorAssessor,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    lhs_id: fusion_pcu::dialect::tensor::ValueId,
    rhs_id: fusion_pcu::dialect::tensor::ValueId,
    inplace_kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
    elements: usize,
    observed: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    // Each policy gets a distinct, equivalent fresh sample. This exercises the raw and source
    // consuming paths without borrowing or reconstructing the designated donor.
    for route in 0..3 {
        let mut inputs = SampleInputs::prepare(elements, 70, backend, runtime, pool)?;
        let _ = run_consumed_route::<false>(
            route,
            &mut inputs,
            assessor_root,
            prepared,
            lhs_id,
            rhs_id,
            memory,
            backend,
            inplace_kernel,
            stream,
            pool,
            grid,
            observed,
            None,
        )?;
    }

    let mut census_counts = [alloc::AllocationCounts::default(); 3];
    for (route, counts_for_route) in census_counts.iter_mut().enumerate() {
        let mut inputs = SampleInputs::prepare(elements, 71, backend, runtime, pool)?;
        let mut counts = alloc::AllocationCounts::default();
        let _ = run_consumed_route::<true>(
            route,
            &mut inputs,
            assessor_root,
            prepared,
            lhs_id,
            rhs_id,
            memory,
            backend,
            inplace_kernel,
            stream,
            pool,
            grid,
            observed,
            Some(&mut counts),
        )?;
        *counts_for_route = counts;
    }
    println!(
        "Owned multi-consuming donor Rust heap census {elements}: raw {}/{}/{} B, two-owner source {}/{}/{} B, native in-place {}/{}/{} B allocations/reallocations/requested. Input setup/readback are excluded; RHS drop is inside execution and donor release after readback is included.",
        census_counts[0].alloc_calls,
        census_counts[0].realloc_calls,
        census_counts[0].requested_bytes,
        census_counts[1].alloc_calls,
        census_counts[1].realloc_calls,
        census_counts[1].requested_bytes,
        census_counts[2].alloc_calls,
        census_counts[2].realloc_calls,
        census_counts[2].requested_bytes,
    );

    let mut times = [[Duration::ZERO; 36]; 3];
    let mut ratios = [[0.0_f64; 36]; 2];
    for sample in 0..36 {
        let mut inputs = SampleInputs::prepare(
            elements,
            u64::try_from(sample + 72)?,
            backend,
            runtime,
            pool,
        )?;
        for route in PAIRED_ORDERS[sample % PAIRED_ORDERS.len()] {
            times[route][sample] = run_consumed_route::<false>(
                route,
                &mut inputs,
                assessor_root,
                prepared,
                lhs_id,
                rhs_id,
                memory,
                backend,
                inplace_kernel,
                stream,
                pool,
                grid,
                observed,
                None,
            )?;
        }
        ratios[0][sample] = times[0][sample].as_secs_f64() / times[2][sample].as_secs_f64();
        ratios[1][sample] = times[1][sample].as_secs_f64() / times[2][sample].as_secs_f64();
    }
    for route_times in &mut times {
        route_times.sort_unstable();
    }
    for route_ratios in &mut ratios {
        route_ratios.sort_unstable_by(f64::total_cmp);
    }
    println!(
        "Owned multi-consuming donor paired diagnostic {elements}, 36 balanced verified triples: raw PCU {:?}, two-owner source {:?}, native in-place {:?}; paired raw/native {:.4}, source/native {:.4}. Diagnostic medians, not Criterion intervals.",
        midpoint(times[0][17], times[0][18]),
        midpoint(times[1][17], times[1][18]),
        midpoint(times[2][17], times[2][18]),
        midpoint_f64(ratios[0][17], ratios[0][18]),
        midpoint_f64(ratios[1][17], ratios[1][18]),
    );

    let mut group = criterion.benchmark_group("owned_multi_consuming_donor_inplace");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    for route in 0..3 {
        let mut job = 120_u64;
        let (name, route_label) = match route {
            0 => ("raw_consuming_pair", "raw consuming pair"),
            1 => ("source_two_owner", "two-owner source"),
            _ => ("native_inplace", "native in-place"),
        };
        group.bench_function(BenchmarkId::new(name, elements), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    let mut inputs = SampleInputs::prepare(elements, job, backend, runtime, pool)
                        .expect("prepare matching fresh input owners");
                    total += run_consumed_route::<false>(
                        route,
                        &mut inputs,
                        assessor_root,
                        prepared,
                        lhs_id,
                        rhs_id,
                        memory,
                        backend,
                        inplace_kernel,
                        stream,
                        pool,
                        grid,
                        observed,
                        None,
                    )
                    .unwrap_or_else(|error| panic!("{route_label} execution failed: {error}"));
                }
                total
            });
        });
    }
    group.finish();
    Ok(())
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)] // Native HIP launch uses a matching exact-buffer kernel ABI.
fn run_route<const TRACK_ALLOCATIONS: bool>(
    route: usize,
    inputs: &mut SampleInputs,
    assessor_root: &RocmOwnedTensorAssessor,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    lhs_id: fusion_pcu::dialect::tensor::ValueId,
    rhs_id: fusion_pcu::dialect::tensor::ValueId,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    pool: PcuMemoryPoolId,
    grid: u32,
    output_bytes: usize,
    observed: &mut [f32],
    mut allocation_counts: Option<&mut alloc::AllocationCounts>,
) -> Result<Duration, Box<dyn Error>> {
    match route {
        0 => {
            let (lhs, rhs) = inputs.raw.take().expect("raw route inputs are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let start = Instant::now();
            let outputs = assessor_root.assessor().execute_owned_program_outputs(
                prepared,
                &[(lhs_id, &lhs), (rhs_id, &rhs)],
                pool,
                memory,
            )?;
            drop((lhs, rhs));
            let execution = start.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            backend.download_buffer(pool, outputs[0].1.buffer(), observed)?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let start = Instant::now();
            drop(black_box(outputs));
            let release = start.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        1 => {
            let (lhs, rhs) = inputs.source.take().expect("source route inputs are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let start = Instant::now();
            let output = add_fresh_wrapper(lhs, rhs)?;
            let execution = start.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            output.read_into(observed)?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let start = Instant::now();
            drop(black_box(output));
            let release = start.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        2 => {
            let (lhs, rhs) = inputs.native.take().expect("native route inputs are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let start = Instant::now();
            let output = runtime.allocate(output_bytes)?;
            let arguments = [
                HipKernelArgument::Buffer(&lhs),
                HipKernelArgument::Buffer(&rhs),
                HipKernelArgument::Buffer(&output),
            ];
            // SAFETY: Input buffers each contain `elements` f32s; output, grid and kernel ABI match.
            let mut completion =
                unsafe { kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
            completion.wait()?;
            drop(completion);
            drop((lhs, rhs));
            let execution = start.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            output.copy_to(bytemuck::cast_slice_mut(observed))?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let start = Instant::now();
            drop(black_box(output));
            let release = start.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        _ => unreachable!("three routes are declared"),
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)] // Native HIP launch is the matching in-place consuming control.
fn run_consumed_route<const TRACK_ALLOCATIONS: bool>(
    route: usize,
    inputs: &mut SampleInputs,
    assessor_root: &RocmOwnedTensorAssessor,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    lhs_id: fusion_pcu::dialect::tensor::ValueId,
    rhs_id: fusion_pcu::dialect::tensor::ValueId,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
    backend: &RocmOwnedDispatchBackend,
    inplace_kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    pool: PcuMemoryPoolId,
    grid: u32,
    observed: &mut [f32],
    mut allocation_counts: Option<&mut alloc::AllocationCounts>,
) -> Result<Duration, Box<dyn Error>> {
    match route {
        0 => {
            let (lhs, rhs) = inputs.raw.take().expect("raw owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let output = assessor_root
                .assessor()
                .execute_owned_program_consuming_binary_pair(
                    prepared,
                    [(lhs_id, lhs), (rhs_id, rhs)],
                    pool,
                    memory,
                )?;
            let execution = started.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            backend.download_buffer(pool, output.buffer(), observed)?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            drop(black_box(output));
            let release = started.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        1 => {
            let (lhs, rhs) = inputs.source.take().expect("source owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let output = add_owned(lhs, rhs)?;
            let execution = started.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            output.read_into(observed)?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            drop(black_box(output));
            let release = started.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        2 => {
            let (lhs, rhs) = inputs.native.take().expect("native owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let arguments = [
                HipKernelArgument::Buffer(&lhs),
                HipKernelArgument::Buffer(&rhs),
            ];
            // SAFETY: The in-place kernel writes the full `lhs` allocation, reads `rhs`, and the
            // checked launch extent covers exactly the two `elements`-sized f32 inputs.
            let mut completion = unsafe {
                inplace_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments)
            }?;
            completion.wait()?;
            drop(completion);
            drop(rhs);
            let execution = started.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            lhs.copy_to(bytemuck::cast_slice_mut(observed))?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            drop(black_box(lhs));
            let release = started.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        _ => unreachable!("three consumed routes are declared"),
    }
}

#[allow(clippy::too_many_arguments)]
fn census(
    route: usize,
    mut inputs: SampleInputs,
    assessor_root: &RocmOwnedTensorAssessor,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    lhs_id: fusion_pcu::dialect::tensor::ValueId,
    rhs_id: fusion_pcu::dialect::tensor::ValueId,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    pool: PcuMemoryPoolId,
    grid: u32,
    output_bytes: usize,
    observed: &mut [f32],
) -> Result<(alloc::AllocationCounts, Duration), Box<dyn Error>> {
    let mut counts = alloc::AllocationCounts::default();
    let elapsed = run_route::<true>(
        route,
        &mut inputs,
        assessor_root,
        prepared,
        lhs_id,
        rhs_id,
        memory,
        backend,
        runtime,
        kernel,
        stream,
        pool,
        grid,
        output_bytes,
        observed,
        Some(&mut counts),
    )?;
    Ok((counts, elapsed))
}

fn start_capture<const TRACK_ALLOCATIONS: bool>() -> Option<alloc::AllocationCapture> {
    if TRACK_ALLOCATIONS {
        Some(alloc::AllocationCapture::start())
    } else {
        None
    }
}

fn record_capture(
    capture: Option<alloc::AllocationCapture>,
    destination: Option<&mut alloc::AllocationCounts>,
) {
    if let Some(capture) = capture {
        let captured = alloc::AllocationCapture::finish();
        drop(capture);
        if let Some(destination) = destination {
            destination.alloc_calls += captured.alloc_calls;
            destination.realloc_calls += captured.realloc_calls;
            destination.dealloc_calls += captured.dealloc_calls;
            destination.requested_bytes += captured.requested_bytes;
        }
    }
}

fn fill(lhs: &mut [f32], rhs: &mut [f32], job: u64) {
    for (index, (left, right)) in lhs.iter_mut().zip(rhs).enumerate() {
        let lane = f32::from(u16::try_from(index % 2048).expect("lane fits"));
        let shift = u32::try_from((index % 4) * 16).expect("shift fits");
        let job_chunk = f32::from(u16::try_from((job >> shift) & 0xffff).expect("chunk fits"));
        let phase = f32::from(u16::try_from(job % 97).expect("phase fits")) * 0.125;
        // Four disjoint 16-bit chunks in the first four exact f32 values distinguish every job
        // counter; the fractional lane term keeps the remaining payload nonuniform.
        *left = lane.mul_add(0.0625, job_chunk);
        *right = if index.is_multiple_of(3) {
            phase
        } else {
            -phase
        };
    }
}

fn verify(lhs: &[f32], rhs: &[f32], oracle: &[f32], observed: &[f32]) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), oracle.len());
    assert_eq!(lhs.len(), observed.len());
    for (((&left, &right), &expected), &actual) in lhs.iter().zip(rhs).zip(oracle).zip(observed) {
        assert_eq!(expected.to_bits(), (left + right).to_bits());
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

fn midpoint(lhs: Duration, rhs: Duration) -> Duration {
    Duration::from_secs_f64(lhs.as_secs_f64().midpoint(rhs.as_secs_f64()))
}

const fn midpoint_f64(lhs: f64, rhs: f64) -> f64 {
    lhs.midpoint(rhs)
}
