//! Fresh-output and consumed-in-place `ReLU` routes with matching input refresh policies.

#[path = "alloc.rs"]
#[allow(dead_code)]
mod alloc;

#[rustfmt::skip]
use std::{
    error::Error,
    hint::black_box,
    mem::size_of,
    rc::Rc,
    time::Duration,
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
    HipKernel,
    HipKernelArgument,
    HipStreamHandle,
    HipRuntime,
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    RocmOwnedTensorAssessor,
    compile_hip_source,
};

const BLOCK: u32 = 256;
const SAMPLES: usize = 24;
const FRESH_ORDERS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];
const CONSUMED_ORDERS: [[usize; 2]; 2] = [[0, 1], [1, 0]];

#[fusion_pcu::pcu]
fn source_relu_borrowed(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::relu(input)
}

#[fusion_pcu::pcu]
fn source_relu_helper(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let activated = pcu::relu(input)?;
    Ok(activated)
}

#[fusion_pcu::pcu]
fn source_relu_consuming(input: PcuTensor<f32>) -> Result<PcuTensor<f32>, PcuExecutionError> {
    let input = input;
    let input = input;
    Ok(source_relu_helper(input)?)
}

#[fusion_pcu::pcu]
fn source_identity(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu(invocations: N)]
fn source_refresh<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id];
}

fn refresh_source_for_extent(
    elements: usize,
    input: &[f32],
    output: &mut PcuTensor<f32>,
) -> Result<(), PcuExecutionError> {
    match elements {
        65 => source_refresh::<65>(input, output),
        1_048_576 => source_refresh::<1_048_576>(input, output),
        _ => unreachable!("benchmark declares both extents"),
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
    println!("Owned ReLU benchmark device: {}", selected.name);
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
#[allow(clippy::branches_sharing_code)]
#[allow(unsafe_code)] // Native HIP kernels use the same ReLU operation and checked extents.
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
    let input_id = graph.input([elements], fusion_pcu::PcuScalarType::F32)?;
    let output_id = graph.relu(input_id)?;
    let program = graph.into_selected_program(
        &[output_id],
        TensorArithmeticRewritePolicy::Disabled,
        TensorArithmeticCapability::Strict,
        TensorPointwiseGroupingPolicy::Disabled,
    )?;
    let prepared = crate::support::cold_once("owned ReLU graph preparation", || {
        assessor.prepare_owned_program(program)
    })?;
    let hip_source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_relu_fresh(const float* input, float* output) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) output[id] = input[id] > 0.0f ? input[id] : 0.0f; }}\nextern \"C\" __global__ void native_relu_inplace(float* values) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) values[id] = values[id] > 0.0f ? values[id] : 0.0f; }}\n"
    );
    let image = crate::support::cold_once("native ReLU HIP compilation", || {
        compile_hip_source(&hip_source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let fresh_kernel = module.function(c"native_relu_fresh")?;
    let inplace_kernel = module.function(c"native_relu_inplace")?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(BLOCK);

    let mut host = vec![0.0_f32; elements];
    let mut observed = vec![0.0_f32; elements];
    fill(&mut host, 0);
    let mut borrowed_input = Some(source_identity(&host)?);
    let mut consumed_input = Some(source_identity(&host)?);
    let mut raw_input = Some(PcuDeviceTensor::new(
        [elements],
        backend.upload_buffer(pool, &host)?,
    )?);
    let mut native_fresh_input = Some(runtime.allocate(elements * size_of::<f32>())?);
    native_fresh_input
        .as_mut()
        .expect("native fresh input")
        .copy_from(bytemuck::cast_slice(&host))?;
    let mut native_inplace_input = Some(runtime.allocate(elements * size_of::<f32>())?);
    native_inplace_input
        .as_mut()
        .expect("native in-place input")
        .copy_from(bytemuck::cast_slice(&host))?;
    let mut memory = backend.memory_provider(pool);

    // Compile and verify every route before timing. Each route has independent input storage.
    let borrowed_output = source_relu_borrowed(borrowed_input.as_ref().expect("borrowed input"))?;
    borrowed_output.read_into(&mut observed)?;
    verify(&host, &observed);
    drop(borrowed_output);

    let consumed_output = source_relu_consuming(consumed_input.take().expect("consumed input"))?;
    consumed_output.read_into(&mut observed)?;
    verify(&host, &observed);
    consumed_input = Some(consumed_output);

    let raw_owner = raw_input.as_ref().expect("raw input");
    let raw_outputs = assessor.execute_owned_program_outputs(
        &prepared,
        &[(input_id, raw_owner)],
        pool,
        &mut memory,
    )?;
    let (_, raw_output) = raw_outputs.into_iter().next().expect("one ReLU output");
    backend.download_buffer(pool, raw_output.buffer(), &mut observed)?;
    verify(&host, &observed);
    drop(raw_output);

    let native_fresh_owner = native_fresh_input.as_ref().expect("native fresh input");
    let native_fresh_output = runtime.allocate(elements * size_of::<f32>())?;
    let arguments = [
        HipKernelArgument::Buffer(native_fresh_owner),
        HipKernelArgument::Buffer(&native_fresh_output),
    ];
    // SAFETY: Both buffers cover `elements` f32 values matching native_relu_fresh's ABI.
    let mut completion =
        unsafe { fresh_kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
    completion.wait()?;
    drop(completion);
    native_fresh_output.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
    verify(&host, &observed);
    drop(native_fresh_output);

    let native_inplace_owner = native_inplace_input
        .as_ref()
        .expect("native in-place input");
    let arguments = [HipKernelArgument::Buffer(native_inplace_owner)];
    // SAFETY: This writable buffer covers the complete in-place ReLU extent.
    let mut completion =
        unsafe { inplace_kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
    completion.wait()?;
    drop(completion);
    native_inplace_owner.copy_to(bytemuck::cast_slice_mut(&mut observed))?;
    verify(&host, &observed);
    native_inplace_input
        .as_mut()
        .expect("native in-place input")
        .copy_from(bytemuck::cast_slice(&host))?;

    print_heap_census(
        elements,
        &assessor_root,
        &prepared,
        backend,
        runtime,
        &fresh_kernel,
        &inplace_kernel,
        &stream,
        grid,
        pool,
        input_id,
        borrowed_input.as_ref(),
        &mut consumed_input,
        raw_input.as_ref(),
        native_fresh_input.as_ref(),
        native_inplace_input.as_ref(),
        &mut memory,
    )?;

    let fresh_times = paired_fresh_diagnostic(
        SAMPLES,
        elements,
        &mut host,
        &mut observed,
        &assessor_root,
        &prepared,
        backend,
        runtime,
        &fresh_kernel,
        &stream,
        grid,
        pool,
        input_id,
        &mut borrowed_input,
        &mut raw_input,
        &mut native_fresh_input,
        &mut memory,
    )?;
    print_three_route_diagnostic(elements, &fresh_times, "borrowed/raw/native fresh-output");
    let consumed_times = paired_consumed_diagnostic(
        SAMPLES,
        elements,
        &mut host,
        &mut observed,
        &inplace_kernel,
        &stream,
        grid,
        &mut consumed_input,
        &mut native_inplace_input,
    )?;
    print_two_route_diagnostic(elements, &consumed_times);

    let mut job = u64::try_from(SAMPLES + 1)?;
    let mut fresh_group = criterion.benchmark_group("owned_relu_fresh_output");
    fresh_group.throughput(Throughput::Elements(u64::try_from(elements)?));
    fresh_group.bench_function(BenchmarkId::new("source_borrowed", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                refresh_fresh_inputs(
                    elements,
                    &host,
                    &mut borrowed_input,
                    &mut raw_input,
                    &mut native_fresh_input,
                    backend,
                    pool,
                )
                .expect("refresh fresh-output inputs");
                let start = std::time::Instant::now();
                let output = source_relu_borrowed(borrowed_input.as_ref().expect("borrowed input"))
                    .expect("borrowed source ReLU");
                total += start.elapsed();
                output.read_into(&mut observed).expect("source readback");
                verify(&host, &observed);
                let release = std::time::Instant::now();
                drop(black_box(output));
                total += release.elapsed();
            }
            total
        });
    });
    fresh_group.bench_function(BenchmarkId::new("raw_prepared", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                refresh_fresh_inputs(
                    elements,
                    &host,
                    &mut borrowed_input,
                    &mut raw_input,
                    &mut native_fresh_input,
                    backend,
                    pool,
                )
                .expect("refresh fresh-output inputs");
                let input = raw_input.as_ref().expect("raw input");
                let start = std::time::Instant::now();
                let outputs = assessor_root
                    .assessor()
                    .execute_owned_program_outputs(
                        &prepared,
                        &[(input_id, input)],
                        pool,
                        &mut memory,
                    )
                    .expect("raw fresh-output ReLU");
                let (_, output) = outputs.into_iter().next().expect("one raw output");
                total += start.elapsed();
                backend
                    .download_buffer(pool, output.buffer(), &mut observed)
                    .expect("raw output readback");
                verify(&host, &observed);
                let release = std::time::Instant::now();
                drop(black_box(output));
                total += release.elapsed();
            }
            total
        });
    });
    fresh_group.bench_function(BenchmarkId::new("native_fresh", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                refresh_fresh_inputs(
                    elements,
                    &host,
                    &mut borrowed_input,
                    &mut raw_input,
                    &mut native_fresh_input,
                    backend,
                    pool,
                )
                .expect("refresh fresh-output inputs");
                let input = native_fresh_input.as_ref().expect("native fresh input");
                let start = std::time::Instant::now();
                let output = runtime
                    .allocate(elements * size_of::<f32>())
                    .expect("native fresh output");
                let arguments = [
                    HipKernelArgument::Buffer(input),
                    HipKernelArgument::Buffer(&output),
                ];
                // SAFETY: The two buffers match native_relu_fresh's full-sized f32 signature.
                let mut completion = unsafe {
                    fresh_kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments)
                }
                .expect("native fresh ReLU launch");
                completion.wait().expect("native terminal completion");
                drop(completion);
                total += start.elapsed();
                output
                    .copy_to(bytemuck::cast_slice_mut(&mut observed))
                    .expect("native fresh output readback");
                verify(&host, &observed);
                let release = std::time::Instant::now();
                drop(black_box(output));
                total += release.elapsed();
            }
            total
        });
    });
    fresh_group.finish();

    let mut consumed_group = criterion.benchmark_group("owned_relu_consumed_inplace");
    consumed_group.throughput(Throughput::Elements(u64::try_from(elements)?));
    consumed_group.bench_function(BenchmarkId::new("source_consuming", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                refresh_consumed_inputs(
                    elements,
                    &host,
                    &mut consumed_input,
                    &mut native_inplace_input,
                )
                .expect("refresh consuming/in-place inputs");
                let input = consumed_input.take().expect("consumed input");
                let start = std::time::Instant::now();
                let output = source_relu_consuming(input).expect("consuming source ReLU");
                total += start.elapsed();
                output
                    .read_into(&mut observed)
                    .expect("consumed output readback");
                verify(&host, &observed);
                consumed_input = Some(output);
            }
            total
        });
    });
    consumed_group.bench_function(BenchmarkId::new("native_inplace", elements), |bencher| {
        bencher.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                job = job.wrapping_add(1);
                fill(&mut host, job);
                refresh_consumed_inputs(
                    elements,
                    &host,
                    &mut consumed_input,
                    &mut native_inplace_input,
                )
                .expect("refresh consuming/in-place inputs");
                let start = std::time::Instant::now();
                let input = native_inplace_input
                    .as_ref()
                    .expect("native in-place input");
                let arguments = [HipKernelArgument::Buffer(input)];
                // SAFETY: The single buffer is mutable storage for native_relu_inplace.
                let mut completion = unsafe {
                    inplace_kernel.launch(&stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments)
                }
                .expect("native in-place ReLU launch");
                completion.wait().expect("native terminal completion");
                drop(completion);
                total += start.elapsed();
                input
                    .copy_to(bytemuck::cast_slice_mut(&mut observed))
                    .expect("native in-place readback");
                verify(&host, &observed);
            }
            total
        });
    });
    consumed_group.finish();
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(unsafe_code)]
fn print_heap_census(
    elements: usize,
    assessor: &RocmOwnedTensorAssessor,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    fresh_kernel: &HipKernel,
    inplace_kernel: &HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    pool: PcuMemoryPoolId,
    input_id: fusion_pcu::dialect::tensor::ValueId,
    borrowed_input: Option<&PcuTensor<f32>>,
    consumed_input: &mut Option<PcuTensor<f32>>,
    raw_input: Option<&PcuDeviceTensor<f32, RocmMemoryResource>>,
    native_fresh_input: Option<&fusion_pcu_rocm::DeviceBuffer>,
    native_inplace_input: Option<&fusion_pcu_rocm::DeviceBuffer>,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<(), Box<dyn Error>> {
    let mut observed = vec![0.0_f32; elements];
    let capture = alloc::AllocationCapture::start();
    let output = source_relu_borrowed(borrowed_input.expect("borrowed input"))?;
    drop(output);
    let borrowed = alloc::AllocationCapture::finish();
    drop(capture);

    let capture = alloc::AllocationCapture::start();
    let output = source_relu_consuming(consumed_input.take().expect("consumed input"))?;
    let consuming = alloc::AllocationCapture::finish();
    drop(capture);
    *consumed_input = Some(output);

    let capture = alloc::AllocationCapture::start();
    let input = raw_input.expect("raw input");
    let outputs = assessor.assessor().execute_owned_program_outputs(
        prepared,
        &[(input_id, input)],
        pool,
        memory,
    )?;
    let (_, output) = outputs.into_iter().next().expect("one output");
    drop(output);
    let raw = alloc::AllocationCapture::finish();
    drop(capture);

    let capture = alloc::AllocationCapture::start();
    let input = native_fresh_input.expect("native fresh input");
    let output = runtime.allocate(elements * size_of::<f32>())?;
    let arguments = [
        HipKernelArgument::Buffer(input),
        HipKernelArgument::Buffer(&output),
    ];
    // SAFETY: Both buffers have the exact native fresh-output kernel extent and type.
    let mut completion =
        unsafe { fresh_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
    completion.wait()?;
    drop(completion);
    drop(output);
    let native_fresh = alloc::AllocationCapture::finish();
    drop(capture);

    let capture = alloc::AllocationCapture::start();
    let input = native_inplace_input.expect("native in-place input");
    let arguments = [HipKernelArgument::Buffer(input)];
    // SAFETY: This buffer has the full extent expected by the in-place kernel.
    let mut completion =
        unsafe { inplace_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
    completion.wait()?;
    drop(completion);
    let native_inplace = alloc::AllocationCapture::finish();
    drop(capture);

    let _ = (backend, &mut observed);
    println!(
        "Owned ReLU Rust heap census {elements}: borrowed {}/{}/{} B, consuming {}/{}/{} B, raw fresh {}/{}/{} B, native fresh {}/{}/{} B, native in-place {}/{}/{} B allocations/reallocations/requested. Rust allocator counts do not include HIP/driver allocations and do not prove physical reuse.",
        borrowed.alloc_calls,
        borrowed.realloc_calls,
        borrowed.requested_bytes,
        consuming.alloc_calls,
        consuming.realloc_calls,
        consuming.requested_bytes,
        raw.alloc_calls,
        raw.realloc_calls,
        raw.requested_bytes,
        native_fresh.alloc_calls,
        native_fresh.realloc_calls,
        native_fresh.requested_bytes,
        native_inplace.alloc_calls,
        native_inplace.realloc_calls,
        native_inplace.requested_bytes,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(unsafe_code)]
fn paired_fresh_diagnostic(
    samples: usize,
    elements: usize,
    host: &mut [f32],
    observed: &mut [f32],
    assessor_root: &RocmOwnedTensorAssessor,
    prepared: &fusion_pcu_rocm::RocmOwnedPreparedTensorGraph,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    fresh_kernel: &HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    pool: PcuMemoryPoolId,
    input_id: fusion_pcu::dialect::tensor::ValueId,
    borrowed_input: &mut Option<PcuTensor<f32>>,
    raw_input: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    native_input: &mut Option<fusion_pcu_rocm::DeviceBuffer>,
    memory: &mut impl fusion_pcu::PcuMemoryProvider<Resource = RocmMemoryResource>,
) -> Result<[[Duration; 3]; SAMPLES], Box<dyn Error>> {
    let mut times = [[Duration::ZERO; 3]; SAMPLES];
    for sample in 0..samples {
        fill(host, u64::try_from(sample + 1)?);
        refresh_source_for_extent(
            elements,
            host,
            borrowed_input.as_mut().expect("borrowed input"),
        )?;
        refresh_raw(backend, pool, elements, host, raw_input)?;
        native_input
            .as_mut()
            .expect("native input")
            .copy_from(bytemuck::cast_slice(host))?;
        for route in FRESH_ORDERS[sample % FRESH_ORDERS.len()] {
            match route {
                0 => {
                    let start = std::time::Instant::now();
                    let output =
                        source_relu_borrowed(borrowed_input.as_ref().expect("borrowed input"))?;
                    let execution = start.elapsed();
                    output.read_into(observed)?;
                    verify(host, observed);
                    let release = std::time::Instant::now();
                    drop(black_box(output));
                    times[sample][0] = execution + release.elapsed();
                }
                1 => {
                    let input = raw_input.as_ref().expect("raw input");
                    let start = std::time::Instant::now();
                    let outputs = assessor_root.assessor().execute_owned_program_outputs(
                        prepared,
                        &[(input_id, input)],
                        pool,
                        memory,
                    )?;
                    let (_, output) = outputs.into_iter().next().expect("one raw output");
                    let execution = start.elapsed();
                    backend.download_buffer(pool, output.buffer(), observed)?;
                    verify(host, observed);
                    let release = std::time::Instant::now();
                    drop(black_box(output));
                    times[sample][1] = execution + release.elapsed();
                }
                2 => {
                    let input = native_input.as_ref().expect("native input");
                    let start = std::time::Instant::now();
                    let output = runtime.allocate(elements * size_of::<f32>())?;
                    let arguments = [
                        HipKernelArgument::Buffer(input),
                        HipKernelArgument::Buffer(&output),
                    ];
                    // SAFETY: Buffers match native_relu_fresh's exact f32 signature and extent.
                    let mut completion = unsafe {
                        fresh_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments)
                    }?;
                    completion.wait()?;
                    drop(completion);
                    let execution = start.elapsed();
                    output.copy_to(bytemuck::cast_slice_mut(observed))?;
                    verify(host, observed);
                    let release = std::time::Instant::now();
                    drop(black_box(output));
                    times[sample][2] = execution + release.elapsed();
                }
                _ => unreachable!("three fresh-output routes"),
            }
        }
    }
    Ok(times)
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)]
#[allow(unsafe_code)]
fn paired_consumed_diagnostic(
    samples: usize,
    elements: usize,
    host: &mut [f32],
    observed: &mut [f32],
    inplace_kernel: &HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    consumed_input: &mut Option<PcuTensor<f32>>,
    native_input: &mut Option<fusion_pcu_rocm::DeviceBuffer>,
) -> Result<[[Duration; 2]; SAMPLES], Box<dyn Error>> {
    let mut times = [[Duration::ZERO; 2]; SAMPLES];
    for sample in 0..samples {
        fill(host, u64::try_from(sample + 1)?);
        refresh_source_for_extent(
            elements,
            host,
            consumed_input.as_mut().expect("consumed source input"),
        )?;
        native_input
            .as_mut()
            .expect("native in-place input")
            .copy_from(bytemuck::cast_slice(host))?;
        for route in CONSUMED_ORDERS[sample % CONSUMED_ORDERS.len()] {
            match route {
                0 => {
                    let input = consumed_input.take().expect("consumed source input");
                    let start = std::time::Instant::now();
                    let output = source_relu_consuming(input)?;
                    times[sample][0] = start.elapsed();
                    output.read_into(observed)?;
                    verify(host, observed);
                    *consumed_input = Some(output);
                }
                1 => {
                    let input = native_input.as_ref().expect("native in-place input");
                    let start = std::time::Instant::now();
                    let arguments = [HipKernelArgument::Buffer(input)];
                    // SAFETY: The single f32 buffer is the kernel's read/write in-place argument.
                    let mut completion = unsafe {
                        inplace_kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments)
                    }?;
                    completion.wait()?;
                    drop(completion);
                    times[sample][1] = start.elapsed();
                    input.copy_to(bytemuck::cast_slice_mut(observed))?;
                    verify(host, observed);
                }
                _ => unreachable!("two consumed/in-place routes"),
            }
        }
    }
    Ok(times)
}

fn refresh_raw(
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    elements: usize,
    values: &[f32],
    owner: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
) -> Result<(), Box<dyn Error>> {
    let current = owner.take().expect("raw input owner");
    let mut buffer = current.into_buffer();
    backend.refresh_buffer(pool, &mut buffer, values)?;
    *owner = Some(PcuDeviceTensor::new([elements], buffer)?);
    Ok(())
}

fn refresh_fresh_inputs(
    elements: usize,
    host: &[f32],
    borrowed_input: &mut Option<PcuTensor<f32>>,
    raw_input: &mut Option<PcuDeviceTensor<f32, RocmMemoryResource>>,
    native_input: &mut Option<fusion_pcu_rocm::DeviceBuffer>,
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
) -> Result<(), Box<dyn Error>> {
    refresh_source_for_extent(
        elements,
        host,
        borrowed_input.as_mut().expect("borrowed input"),
    )?;
    refresh_raw(backend, pool, elements, host, raw_input)?;
    native_input
        .as_mut()
        .expect("native fresh input")
        .copy_from(bytemuck::cast_slice(host))?;
    Ok(())
}

fn refresh_consumed_inputs(
    elements: usize,
    host: &[f32],
    source_input: &mut Option<PcuTensor<f32>>,
    native_input: &mut Option<fusion_pcu_rocm::DeviceBuffer>,
) -> Result<(), Box<dyn Error>> {
    refresh_source_for_extent(
        elements,
        host,
        source_input.as_mut().expect("consumed source input"),
    )?;
    native_input
        .as_mut()
        .expect("native in-place input")
        .copy_from(bytemuck::cast_slice(host))?;
    Ok(())
}

fn fill(values: &mut [f32], job: u64) {
    let phase = f32::from(u8::try_from(job % 17).expect("phase fits")) / 16.0;
    for (index, value) in values.iter_mut().enumerate() {
        let base = f32::from(u16::try_from(index % 1024).expect("sample fits"));
        *value = if (index + usize::try_from(job).expect("job fits")) % 2 == 0 {
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
        "ReLU output differs from independent CPU oracle"
    );
}

fn print_three_route_diagnostic(elements: usize, times: &[[Duration; 3]; SAMPLES], label: &str) {
    let mut borrowed = [Duration::ZERO; SAMPLES];
    let mut raw = [Duration::ZERO; SAMPLES];
    let mut native = [Duration::ZERO; SAMPLES];
    for (index, [borrowed_time, raw_time, native_time]) in times.iter().copied().enumerate() {
        borrowed[index] = borrowed_time;
        raw[index] = raw_time;
        native[index] = native_time;
    }
    borrowed.sort_unstable();
    raw.sort_unstable();
    native.sort_unstable();
    println!(
        "Owned ReLU {label} diagnostic {elements}: borrowed-source {:?}, raw-PCU {:?}, native {:?}; matched inputs refreshed outside timing; timings include terminal completion and fresh-output release. Medians are paired host diagnostics, not Criterion intervals.",
        (borrowed[11] + borrowed[12]) / 2,
        (raw[11] + raw[12]) / 2,
        (native[11] + native[12]) / 2,
    );
}

fn print_two_route_diagnostic(elements: usize, times: &[[Duration; 2]; SAMPLES]) {
    let mut source = [Duration::ZERO; SAMPLES];
    let mut native = [Duration::ZERO; SAMPLES];
    for (index, [source_time, native_time]) in times.iter().copied().enumerate() {
        source[index] = source_time;
        native[index] = native_time;
    }
    source.sort_unstable();
    native.sort_unstable();
    println!(
        "Owned ReLU consumed/in-place diagnostic {elements}: consuming-source {:?}, native-in-place {:?}; matched inputs refreshed outside timing and terminal completion included. No donation claim is inferred from timing. Medians are paired host diagnostics, not Criterion intervals.",
        (source[11] + source[12]) / 2,
        (native[11] + native[12]) / 2,
    );
}
