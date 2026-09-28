//! Shared host-slice and resident-buffer benchmark implementation.

#[path = "alloc.rs"]
#[allow(dead_code)] // The shared allocator helper also exposes phased counters for other benches.
mod alloc;

use std::{
    error::Error,
    hint::black_box,
    time::{
        Duration,
        Instant,
    },
};

use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
};
use fusion_pcu::{
    PcuBindingRef,
    PcuDeviceBuffer,
    PcuHostArgument,
    PcuMemoryPoolId,
};
use fusion_pcu_macros::pcu;
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernel,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    RocmMemoryResource,
    RocmOwnedDispatchBackend,
    compile_hip_source,
};

#[pcu(invocations = N)]
fn transform<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = pcu::context::global_invocation_id();
    output[id] = input[id] * 2.0 + 1.0;
}

pub fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = crate::support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    if candidates.is_empty() {
        return Err(
            "no selected ROCm device has an architecture for native HIP compilation".into(),
        );
    }
    let (backend, selected) = crate::support::selection::open_ranked(&discovery, candidates, 256)?;
    let architecture = selected
        .architecture
        .as_deref()
        .ok_or("selected device has no HIP architecture")?;
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "Typed kernel benchmark device: {} ({architecture})",
        selected.name
    );

    run_case::<65>(
        criterion,
        &backend,
        &runtime,
        &selected.name,
        selected.pool,
        architecture,
    )?;
    run_case::<1_048_576>(
        criterion,
        &backend,
        &runtime,
        &selected.name,
        selected.pool,
        architecture,
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    device: &str,
    pool: PcuMemoryPoolId,
    architecture: &str,
) -> Result<(), Box<dyn Error>> {
    println!("Typed kernel shape {N} on {device}");
    let mut host_call = crate::support::cold_once("typed PCU host preparation", || {
        transform_prepare::<N, _>(backend)
    })?;
    let mut resident_call = crate::support::cold_once("typed PCU resident preparation", || {
        transform_prepare_device::<N, _>(backend)
    })?;
    let native_source = format!(
        "#include <hip/hip_runtime.h>\n#pragma clang fp contract(off)\nextern \"C\" __global__ void native_transform(const float* input, float* output) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {N}u) output[id] = input[id] * 2.0f + 1.0f; }}\n"
    );
    let image = crate::support::cold_once("native HIP compilation", || {
        compile_hip_source(&native_source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let function = module.function(c"native_transform")?;
    let stream = runtime.create_stream()?;
    let mut host_input = Vec::with_capacity(N);
    let mut cpu_output = vec![0.0_f32; N];
    let initial_output = vec![-5.0_f32; N];
    let mut host_output = initial_output.clone();
    let mut native_input = runtime.allocate(N * core::mem::size_of::<f32>())?;
    let mut native_output = runtime.allocate(N * core::mem::size_of::<f32>())?;
    let mut native_result = vec![0.0_f32; N];
    let grid = u32::try_from(N)?.div_ceil(256);

    preflight_host::<N, _>(&mut host_call, &mut host_output, &mut cpu_output)?;
    sample_host::<N, _>(
        criterion,
        &mut host_call,
        &mut host_input,
        &mut host_output,
        &mut cpu_output,
        &initial_output,
        &mut native_input,
        &mut native_output,
        &function,
        &stream,
        grid,
    )?;
    allocation_diagnostic::<N, _>(
        &mut host_call,
        &mut host_output,
        &initial_output,
        &mut native_input,
        &mut native_output,
        &mut native_result,
        &function,
        &stream,
        grid,
    )?;

    let mut resident_input = backend.upload_buffer(pool, &fresh_input::<N>(0))?;
    let mut resident_output = backend.upload_buffer(pool, &initial_output)?;
    let mut native_resident_input = runtime.allocate(N * core::mem::size_of::<f32>())?;
    let native_resident_output = runtime.allocate(N * core::mem::size_of::<f32>())?;
    // Warm up the reusable paths and inspect benchmark-thread Rust allocations around a call.
    host_call(&fresh_input::<N>(1), &mut host_output)?;
    backend.refresh_buffer(pool, &mut resident_input, &fresh_input::<N>(1))?;
    let capture = alloc::AllocationCapture::start();
    resident_call(&resident_input, &mut resident_output)?;
    let resident_allocations = alloc::AllocationCapture::finish();
    println!(
        "Prepared resident PCU warm call Rust heap: {} alloc/realloc calls, {} requested B; this is a benchmark-thread Rust allocator count, not a device/driver allocation count",
        resident_allocations.alloc_calls + resident_allocations.realloc_calls,
        resident_allocations.requested_bytes
    );
    drop(capture);

    sample_resident::<N, _>(
        criterion,
        &mut resident_call,
        backend,
        pool,
        &mut resident_input,
        &mut resident_output,
        &mut native_resident_input,
        &native_resident_output,
        &function,
        &stream,
        &mut native_result,
        &mut cpu_output,
        grid,
    )?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn preflight_host<const N: usize, E: Error + 'static>(
    call: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    output: &mut [f32],
    oracle: &mut [f32],
) -> Result<(), Box<dyn Error>> {
    let input = fresh_input::<N>(0);
    fill_oracle(&input, oracle);
    call(&input, output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    verify("host preflight", output, oracle)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(unsafe_code)]
// These borrow-only fixtures are clearer separately than nested in one mutable context object.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::significant_drop_tightening)] // `finish` is required to emit Criterion's group.
fn sample_host<const N: usize, E: Error + 'static>(
    criterion: &mut Criterion,
    call: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    input: &mut Vec<f32>,
    output: &mut [f32],
    oracle: &mut [f32],
    initial: &[f32],
    native_input: &mut DeviceBuffer,
    native_output: &mut DeviceBuffer,
    function: &HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group(format!("typed-host-{N}"));
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for pcu_target in [true, false] {
        let label = if pcu_target {
            "Prepared typed PCU"
        } else {
            "Native HIP full boundary"
        };
        let mut sequence = 0_u64;
        group.bench_function(BenchmarkId::new(label, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let job = sequence;
                    sequence = sequence.saturating_add(1);
                    *input = fresh_input::<N>(job);
                    fill_oracle(input, oracle);
                    // Both routes begin from the same initialized mutable host output.
                    output.copy_from_slice(initial);
                    let started = Instant::now();
                    if pcu_target {
                        call(input, &mut *output).expect("typed host call");
                    } else {
                        let input_view = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
                        let output_view = PcuHostArgument::read(PcuBindingRef::new(0, 1), output);
                        native_input
                            .copy_from(input_view.bytes())
                            .expect("native input upload");
                        native_output
                            .copy_from(output_view.bytes())
                            .expect("native mutable-output upload");
                        let arguments = [
                            HipKernelArgument::Buffer(native_input),
                            HipKernelArgument::Buffer(native_output),
                        ];
                        // SAFETY: The handwritten HIP kernel has the same two-buffer ABI and extent.
                        let mut completion = unsafe {
                            function.launch(stream, [grid, 1, 1], [256, 1, 1], 0, &arguments)
                        }
                        .expect("native HIP launch");
                        completion.wait().expect("native HIP completion");
                        let mut native_bytes =
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut *output);
                        native_output
                            .copy_to(native_bytes.bytes_mut().expect("native mutable bytes"))
                            .expect("native output download");
                    }
                    elapsed += started.elapsed();
                    assert!(
                        verify(label, output, oracle).is_ok(),
                        "{label} output differs from CPU oracle at job {job}"
                    );
                    black_box(&*output);
                }
                elapsed
            });
        });
    }
    group.finish();
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(unsafe_code)]
// The sample owns one reusable mutable buffer set; splitting it obscures the paired work boundary.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::significant_drop_tightening)] // `finish` is required to emit Criterion's group.
fn sample_resident<const N: usize, E: Error + 'static>(
    criterion: &mut Criterion,
    call: &mut impl FnMut(
        &PcuDeviceBuffer<f32, RocmMemoryResource>,
        &mut PcuDeviceBuffer<f32, RocmMemoryResource>,
    ) -> Result<(), E>,
    backend: &RocmOwnedDispatchBackend,
    pool: PcuMemoryPoolId,
    input: &mut PcuDeviceBuffer<f32, RocmMemoryResource>,
    output: &mut PcuDeviceBuffer<f32, RocmMemoryResource>,
    native_input: &mut DeviceBuffer,
    native_output: &DeviceBuffer,
    function: &HipKernel,
    stream: &HipStreamHandle,
    native_result: &mut [f32],
    oracle: &mut [f32],
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let mut group = criterion.benchmark_group(format!("typed-resident-{N}"));
    group.throughput(Throughput::Elements(u64::try_from(N)?));
    for pcu_target in [true, false] {
        let label = if pcu_target {
            "Prepared typed PCU"
        } else {
            "Native HIP"
        };
        let mut sequence = 0_u64;
        group.bench_function(BenchmarkId::new(label, N), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    let job = sequence;
                    sequence = sequence.saturating_add(1);
                    let input_values = fresh_input::<N>(job);
                    fill_oracle(&input_values, oracle);
                    if pcu_target {
                        backend
                            .refresh_buffer(pool, input, &input_values)
                            .expect("resident PCU input refresh");
                        let started = Instant::now();
                        call(input, output).expect("resident PCU call");
                        elapsed += started.elapsed();
                        backend
                            .download_buffer(pool, output, native_result)
                            .expect("resident PCU result download");
                    } else {
                        let input_view =
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), &input_values);
                        native_input
                            .copy_from(input_view.bytes())
                            .expect("resident HIP input upload");
                        let arguments = [
                            HipKernelArgument::Buffer(native_input),
                            HipKernelArgument::Buffer(native_output),
                        ];
                        // SAFETY: The native kernel and resident PCU kernel have the same extent,
                        // bindings, and geometry; uploads and readback stay outside this timer.
                        let started = Instant::now();
                        let mut completion = unsafe {
                            function.launch(stream, [grid, 1, 1], [256, 1, 1], 0, &arguments)
                        }
                        .expect("resident HIP launch");
                        completion.wait().expect("resident HIP completion");
                        elapsed += started.elapsed();
                        let mut result = PcuHostArgument::read_write(
                            PcuBindingRef::new(0, 0),
                            &mut *native_result,
                        );
                        native_output
                            .copy_to(result.bytes_mut().expect("mutable native result bytes"))
                            .expect("resident HIP result download");
                    }
                    verify(label, native_result, oracle)
                        .unwrap_or_else(|_| panic!("{label} differs from CPU oracle at job {job}"));
                    black_box(&*native_result);
                }
                elapsed
            });
        });
    }
    group.finish();
    Ok(())
}

fn fresh_input<const N: usize>(iteration: u64) -> Vec<f32> {
    let phase_bits = 0x3f00_0000 | (u32::try_from(iteration).unwrap_or(u32::MAX) & 0x007f_ffff);
    let phase = f32::from_bits(phase_bits);
    (0..N)
        .map(|index| {
            let base = f32::from(u16::try_from(index % 1024).expect("sample fits")) * 0.25;
            base + phase
        })
        .collect()
}

fn fill_oracle(input: &[f32], output: &mut [f32]) {
    for (destination, source) in output.iter_mut().zip(input) {
        *destination = source * 2.0 + 1.0;
    }
}

fn verify(label: &str, actual: &[f32], expected: &[f32]) -> Result<(), Box<dyn Error>> {
    if actual.len() != expected.len()
        || actual
            .iter()
            .zip(expected)
            .any(|(actual, expected)| actual.to_bits() != expected.to_bits())
    {
        return Err(format!("{label} output differs from CPU oracle").into());
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(unsafe_code)]
// This one-off diagnostic deliberately compares the same two host-call boundaries.
#[allow(clippy::too_many_arguments)]
fn allocation_diagnostic<const N: usize, E: Error + 'static>(
    call: &mut impl FnMut(&[f32], &mut [f32]) -> Result<(), E>,
    pcu_output: &mut [f32],
    initial: &[f32],
    native_input: &mut DeviceBuffer,
    native_output: &mut DeviceBuffer,
    native_result: &mut [f32],
    function: &HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let input = fresh_input::<N>(u64::MAX - 8);
    let mut expected = vec![0.0_f32; N];
    fill_oracle(&input, &mut expected);
    pcu_output.copy_from_slice(initial);
    let capture = alloc::AllocationCapture::start();
    call(&input, pcu_output).map_err(|error| Box::new(error) as Box<dyn Error>)?;
    let pcu_counts = alloc::AllocationCapture::finish();
    drop(capture);
    verify("PCU heap census", pcu_output, &expected)?;

    let input_view = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
    let output_view = PcuHostArgument::read(PcuBindingRef::new(0, 1), initial);
    let capture = alloc::AllocationCapture::start();
    native_input.copy_from(input_view.bytes())?;
    native_output.copy_from(output_view.bytes())?;
    let arguments = [
        HipKernelArgument::Buffer(native_input),
        HipKernelArgument::Buffer(native_output),
    ];
    // SAFETY: The native kernel ABI and buffer extents match the PCU specialization.
    let mut completion =
        unsafe { function.launch(stream, [grid, 1, 1], [256, 1, 1], 0, &arguments) }?;
    completion.wait()?;
    let mut result = PcuHostArgument::read_write(PcuBindingRef::new(0, 0), native_result);
    native_output.copy_to(result.bytes_mut().expect("mutable native result bytes"))?;
    let native_counts = alloc::AllocationCapture::finish();
    drop(capture);
    verify("native HIP heap census", native_result, &expected)?;
    println!(
        "Warm host-call Rust heap census (outside Criterion): PCU {} alloc/realloc calls, {} requested B; HIP {} calls, {} requested B. HIP/driver allocations are not visible to this Rust allocator counter.",
        pcu_counts.alloc_calls + pcu_counts.realloc_calls,
        pcu_counts.requested_bytes,
        native_counts.alloc_calls + native_counts.realloc_calls,
        native_counts.requested_bytes,
    );
    Ok(())
}
