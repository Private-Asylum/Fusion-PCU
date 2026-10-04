//! Consuming selected owners while pruning a differently shaped unused source input.

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
    PcuExecutionError,
    PcuTensor,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    DeviceBuffer,
    HipKernelArgument,
    HipRuntime,
    HipStreamHandle,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    compile_hip_source,
};

const BLOCK: u32 = 256;
const PAIRED_ORDERS: [[usize; 2]; 2] = [[0, 1], [1, 0]];

#[fusion_pcu::pcu]
fn seed(input: &[f32]) -> Result<PcuTensor<f32>, PcuExecutionError> {
    pcu::identity(input)
}

#[fusion_pcu::pcu]
fn add_selected_owners(
    _unused: PcuTensor<f32>,
    lhs: PcuTensor<f32>,
    rhs: PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

#[fusion_pcu::pcu]
fn add_mixed_selected_owners(
    _unused: &PcuTensor<f32>,
    lhs: PcuTensor<f32>,
    rhs: &PcuTensor<f32>,
) -> Result<PcuTensor<f32>, PcuExecutionError> {
    Ok(pcu::add(lhs, rhs)?)
}

struct SampleInputs {
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    oracle: Vec<f32>,
    source: Option<(PcuTensor<f32>, PcuTensor<f32>, PcuTensor<f32>)>,
    native: Option<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
}

impl SampleInputs {
    fn prepare(elements: usize, job: u64, runtime: &HipRuntime) -> Result<Self, Box<dyn Error>> {
        let mut lhs = vec![0.0_f32; elements];
        let mut rhs = vec![0.0_f32; elements];
        fill(&mut lhs, &mut rhs, job);
        let oracle = lhs
            .iter()
            .zip(&rhs)
            .map(|(&left, &right)| left + right)
            .collect();
        let unused_value = f32::from(u16::try_from(job % 32_768)?);
        let unused = [unused_value, -unused_value];

        // All inputs are freshly uploaded outside the measured call. The first owner is
        // deliberately length two while the selected Add inputs use `elements`.
        let source = (seed(&unused)?, seed(&lhs)?, seed(&rhs)?);
        let native = (
            upload(runtime, &unused)?,
            upload(runtime, &lhs)?,
            upload(runtime, &rhs)?,
        );
        Ok(Self {
            lhs,
            rhs,
            oracle,
            source: Some(source),
            native: Some(native),
        })
    }
}

struct MixedInputs {
    lhs: Vec<f32>,
    rhs: Vec<f32>,
    oracle: Vec<f32>,
    unused: [f32; 2],
    source: Option<(PcuTensor<f32>, PcuTensor<f32>, PcuTensor<f32>)>,
    native: Option<(DeviceBuffer, DeviceBuffer, DeviceBuffer)>,
}

impl MixedInputs {
    fn prepare(elements: usize, job: u64, runtime: &HipRuntime) -> Result<Self, Box<dyn Error>> {
        let mut lhs = vec![0.0_f32; elements];
        let mut rhs = vec![0.0_f32; elements];
        fill(&mut lhs, &mut rhs, job);
        let oracle = lhs
            .iter()
            .zip(&rhs)
            .map(|(&left, &right)| left + right)
            .collect();
        let unused_value = f32::from(u16::try_from(job % 32_768)?);
        let unused = [unused_value, -unused_value];
        let source = (seed(&unused)?, seed(&lhs)?, seed(&rhs)?);
        let native = (
            upload(runtime, &unused)?,
            upload(runtime, &lhs)?,
            upload(runtime, &rhs)?,
        );
        Ok(Self {
            lhs,
            rhs,
            oracle,
            unused,
            source: Some(source),
            native: Some(native),
        })
    }
}

fn upload(runtime: &HipRuntime, values: &[f32]) -> Result<DeviceBuffer, Box<dyn Error>> {
    let bytes = values
        .len()
        .checked_mul(size_of::<f32>())
        .ok_or("selected-owner input size overflow")?;
    let mut buffer = runtime.allocate(bytes)?;
    buffer.copy_from(bytemuck::cast_slice(values))?;
    Ok(buffer)
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
    println!("Selected-owner pruning benchmark device: {}", selected.name);
    for elements in [65, 1_048_576] {
        run_case(criterion, &backend, &runtime, architecture, elements)?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)] // Criterion groups own registrations until finish.
#[allow(unsafe_code)] // Native control uses a checked in-place HIP launch over exact extents.
fn run_case(
    criterion: &mut Criterion,
    _backend: &RocmOwnedDispatchBackend,
    runtime: &HipRuntime,
    architecture: &str,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let source = format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_add_inplace(float* lhs, const float* rhs) {{ const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x; if (id < {elements}u) lhs[id] = lhs[id] + rhs[id]; }}\n"
    );
    let image = crate::support::cold_once("selected-owner HIP compilation", || {
        compile_hip_source(&source, architecture)
    })?;
    let module = runtime.load_module(&image)?;
    let kernel = module.function(c"native_add_inplace")?;
    let stream = runtime.create_stream()?;
    let grid = u32::try_from(elements)?.div_ceil(BLOCK);
    let mut observed = vec![0.0_f32; elements];

    for route in 0..2 {
        let mut inputs = SampleInputs::prepare(elements, 0, runtime)?;
        let _ = run_route::<false>(
            route,
            &mut inputs,
            &kernel,
            &stream,
            grid,
            &mut observed,
            None,
        )?;
    }

    let mut counts = [alloc::AllocationCounts::default(); 2];
    for (route, route_counts) in counts.iter_mut().enumerate() {
        let mut inputs = SampleInputs::prepare(elements, 1, runtime)?;
        let mut route_capture = alloc::AllocationCounts::default();
        let _ = run_route::<true>(
            route,
            &mut inputs,
            &kernel,
            &stream,
            grid,
            &mut observed,
            Some(&mut route_capture),
        )?;
        *route_counts = route_capture;
    }
    println!(
        "Selected-owner Rust heap census {elements}: source {}/{}/{} B, native in-place {}/{}/{} B allocations/reallocations/requested. Fresh input setup/readback excluded; output release included.",
        counts[0].alloc_calls,
        counts[0].realloc_calls,
        counts[0].requested_bytes,
        counts[1].alloc_calls,
        counts[1].realloc_calls,
        counts[1].requested_bytes,
    );

    let mut times = [[Duration::ZERO; 24]; 2];
    let mut ratios = [0.0_f64; 24];
    for sample in 0..24 {
        let mut inputs = SampleInputs::prepare(elements, u64::try_from(sample + 2)?, runtime)?;
        for route in PAIRED_ORDERS[sample % PAIRED_ORDERS.len()] {
            times[route][sample] = run_route::<false>(
                route,
                &mut inputs,
                &kernel,
                &stream,
                grid,
                &mut observed,
                None,
            )?;
        }
        ratios[sample] = times[0][sample].as_secs_f64() / times[1][sample].as_secs_f64();
    }
    for route_times in &mut times {
        route_times.sort_unstable();
    }
    ratios.sort_unstable_by(f64::total_cmp);
    println!(
        "Selected-owner paired diagnostic {elements}, 24 balanced verified pairs: source {:?}, native in-place {:?}; paired source/native {:.4}. Diagnostic medians, not Criterion intervals.",
        midpoint(times[0][11], times[0][12]),
        midpoint(times[1][11], times[1][12]),
        midpoint_f64(ratios[11], ratios[12]),
    );

    let mut group = criterion.benchmark_group("owned_selected_owners_add_donor");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    for route in 0..2 {
        let mut job = 30_u64;
        let (name, route_label) = match route {
            0 => (
                "source_three_owner_selected_add",
                "source selected-owner Add",
            ),
            _ => ("native_inplace_add", "native in-place Add"),
        };
        group.bench_function(BenchmarkId::new(name, elements), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    let mut inputs = SampleInputs::prepare(elements, job, runtime)
                        .expect("prepare fresh selected-owner inputs");
                    total += run_route::<false>(
                        route,
                        &mut inputs,
                        &kernel,
                        &stream,
                        grid,
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
    run_mixed_group(criterion, runtime, &kernel, &stream, grid, elements)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
#[allow(clippy::significant_drop_tightening)] // Criterion groups own registrations until finish.
fn run_mixed_group(
    criterion: &mut Criterion,
    runtime: &HipRuntime,
    kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    let mut observed = vec![0.0_f32; elements];
    let mut observed_rhs = vec![0.0_f32; elements];
    for route in 0..2 {
        let mut inputs = MixedInputs::prepare(elements, 0, runtime)?;
        let _ = run_mixed_route::<false>(
            route,
            &mut inputs,
            kernel,
            stream,
            grid,
            &mut observed,
            &mut observed_rhs,
            None,
        )?;
    }

    let mut counts = [alloc::AllocationCounts::default(); 2];
    for (route, route_counts) in counts.iter_mut().enumerate() {
        let mut inputs = MixedInputs::prepare(elements, 1, runtime)?;
        let mut route_capture = alloc::AllocationCounts::default();
        let _ = run_mixed_route::<true>(
            route,
            &mut inputs,
            kernel,
            stream,
            grid,
            &mut observed,
            &mut observed_rhs,
            Some(&mut route_capture),
        )?;
        *route_counts = route_capture;
    }
    println!(
        "Mixed-owner Rust heap census {elements}: source {}/{}/{} B, native in-place {}/{}/{} B allocations/reallocations/requested. Fresh setup/readback excluded; output release included.",
        counts[0].alloc_calls,
        counts[0].realloc_calls,
        counts[0].requested_bytes,
        counts[1].alloc_calls,
        counts[1].realloc_calls,
        counts[1].requested_bytes,
    );

    let mut times = [[Duration::ZERO; 24]; 2];
    let mut ratios = [0.0_f64; 24];
    for sample in 0..24 {
        let mut inputs = MixedInputs::prepare(elements, u64::try_from(sample + 2)?, runtime)?;
        for route in PAIRED_ORDERS[sample % PAIRED_ORDERS.len()] {
            times[route][sample] = run_mixed_route::<false>(
                route,
                &mut inputs,
                kernel,
                stream,
                grid,
                &mut observed,
                &mut observed_rhs,
                None,
            )?;
        }
        ratios[sample] = times[0][sample].as_secs_f64() / times[1][sample].as_secs_f64();
    }
    for route_times in &mut times {
        route_times.sort_unstable();
    }
    ratios.sort_unstable_by(f64::total_cmp);
    println!(
        "Mixed-owner paired diagnostic {elements}, 24 balanced verified pairs: source {:?}, native in-place {:?}; paired source/native {:.4}. Diagnostic medians, not Criterion intervals.",
        midpoint(times[0][11], times[0][12]),
        midpoint(times[1][11], times[1][12]),
        midpoint_f64(ratios[11], ratios[12]),
    );

    let mut group = criterion.benchmark_group("owned_selected_owners_mixed_add_donor");
    group.throughput(Throughput::Elements(u64::try_from(elements)?));
    for route in 0..2 {
        let mut job = 40_u64;
        let (name, route_label) = match route {
            0 => (
                "source_mixed_selected_add",
                "mixed source selected-owner Add",
            ),
            _ => ("native_mixed_inplace_add", "native mixed in-place Add"),
        };
        group.bench_function(BenchmarkId::new(name, elements), |bencher| {
            bencher.iter_custom(|iterations| {
                let mut total = Duration::ZERO;
                for _ in 0..iterations {
                    job = job.wrapping_add(1);
                    let mut inputs = MixedInputs::prepare(elements, job, runtime)
                        .expect("prepare fresh mixed-owner inputs");
                    total += run_mixed_route::<false>(
                        route,
                        &mut inputs,
                        kernel,
                        stream,
                        grid,
                        &mut observed,
                        &mut observed_rhs,
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
#[allow(unsafe_code)] // Native control retains read-only peers and mutates only the donor.
fn run_mixed_route<const TRACK_ALLOCATIONS: bool>(
    route: usize,
    inputs: &mut MixedInputs,
    kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    observed: &mut [f32],
    observed_rhs: &mut [f32],
    mut allocation_counts: Option<&mut alloc::AllocationCounts>,
) -> Result<Duration, Box<dyn Error>> {
    match route {
        0 => {
            let (unused, lhs, rhs) = inputs.source.take().expect("mixed source owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let output = add_mixed_selected_owners(&unused, lhs, &rhs)?;
            let execution = started.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            output.read_into(observed)?;
            rhs.read_into(observed_rhs)?;
            let mut unused_observed = [0.0_f32; 2];
            unused.read_into(&mut unused_observed)?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            verify_unchanged(&inputs.rhs, observed_rhs);
            assert_eq!(unused_observed[0].to_bits(), inputs.unused[0].to_bits());
            assert_eq!(unused_observed[1].to_bits(), inputs.unused[1].to_bits());
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            drop(black_box(output));
            let release = started.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        1 => {
            let (unused, lhs, rhs) = inputs.native.take().expect("mixed native owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let arguments = [
                HipKernelArgument::Buffer(&lhs),
                HipKernelArgument::Buffer(&rhs),
            ];
            // SAFETY: `lhs`/`rhs` have the checked `observed.len()` extent and the kernel only
            // mutates `lhs`; `unused` is intentionally retained as a read-only unselected peer.
            let mut completion =
                unsafe { kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
            completion.wait()?;
            drop(completion);
            let execution = started.elapsed();
            record_capture(execution_capture, allocation_counts.as_deref_mut());
            lhs.copy_to(bytemuck::cast_slice_mut(observed))?;
            rhs.copy_to(bytemuck::cast_slice_mut(observed_rhs))?;
            let mut unused_observed = [0.0_f32; 2];
            unused.copy_to(bytemuck::cast_slice_mut(&mut unused_observed))?;
            verify(&inputs.lhs, &inputs.rhs, &inputs.oracle, observed);
            verify_unchanged(&inputs.rhs, observed_rhs);
            assert_eq!(unused_observed[0].to_bits(), inputs.unused[0].to_bits());
            assert_eq!(unused_observed[1].to_bits(), inputs.unused[1].to_bits());
            let release_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            drop(black_box(lhs));
            let release = started.elapsed();
            record_capture(release_capture, allocation_counts);
            Ok(execution + release)
        }
        _ => unreachable!("two mixed-owner routes are declared"),
    }
}

#[allow(clippy::too_many_arguments)]
#[allow(unsafe_code)] // The native route matches source Add and consumes the same owners.
fn run_route<const TRACK_ALLOCATIONS: bool>(
    route: usize,
    inputs: &mut SampleInputs,
    kernel: &fusion_pcu_rocm::HipKernel,
    stream: &HipStreamHandle,
    grid: u32,
    observed: &mut [f32],
    mut allocation_counts: Option<&mut alloc::AllocationCounts>,
) -> Result<Duration, Box<dyn Error>> {
    match route {
        0 => {
            let (unused, lhs, rhs) = inputs.source.take().expect("source owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let output = add_selected_owners(unused, lhs, rhs)?;
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
        1 => {
            let (unused, lhs, rhs) = inputs.native.take().expect("native owners are fresh");
            let execution_capture = start_capture::<TRACK_ALLOCATIONS>();
            let started = Instant::now();
            let arguments = [
                HipKernelArgument::Buffer(&lhs),
                HipKernelArgument::Buffer(&rhs),
            ];
            // SAFETY: `lhs` and `rhs` contain exactly `observed.len()` f32 elements; the kernel
            // bounds-checks that extent and writes only the left donor allocation.
            let mut completion =
                unsafe { kernel.launch(stream, [grid, 1, 1], [BLOCK, 1, 1], 0, &arguments) }?;
            completion.wait()?;
            drop(completion);
            drop((unused, rhs));
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
        _ => unreachable!("two routes are declared"),
    }
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
    let job_value = f32::from(u16::try_from(job % 32_768).expect("job value fits"));
    for (index, (left, right)) in lhs.iter_mut().zip(rhs).enumerate() {
        let lane = f32::from(u16::try_from(index % 1024).expect("lane fits"));
        *left = lane.mul_add(0.0625, job_value);
        *right = if index.is_multiple_of(2) { 1.0 } else { -1.0 };
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

fn verify_unchanged(expected: &[f32], observed: &[f32]) {
    assert_eq!(expected.len(), observed.len());
    for (&expected, &actual) in expected.iter().zip(observed) {
        assert_eq!(expected.to_bits(), actual.to_bits());
    }
}

fn midpoint(lhs: Duration, rhs: Duration) -> Duration {
    Duration::from_secs_f64(lhs.as_secs_f64().midpoint(rhs.as_secs_f64()))
}

const fn midpoint_f64(lhs: f64, rhs: f64) -> f64 {
    lhs.midpoint(rhs)
}
