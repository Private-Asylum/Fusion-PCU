//! Checked u32 quotient/remainder correctness and native HIP comparison.

#[path = "support/dispatch.rs"]
#[allow(dead_code)]
mod dispatch_support;
#[allow(dead_code)] // Selection support exposes utilities shared by the other example benches.
mod support;

use std::{
    error::Error,
    num::NonZeroU32,
};

use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
use fusion_pcu::{
    model::dispatch::{
        PcuDispatchKernelIr,
    },
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingType,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuOwnedBinding,
    PcuOwnedSubmission,
    PcuSubmissionWaitError,
    PcuValueType,
};
use fusion_pcu_macros::{
    pcu,
    pcu_dispatch,
};
use fusion_pcu_rocm::{
    compile_hip_source,
    HipKernelArgument,
    lower_dispatch_to_hip_source,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
};

use dispatch_support::{
    AllocationCapture,
    encode_u32,
    run_direct,
    BLOCK_SIZE,
};

#[pcu(invocations = N)]
fn checked_u32_div_rem_direct<const N: usize>(
    lhs: &[u32],
    rhs: &[u32],
    quotient: &mut [u32],
    remainder: &mut [u32],
) {
    let id = context.global_invocation_id;
    let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
    quotient[id] = q;
    remainder[id] = r;
}

#[pcu_dispatch(invocations = 17)]
fn checked_u32_div_rem_grid<const EXTENT: usize>(
    lhs: &[u32],
    rhs: &[u32],
    quotient: &mut [u32],
    remainder: &mut [u32],
) {
    let mut id = context.global_invocation_id;
    let stride = context.invocation_count;
    while id < EXTENT {
        let (q, r) = pcu::checked_div_rem(lhs[id], rhs[id]);
        quotient[id] = q;
        remainder[id] = r;
        id += stride;
    }
}

enum CheckedDivRemBuilder<'a> {
    Direct(fusion_pcu::model::PcuDispatchKernelBuilder<'a, 6>),
    Grid(fusion_pcu::model::PcuDispatchKernelBuilder<'a, 2>),
}

impl CheckedDivRemBuilder<'_> {
    fn ir(&self) -> PcuDispatchKernelIr<'_> {
        match self {
            Self::Direct(builder) => builder.ir(),
            Self::Grid(builder) => builder.ir(),
        }
    }
}

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("checked u32 DivRem benchmark setup failed");
}

fn run(criterion: &mut Criterion) -> Result<(), Box<dyn Error>> {
    let discovery = RocmDiscovery::new();
    let candidates = support::selected_candidates(&discovery)?
        .into_iter()
        .filter(|candidate| candidate.architecture.is_some())
        .collect::<Vec<_>>();
    let (backend, selected) = support::selection::open_ranked(&discovery, candidates, BLOCK_SIZE)?;
    let architecture = selected
        .architecture
        .ok_or("device has no HIP architecture")?;
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "checked u32 DivRem device: {} ({architecture}); PCU payload buffers are reused; each timed PCU sample includes backend fault-word allocation/init, launch, wait, and status readback; native includes the matching HIP allocation/init/launch/wait/readback",
        discovery.device_info(selected.device)?.name
    );

    // Correctness preflight covers direct and grid-stride dispatches at both requested extents.
    run_case::<65>(criterion, &backend, &runtime, &architecture, 65, false)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &runtime, &architecture, 1 << 20, false)?;
    run_case::<65>(criterion, &backend, &runtime, &architecture, 17, true)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &runtime, &architecture, 17, true)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn run_case<const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    runtime: &fusion_pcu_rocm::HipRuntime,
    architecture: &str,
    invocations: u32,
    grid_stride: bool,
) -> Result<(), Box<dyn Error>> {
    let left = (0..N)
        .map(|i| {
            u32::try_from(i)
                .unwrap_or(u32::MAX)
                .wrapping_mul(19)
                .wrapping_add(7)
        })
        .collect::<Vec<_>>();
    let right = (0..N)
        .map(|i| u32::try_from(i % 31 + 1).expect("divisor fits"))
        .collect::<Vec<_>>();
    let expected_q = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a / b)
        .collect::<Vec<_>>();
    let expected_r = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a % b)
        .collect::<Vec<_>>();
    let lhs_bytes = encode_u32(&left);
    let rhs_bytes = encode_u32(&right);
    let mut pcu_lhs = backend.allocate(lhs_bytes.len())?;
    let mut pcu_rhs = backend.allocate(rhs_bytes.len())?;
    let pcu_q = backend.allocate(lhs_bytes.len())?;
    let pcu_r = backend.allocate(lhs_bytes.len())?;
    pcu_lhs.copy_from(&lhs_bytes)?;
    pcu_rhs.copy_from(&rhs_bytes)?;
    let mut hip_lhs = runtime.allocate(lhs_bytes.len())?;
    let mut hip_rhs = runtime.allocate(rhs_bytes.len())?;
    let hip_q = runtime.allocate(lhs_bytes.len())?;
    let hip_r = runtime.allocate(lhs_bytes.len())?;
    hip_lhs.copy_from(&lhs_bytes)?;
    hip_rhs.copy_from(&rhs_bytes)?;

    let bindings = if grid_stride {
        checked_u32_div_rem_grid_bindings()
    } else {
        checked_u32_div_rem_direct_bindings()
    };
    let builder = if grid_stride {
        CheckedDivRemBuilder::Grid(checked_u32_div_rem_grid::<N>(&bindings)?)
    } else {
        CheckedDivRemBuilder::Direct(checked_u32_div_rem_direct::<N>(&bindings)?)
    };
    let kernel = builder.ir();
    let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(invocations).expect("nonzero")),
    })?;
    let lowered_source = lower_dispatch_to_hip_source(&kernel)?;
    let lowered_image = compile_hip_source(&lowered_source, architecture)?;
    let lowered_module = runtime.load_module(&lowered_image)?;
    let lowered_function = lowered_module.function(c"fusion_kernel")?;
    let native_source = native_source(u32::try_from(N)?, invocations, grid_stride);
    let image = compile_hip_source(&native_source, architecture)?;
    let module = runtime.load_module(&image)?;
    let function = module.function(c"native_checked_u32_div_rem")?;
    let stream = runtime.create_stream()?;
    let native_args = [
        HipKernelArgument::Buffer(&hip_lhs),
        HipKernelArgument::Buffer(&hip_rhs),
        HipKernelArgument::Buffer(&hip_q),
        HipKernelArgument::Buffer(&hip_r),
    ];
    let grid = invocations.div_ceil(BLOCK_SIZE);
    let bindings_owned = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            pcu_lhs.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            pcu_rhs.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            pcu_q.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            pcu_r.clone(),
        )?,
    ];
    run_pcu(backend, &prepared, &bindings_owned)?;
    verify_pair(&pcu_q, &pcu_r, &expected_q, &expected_r)?;
    run_native(runtime, &function, &stream, &native_args, grid)?;
    verify_pair(&hip_q, &hip_r, &expected_q, &expected_r)?;
    run_native(runtime, &lowered_function, &stream, &native_args, grid)?;
    verify_pair(&hip_q, &hip_r, &expected_q, &expected_r)?;
    println!(
        "checked u32 DivRem verified extent={N}, invocations={invocations}, grid_stride={grid_stride}"
    );

    if !grid_stride {
        report_stages(
            backend,
            runtime,
            &prepared,
            &bindings_owned,
            &function,
            &lowered_function,
            &stream,
            &native_args,
            grid,
            N,
        )?;
        if N == 65 {
            verify_fault(
                backend,
                runtime,
                &prepared,
                &bindings_owned,
                &function,
                &lowered_function,
                &stream,
                &native_args,
                grid,
                N,
                7,
            )?;
        }
        let mut group = criterion.benchmark_group("u32-checked-div-rem-direct");
        group.throughput(Throughput::Elements(N as u64));
        for route in 0..3 {
            let label = ["Prepared PCU", "Handwritten HIP", "Lowered HIP direct"][route];
            group.bench_function(BenchmarkId::new(label, N), |bencher| {
                bencher.iter_custom(|iterations| {
                    let started = std::time::Instant::now();
                    for _ in 0..iterations {
                        match route {
                            0 => run_pcu(backend, &prepared, &bindings_owned).expect("PCU launch"),
                            1 => run_native(runtime, &function, &stream, &native_args, grid)
                                .expect("handwritten HIP launch"),
                            _ => {
                                run_native(runtime, &lowered_function, &stream, &native_args, grid)
                                    .expect("lowered HIP launch");
                            }
                        }
                    }
                    started.elapsed()
                });
            });
        }
        group.finish();
    }
    if grid_stride && invocations == 17 {
        let first_fault_id = if N >= 90 { 72 } else { 24 };
        verify_fault(
            backend,
            runtime,
            &prepared,
            &bindings_owned,
            &function,
            &lowered_function,
            &stream,
            &native_args,
            grid,
            N,
            first_fault_id,
        )?;
    }
    Ok(())
}

fn run_pcu(
    backend: &RocmOwnedDispatchBackend,
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
) -> Result<(), Box<dyn Error>> {
    let completion = prepared.submit(bindings)?;
    let mut queued = PcuOwnedSubmission::new(completion, ());
    queued.wait_result().map_err(|error| match error {
        PcuSubmissionWaitError::Backend(error) => Box::<dyn Error>::from(error),
        PcuSubmissionWaitError::Failed => "PCU dispatch failed".into(),
        PcuSubmissionWaitError::Fault(fault) => format!("unexpected fault {fault:?}").into(),
    })?;
    let _ = backend;
    Ok(())
}

fn run_native(
    runtime: &fusion_pcu_rocm::HipRuntime,
    function: &fusion_pcu_rocm::HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    args: &[HipKernelArgument<'_>],
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let mut status = runtime.allocate(8)?;
    status.copy_from(&u64::MAX.to_le_bytes())?;
    let [
        HipKernelArgument::Buffer(lhs),
        HipKernelArgument::Buffer(rhs),
        HipKernelArgument::Buffer(quotient),
        HipKernelArgument::Buffer(remainder),
    ] = args
    else {
        return Err("native argument shape mismatch".into());
    };
    let full_args = [
        HipKernelArgument::Buffer(lhs),
        HipKernelArgument::Buffer(rhs),
        HipKernelArgument::Buffer(quotient),
        HipKernelArgument::Buffer(remainder),
        HipKernelArgument::Buffer(&status),
    ];
    let _sample = run_direct(function, stream, &full_args, grid)?;
    let mut bytes = [0_u8; 8];
    status.copy_to(&mut bytes)?;
    if u64::from_le_bytes(bytes) != u64::MAX {
        return Err("native HIP unexpected fault status".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Stage probes share the prepared kernel and resources.
fn report_stages(
    backend: &RocmOwnedDispatchBackend,
    runtime: &fusion_pcu_rocm::HipRuntime,
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    function: &fusion_pcu_rocm::HipKernel,
    lowered_function: &fusion_pcu_rocm::HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    arguments: &[HipKernelArgument<'_>],
    grid: u32,
    elements: usize,
) -> Result<(), Box<dyn Error>> {
    const SAMPLES: usize = 32;
    let mut pcu = std::array::from_fn::<_, 3, _>(|_| StageSeries::with_capacity(SAMPLES));
    let mut hip = std::array::from_fn::<_, 5, _>(|_| StageSeries::with_capacity(SAMPLES));
    let mut lowered = std::array::from_fn::<_, 5, _>(|_| StageSeries::with_capacity(SAMPLES));
    for sample in 0..SAMPLES {
        match sample % 3 {
            0 => {
                record_pcu_stages(backend, prepared, bindings, &mut pcu)?;
                record_hip_stages(runtime, function, stream, arguments, grid, &mut hip)?;
                record_hip_stages(
                    runtime,
                    lowered_function,
                    stream,
                    arguments,
                    grid,
                    &mut lowered,
                )?;
            }
            1 => {
                record_hip_stages(
                    runtime,
                    lowered_function,
                    stream,
                    arguments,
                    grid,
                    &mut lowered,
                )?;
                record_pcu_stages(backend, prepared, bindings, &mut pcu)?;
                record_hip_stages(runtime, function, stream, arguments, grid, &mut hip)?;
            }
            _ => {
                record_hip_stages(runtime, function, stream, arguments, grid, &mut hip)?;
                record_hip_stages(
                    runtime,
                    lowered_function,
                    stream,
                    arguments,
                    grid,
                    &mut lowered,
                )?;
                record_pcu_stages(backend, prepared, bindings, &mut pcu)?;
            }
        }
    }
    for (label, stage) in ["bind", "submit", "completion"].into_iter().zip(&pcu) {
        stage.print(&format!("PCU {elements} {label}"));
    }
    for (label, stage) in [
        "fault alloc",
        "sentinel init",
        "launch",
        "event wait",
        "status readback",
    ]
    .into_iter()
    .zip(&hip)
    {
        stage.print(&format!("HIP {elements} {label}"));
    }
    for (label, stage) in [
        "fault alloc",
        "sentinel init",
        "launch",
        "event wait",
        "status readback",
    ]
    .into_iter()
    .zip(&lowered)
    {
        stage.print(&format!("Lowered HIP {elements} {label}"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct StageObservation {
    elapsed: std::time::Duration,
    allocations: dispatch_support::AllocationCounts,
}

struct StageSeries(Vec<StageObservation>);

impl StageSeries {
    fn with_capacity(samples: usize) -> Self {
        Self(Vec::with_capacity(samples))
    }
    fn push(
        &mut self,
        elapsed: std::time::Duration,
        allocations: dispatch_support::AllocationCounts,
    ) {
        self.0.push(StageObservation {
            elapsed,
            allocations,
        });
    }
    fn print(&self, label: &str) {
        let mut times = self
            .0
            .iter()
            .map(|sample| sample.elapsed)
            .collect::<Vec<_>>();
        times.sort_unstable();
        let p50 = times[percentile_index(times.len(), 50)];
        let p95 = times[percentile_index(times.len(), 95)];
        let mut allocations = self
            .0
            .iter()
            .map(|sample| sample.allocations)
            .collect::<Vec<_>>();
        allocations.sort_unstable_by_key(|counts| {
            (
                counts.alloc_calls + counts.realloc_calls,
                counts.requested_bytes,
            )
        });
        let show_heap = |index| heap_text(allocations[index]);
        println!(
            "{label}: p50={:.3} us p95={:.3} us; Rust heap p50={} p95={} ({} samples; excludes HIP/driver allocations)",
            micros(p50),
            micros(p95),
            show_heap(percentile_index(allocations.len(), 50)),
            show_heap(percentile_index(allocations.len(), 95)),
            times.len()
        );
    }
}

fn record_pcu_stages(
    backend: &RocmOwnedDispatchBackend,
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    stages: &mut [StageSeries; 3],
) -> Result<(), Box<dyn Error>> {
    let _capture = AllocationCapture::start();
    let started = std::time::Instant::now();
    let pcu_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[0].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[1].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[2].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[3].resource.clone(),
        )?,
    ];
    let binding_time = started.elapsed();
    let after_binding = AllocationCapture::snapshot();
    let started = std::time::Instant::now();
    let completion = prepared.submit(&pcu_bindings)?;
    let submit_time = started.elapsed();
    let after_submit = AllocationCapture::snapshot();
    let started = std::time::Instant::now();
    let mut queued = PcuOwnedSubmission::new(completion, ());
    queued
        .wait_result()
        .map_err(|error| format!("PCU stage probe failed: {error:?}"))?;
    let wait_time = started.elapsed();
    let total_allocations = AllocationCapture::finish();
    stages[0].push(binding_time, after_binding);
    stages[1].push(submit_time, after_submit.since(after_binding));
    stages[2].push(wait_time, total_allocations.since(after_submit));
    Ok(())
}

fn record_hip_stages(
    runtime: &fusion_pcu_rocm::HipRuntime,
    function: &fusion_pcu_rocm::HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    arguments: &[HipKernelArgument<'_>],
    grid: u32,
    stages: &mut [StageSeries; 5],
) -> Result<(), Box<dyn Error>> {
    let _capture = AllocationCapture::start();
    let started = std::time::Instant::now();
    let mut status = runtime.allocate(8)?;
    let allocation_time = started.elapsed();
    let after_allocation = AllocationCapture::snapshot();
    let started = std::time::Instant::now();
    status.copy_from(&u64::MAX.to_le_bytes())?;
    let init_time = started.elapsed();
    let after_init = AllocationCapture::snapshot();
    let [
        HipKernelArgument::Buffer(lhs),
        HipKernelArgument::Buffer(rhs),
        HipKernelArgument::Buffer(quotient),
        HipKernelArgument::Buffer(remainder),
    ] = arguments
    else {
        return Err("native argument shape mismatch".into());
    };
    let full_args = [
        HipKernelArgument::Buffer(lhs),
        HipKernelArgument::Buffer(rhs),
        HipKernelArgument::Buffer(quotient),
        HipKernelArgument::Buffer(remainder),
        HipKernelArgument::Buffer(&status),
    ];
    let started = std::time::Instant::now();
    let mut completion =
        dispatch_support::direct_launch_unwaited(function, stream, &full_args, grid)?;
    let launch_time = started.elapsed();
    let after_launch = AllocationCapture::snapshot();
    let started = std::time::Instant::now();
    completion.wait()?;
    let wait_time = started.elapsed();
    let after_wait = AllocationCapture::snapshot();
    let started = std::time::Instant::now();
    let mut word = [0_u8; 8];
    status.copy_to(&mut word)?;
    if u64::from_le_bytes(word) != u64::MAX {
        return Err("native stage probe observed unexpected fault".into());
    }
    let readback_time = started.elapsed();
    let total_allocations = AllocationCapture::finish();
    stages[0].push(allocation_time, after_allocation);
    stages[1].push(init_time, after_init.since(after_allocation));
    stages[2].push(launch_time, after_launch.since(after_init));
    stages[3].push(wait_time, after_wait.since(after_launch));
    stages[4].push(readback_time, total_allocations.since(after_wait));
    Ok(())
}

fn micros(value: std::time::Duration) -> f64 {
    value.as_secs_f64() * 1_000_000.0
}

const fn percentile_index(len: usize, percentile: usize) -> usize {
    len.saturating_mul(percentile)
        .div_ceil(100)
        .saturating_sub(1)
}

fn heap_text(counts: dispatch_support::AllocationCounts) -> String {
    format!(
        "{} calls/{} B",
        counts.alloc_calls + counts.realloc_calls,
        counts.requested_bytes
    )
}

fn verify_pair(
    quotient: &fusion_pcu_rocm::DeviceBuffer,
    remainder: &fusion_pcu_rocm::DeviceBuffer,
    expected_q: &[u32],
    expected_r: &[u32],
) -> Result<(), Box<dyn Error>> {
    let mut q = vec![0; expected_q.len() * 4];
    let mut r = vec![0; expected_r.len() * 4];
    quotient.copy_to(&mut q)?;
    remainder.copy_to(&mut r)?;
    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|x| u32::from_le_bytes(x.try_into().expect("four bytes")))
            .collect::<Vec<_>>()
    };
    if decode(&q) != expected_q || decode(&r) != expected_r {
        return Err("checked DivRem output mismatch".into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Keep the paired native and PCU fault proof explicit.
fn verify_fault(
    backend: &RocmOwnedDispatchBackend,
    runtime: &fusion_pcu_rocm::HipRuntime,
    prepared: &fusion_pcu_rocm::RocmPreparedDispatch,
    bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    function: &fusion_pcu_rocm::HipKernel,
    lowered_function: &fusion_pcu_rocm::HipKernel,
    stream: &fusion_pcu_rocm::HipStreamHandle,
    _args: &[HipKernelArgument<'_>],
    grid: u32,
    n: usize,
    expected_id: u64,
) -> Result<(), Box<dyn Error>> {
    let mut divisors = vec![1_u32; n];
    divisors[usize::try_from(expected_id)?] = 0;
    divisors[usize::try_from(expected_id + 17)?] = 0;
    let mut pcu_rhs = backend.allocate(n * 4)?;
    pcu_rhs.copy_from(&encode_u32(&divisors))?;
    let fault_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[0].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            pcu_rhs,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[2].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            bindings[3].resource.clone(),
        )?,
    ];
    let mut queued = PcuOwnedSubmission::new(prepared.submit(&fault_bindings)?, ());
    match queued.wait_result() {
        Err(PcuSubmissionWaitError::Fault(fault))
            if fault.kind == fusion_pcu::PcuExecutionFaultKind::DivideByZero
                && fault.invocation_id == expected_id => {}
        other => {
            return Err(format!(
                "PCU did not report first zero divisor at ID {expected_id}: {other:?}"
            )
            .into());
        }
    }

    let lhs = runtime.allocate(n * 4)?;
    let mut rhs = runtime.allocate(n * 4)?;
    let quotient = runtime.allocate(n * 4)?;
    let remainder = runtime.allocate(n * 4)?;
    rhs.copy_from(&encode_u32(&divisors))?;
    let mut status = runtime.allocate(8)?;
    status.copy_from(&u64::MAX.to_le_bytes())?;
    let mut word = [0_u8; 8];
    {
        let args = [
            HipKernelArgument::Buffer(&lhs),
            HipKernelArgument::Buffer(&rhs),
            HipKernelArgument::Buffer(&quotient),
            HipKernelArgument::Buffer(&remainder),
            HipKernelArgument::Buffer(&status),
        ];
        let _sample = run_direct(function, stream, &args, grid)?;
        status.copy_to(&mut word)?;
    }
    let expected = (expected_id << 2) | 1;
    if u64::from_le_bytes(word) != expected {
        return Err(format!(
            "native first-fault word was {}, expected {expected}",
            u64::from_le_bytes(word)
        )
        .into());
    }
    status.copy_from(&u64::MAX.to_le_bytes())?;
    let args = [
        HipKernelArgument::Buffer(&lhs),
        HipKernelArgument::Buffer(&rhs),
        HipKernelArgument::Buffer(&quotient),
        HipKernelArgument::Buffer(&remainder),
        HipKernelArgument::Buffer(&status),
    ];
    let _sample = run_direct(lowered_function, stream, &args, grid)?;
    status.copy_to(&mut word)?;
    if u64::from_le_bytes(word) != expected {
        return Err(format!(
            "lowered HIP first-fault word was {}, expected {expected}",
            u64::from_le_bytes(word)
        )
        .into());
    }
    println!(
        "first zero-divisor invocation verified: PCU fault ID {expected_id}; handwritten and lowered HIP packed word {expected}"
    );
    Ok(())
}

fn native_source(extent: u32, invocations: u32, grid_stride: bool) -> String {
    let iteration = if grid_stride {
        format!("for (unsigned int i = base; i < {extent}u; i += {invocations}u) {{")
    } else {
        format!("const unsigned int i = base; if (i < {extent}u) {{")
    };
    let close = "}";
    format!(
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_checked_u32_div_rem(const unsigned int* a,const unsigned int* b,unsigned int* q,unsigned int* r,unsigned long long* fault_word) {{\nunsigned int base=blockIdx.x*blockDim.x+threadIdx.x; if (base >= {invocations}u) return; {iteration} if (b[i] == 0u) {{ atomicMin(fault_word, (static_cast<unsigned long long>(i) << 2u) | 1ull); }} else {{ q[i]=a[i]/b[i]; r[i]=a[i]%b[i]; }} {close} }}\n"
    )
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
