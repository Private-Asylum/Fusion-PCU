//! Checked signed i32 quotient/remainder correctness and native HIP comparison.

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
        PcuDispatchDataOp,
        PcuDispatchEntryPoint,
        PcuDispatchIndex,
        PcuDispatchKernelIr,
        PcuDispatchOp,
        PcuDispatchValueId,
        PcuIntegerDivFlags,
    },
    PcuBinding,
    PcuBindingAccess,
    PcuBindingRef,
    PcuBindingStorageClass,
    PcuBindingType,
    PcuDispatchSubmission,
    PcuInvocationShape,
    PcuKernelId,
    PcuOwnedBinding,
    PcuOwnedSubmission,
    PcuSubmissionWaitError,
    PcuValueType,
    PcuValueTypeCaps,
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
    run_direct,
    BLOCK_SIZE,
};

fn bench(criterion: &mut Criterion) {
    run(criterion).expect("checked i32 DivRem benchmark setup failed");
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
        "checked i32 DivRem device: {} ({architecture}); PCU payload buffers are reused; each timed PCU sample includes backend fault-word allocation/init, launch, wait, and status readback; native includes the matching HIP allocation/init/launch/wait/readback",
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
            if i.is_multiple_of(2) {
                i32::MIN + i32::try_from(i).expect("index fits")
            } else {
                i32::try_from(i).expect("index fits") - 500_000
            }
        })
        .collect::<Vec<_>>();
    let right = (0..N)
        .map(|i| if i.is_multiple_of(2) { -3 } else { 7 })
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
    let lhs_bytes = encode_i32(&left);
    let rhs_bytes = encode_i32(&right);
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

    let (bindings, kernel_ops) = build_ops::<N>(grid_stride);
    let kernel = PcuDispatchKernelIr {
        id: PcuKernelId(0xD1_0001),
        entry: PcuDispatchEntryPoint {
            name: "checked_i32_div_rem",
            logical_shape: [invocations, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops: kernel_ops,
        type_caps: PcuValueTypeCaps::INT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: fusion_pcu::model::dispatch::PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(fusion_pcu::model::dispatch::PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
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
    let function = module.function(c"native_checked_i32_div_rem")?;
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
            PcuBindingType::Value(PcuValueType::i32()),
            pcu_lhs.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            pcu_rhs.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            pcu_q.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            pcu_r.clone(),
        )?,
    ];
    run_pcu(backend, &prepared, &bindings_owned)?;
    verify_pair("PCU", &pcu_q, &pcu_r, &expected_q, &expected_r)?;
    run_native(runtime, &function, &stream, &native_args, grid)?;
    verify_pair("handwritten HIP", &hip_q, &hip_r, &expected_q, &expected_r)?;
    run_native(runtime, &lowered_function, &stream, &native_args, grid)?;
    verify_pair("lowered HIP", &hip_q, &hip_r, &expected_q, &expected_r)?;
    println!(
        "checked i32 DivRem verified extent={N}, invocations={invocations}, grid_stride={grid_stride}"
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
                fusion_pcu::PcuExecutionFaultKind::DivideByZero,
            )?;
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
                11,
                fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow,
            )?;
        }
        let mut group = criterion.benchmark_group("i32-checked-div-rem-direct");
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
            fusion_pcu::PcuExecutionFaultKind::DivideByZero,
        )?;
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
            first_fault_id + 1,
            fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow,
        )?;
    }
    Ok(())
}

const BINDINGS: [PcuBinding<'static>; 4] = [
    PcuBinding::value(
        Some("lhs"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::i32(),
    ),
    PcuBinding::value(
        Some("rhs"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::i32(),
    ),
    PcuBinding::value(
        Some("quotient"),
        0,
        2,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::i32(),
    ),
    PcuBinding::value(
        Some("remainder"),
        0,
        3,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::i32(),
    ),
];

const fn load(result: u16, binding: u32, index: PcuDispatchIndex) -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
        result: PcuDispatchValueId(result),
        binding: PcuBindingRef::new(0, binding),
        index,
    })
}

const fn store(binding: u32, index: PcuDispatchIndex, value: u16) -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
        binding: PcuBindingRef::new(0, binding),
        index,
        value: PcuDispatchValueId(value),
    })
}

const fn checked() -> PcuDispatchOp<'static> {
    PcuDispatchOp::Data(PcuDispatchDataOp::CheckedDivRem {
        value_type: PcuValueType::i32(),
        flags: PcuIntegerDivFlags::CHECKED,
        quotient: PcuDispatchValueId(3),
        remainder: PcuDispatchValueId(4),
        lhs: PcuDispatchValueId(1),
        rhs: PcuDispatchValueId(2),
    })
}

static DIRECT_OPS: [PcuDispatchOp<'static>; 6] = [
    load(1, 0, PcuDispatchIndex::InvocationId),
    load(2, 1, PcuDispatchIndex::InvocationId),
    checked(),
    store(2, PcuDispatchIndex::InvocationId, 3),
    store(3, PcuDispatchIndex::InvocationId, 4),
    PcuDispatchOp::Control(fusion_pcu::model::dispatch::PcuDispatchControlOp::Return),
];

static GRID_BODY: [PcuDispatchOp<'static>; 5] = [
    load(1, 0, PcuDispatchIndex::GridStrideId),
    load(2, 1, PcuDispatchIndex::GridStrideId),
    checked(),
    store(2, PcuDispatchIndex::GridStrideId, 3),
    store(3, PcuDispatchIndex::GridStrideId, 4),
];
static GRID_OPS_65: [PcuDispatchOp<'static>; 2] = [
    PcuDispatchOp::GridStrideLoop {
        extent: 65,
        body: &GRID_BODY,
    },
    PcuDispatchOp::Control(fusion_pcu::model::dispatch::PcuDispatchControlOp::Return),
];
static GRID_OPS_1M: [PcuDispatchOp<'static>; 2] = [
    PcuDispatchOp::GridStrideLoop {
        extent: 1 << 20,
        body: &GRID_BODY,
    },
    PcuDispatchOp::Control(fusion_pcu::model::dispatch::PcuDispatchControlOp::Return),
];

fn build_ops<const N: usize>(
    grid_stride: bool,
) -> (
    &'static [PcuBinding<'static>; 4],
    &'static [PcuDispatchOp<'static>],
) {
    let ops: &'static [PcuDispatchOp<'static>] = if !grid_stride {
        &DIRECT_OPS
    } else if N == 65 {
        &GRID_OPS_65
    } else {
        &GRID_OPS_1M
    };
    (&BINDINGS, ops)
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
            PcuBindingType::Value(PcuValueType::i32()),
            bindings[0].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            bindings[1].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            bindings[2].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
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

fn encode_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn verify_pair(
    label: &str,
    quotient: &fusion_pcu_rocm::DeviceBuffer,
    remainder: &fusion_pcu_rocm::DeviceBuffer,
    expected_q: &[i32],
    expected_r: &[i32],
) -> Result<(), Box<dyn Error>> {
    let mut q = vec![0; expected_q.len() * 4];
    let mut r = vec![0; expected_r.len() * 4];
    quotient.copy_to(&mut q)?;
    remainder.copy_to(&mut r)?;
    let decode = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|x| i32::from_le_bytes(x.try_into().expect("four bytes")))
            .collect::<Vec<_>>()
    };
    let actual_q = decode(&q);
    let actual_r = decode(&r);
    if let Some((index, (&actual, &expected))) = actual_q
        .iter()
        .zip(expected_q)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        return Err(format!(
            "{label} quotient mismatch at {index}: got {actual}, expected {expected}"
        )
        .into());
    }
    if let Some((index, (&actual, &expected))) = actual_r
        .iter()
        .zip(expected_r)
        .enumerate()
        .find(|(_, (actual, expected))| actual != expected)
    {
        return Err(format!(
            "{label} remainder mismatch at {index}: got {actual}, expected {expected}"
        )
        .into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)] // Keep the paired native and PCU fault proof explicit.
#[allow(clippy::too_many_lines)] // The same fault scenario is checked through all three routes.
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
    fault_kind: fusion_pcu::PcuExecutionFaultKind,
) -> Result<(), Box<dyn Error>> {
    let mut numerators = vec![11_i32; n];
    let mut divisors = vec![7_i32; n];
    let first = usize::try_from(expected_id)?;
    let second = usize::try_from(expected_id + 17)?;
    match fault_kind {
        fusion_pcu::PcuExecutionFaultKind::DivideByZero => {
            divisors[first] = 0;
            divisors[second] = 0;
        }
        fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow => {
            numerators[first] = i32::MIN;
            numerators[second] = i32::MIN;
            divisors[first] = -1;
            divisors[second] = -1;
        }
    }
    let mut pcu_lhs = backend.allocate(n * 4)?;
    let mut pcu_rhs = backend.allocate(n * 4)?;
    pcu_lhs.copy_from(&encode_i32(&numerators))?;
    pcu_rhs.copy_from(&encode_i32(&divisors))?;
    let fault_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            pcu_lhs,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            pcu_rhs,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            bindings[2].resource.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            bindings[3].resource.clone(),
        )?,
    ];
    let mut queued = PcuOwnedSubmission::new(prepared.submit(&fault_bindings)?, ());
    match queued.wait_result() {
        Err(PcuSubmissionWaitError::Fault(fault))
            if fault.kind == fault_kind && fault.invocation_id == expected_id => {}
        other => {
            return Err(format!(
                "PCU did not report first {fault_kind:?} at ID {expected_id}: {other:?}"
            )
            .into());
        }
    }

    let mut lhs = runtime.allocate(n * 4)?;
    let mut rhs = runtime.allocate(n * 4)?;
    let quotient = runtime.allocate(n * 4)?;
    let remainder = runtime.allocate(n * 4)?;
    lhs.copy_from(&encode_i32(&numerators))?;
    rhs.copy_from(&encode_i32(&divisors))?;
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
    let fault_tag = match fault_kind {
        fusion_pcu::PcuExecutionFaultKind::DivideByZero => 1,
        fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow => 2,
    };
    let expected = (expected_id << 2) | fault_tag;
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
        "first {fault_kind:?} invocation verified: PCU fault ID {expected_id}; handwritten and lowered HIP packed word {expected}"
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
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_checked_i32_div_rem(const int* a,const int* b,int* q,int* r,unsigned long long* fault_word) {{\nunsigned int base=blockIdx.x*blockDim.x+threadIdx.x; if (base >= {invocations}u) return; {iteration} if (b[i] == 0) {{ atomicMin(fault_word, (static_cast<unsigned long long>(i) << 2u) | 1ull); }} else if (a[i] == (-2147483647 - 1) && b[i] == -1) {{ atomicMin(fault_word, (static_cast<unsigned long long>(i) << 2u) | 2ull); }} else {{ q[i]=a[i]/b[i]; r[i]=a[i]%b[i]; }} {close} }}\n"
    )
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
