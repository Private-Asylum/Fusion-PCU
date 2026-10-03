//! Checked signed i32 quotient/remainder correctness and native HIP comparison.

extern crate pcu_facade as fusion_pcu;

#[path = "../support/dispatch.rs"]
#[allow(dead_code)]
mod dispatch_support;
#[path = "../support/owned_host_upload.rs"]
mod owned_host_upload;
#[path = "../support/owned_roundtrip.rs"]
mod owned_roundtrip;
#[path = "../support/owned_same_handle_sgemm.rs"]
mod owned_same_handle_sgemm;
#[path = "../support/owned_sgemm.rs"]
mod owned_sgemm;
#[allow(dead_code)] // Selection support exposes utilities shared by the other example benches.
#[path = "../support/support.rs"]
mod support;

#[rustfmt::skip]
use std::{
    error::Error,
    num::NonZeroU32,
    sync::Arc,
};

#[rustfmt::skip]
use criterion::{
    criterion_group,
    criterion_main,
    BenchmarkId,
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    model::dispatch::{
        PcuDispatchDataOp,
        PcuDispatchControlOp,
        PcuDispatchEntryPoint,
        PcuDispatchFeatureCaps,
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
    PcuExecutionNodeState,
    PcuExecutionSuccessGate,
    PcuInvocationShape,
    PcuKernelId,
    PcuOwnedBinding,
    PcuOwnedSubmission,
    PcuSubmissionWaitError,
    PcuValueType,
    PcuValueTypeCaps,
};
#[rustfmt::skip]
use fusion_pcu_rocm::{
    compile_hip_source,
    HipKernelArgument,
    lower_dispatch_to_hip_source,
    RocmDiscovery,
    RocmOwnedDispatchBackend,
    RocmExecutionStep,
    RocmOwnedExecution,
    RocmOwnedExecutionNode,
    RocmOwnedExecutionOperation,
    RocmOwnedExecutionTwoSlot,
    RocmOwnedExecutionError,
    RocmTwoSlotExecutionStep,
};

#[rustfmt::skip]
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
    if std::env::var_os("FUSION_ROCM_OWNED_SGEMM_EVENT_FIXTURE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        let device_info = discovery.device_info(selected.device)?;
        let device_name = &device_info.name;
        if !device_name.contains("RX 6900 XT") {
            return Err(format!(
                "owned SGEMM event fixture requires an RX 6900 XT, selected {device_name}"
            )
            .into());
        }
        run_owned_execution_sgemm_event_fixture(&backend)?;
        return Ok(());
    }
    if std::env::var_os("FUSION_ROCM_SAME_HANDLE_SGEMM_FIXTURE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_same_handle_sgemm_queue_fixture(&backend)?;
        return Ok(());
    }
    if std::env::var_os("FUSION_ROCM_HOST_UPLOAD_GRAPH_FIXTURE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_host_upload_graph_fixture(&backend)?;
        return Ok(());
    }
    if std::env::var_os("FUSION_ROCM_DEVICE_READBACK_GRAPH_FIXTURE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_device_readback_graph_fixture(&backend)?;
        return Ok(());
    }
    let runtime = discovery.open_device(selected.device)?;
    println!(
        "checked i32 DivRem device: {} ({architecture}); PCU payload buffers are reused; each timed PCU sample includes backend fault-word allocation/init, launch, wait, and status readback; native includes the matching HIP allocation/init/launch/wait/readback",
        discovery.device_info(selected.device)?.name
    );

    run_owned_execution_gate_preflight(&backend)?;
    if std::env::var_os("FUSION_ROCM_GATE_PREFLIGHT_ONLY").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        println!("ROCm owned execution gate preflight only; benchmark timings skipped");
        return Ok(());
    }

    // Correctness preflight covers direct and grid-stride dispatches at both requested extents.
    run_case::<65>(criterion, &backend, &runtime, &architecture, 65, false)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &runtime, &architecture, 1 << 20, false)?;
    run_case::<65>(criterion, &backend, &runtime, &architecture, 17, true)?;
    run_case::<{ 1 << 20 }>(criterion, &backend, &runtime, &architecture, 17, true)?;
    owned_sgemm::benchmark(criterion, &backend)?;
    benchmark_owned_execution_dependency_chain(criterion, &backend)?;
    owned_host_upload::benchmark(criterion, &backend)?;
    owned_roundtrip::benchmark(criterion, &backend)?;
    owned_same_handle_sgemm::benchmark(criterion, &backend)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn benchmark_owned_execution_dependency_chain(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const N: usize = 1 << 16;
    let expected = (0..N)
        .map(|value| u32::try_from(value).expect("bounded fixture") * 3 + 7)
        .collect::<Vec<_>>();
    let input_bytes = encode_u32(&expected);
    let mut input = backend.allocate(input_bytes.len())?;
    let intermediate = backend.allocate(input_bytes.len())?;
    let output = backend.allocate(input_bytes.len())?;
    input.copy_from(&input_bytes)?;

    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0042),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_bench_dependency_copy",
            logical_shape: [u32::try_from(N).expect("bounded fixture"), 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let shape = PcuInvocationShape::invocations(
        NonZeroU32::new(u32::try_from(N).expect("bounded fixture")).expect("nonzero"),
    );
    // Keep compilation, stream creation, allocations, binding construction, and input upload out
    // of the repeated timed path. Both modes reuse this same graph and captured stream pair.
    let producer = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let consumer = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let producer_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            input,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            intermediate.clone(),
        )?,
    ];
    let consumer_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            intermediate,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            output.clone(),
        )?,
    ];

    // Correctness preflight runs each route once before Criterion starts collecting samples.
    run_owned_dependency_chain_once(
        &producer,
        &producer_bindings,
        &consumer,
        &consumer_bindings,
        false,
    )?;
    verify_owned_dependency_chain_output(&output, &input_bytes, &expected, "host-wait")?;
    run_owned_dependency_chain_once(
        &producer,
        &producer_bindings,
        &consumer,
        &consumer_bindings,
        true,
    )?;
    verify_owned_dependency_chain_output(&output, &input_bytes, &expected, "event-chain")?;

    if std::env::var_os("FUSION_ROCM_CHAIN_PAIRED_DIAGNOSTIC").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_owned_dependency_chain_paired_diagnostic(
            &producer,
            &producer_bindings,
            &consumer,
            &consumer_bindings,
            &output,
            &input_bytes,
            &expected,
        )?;
    }
    if std::env::var_os("FUSION_ROCM_CHAIN_PHASE_PROFILE").as_deref()
        == Some(std::ffi::OsStr::new("1"))
    {
        run_owned_dependency_chain_phase_profile(
            &producer,
            &producer_bindings,
            &consumer,
            &consumer_bindings,
            &output,
            &input_bytes,
            &expected,
        )?;
    }

    {
        let mut group = criterion.benchmark_group("rocm-owned-two-copy-dependency-chain");
        group.throughput(Throughput::Elements((N * 2) as u64));
        let routes = if std::env::var_os("FUSION_ROCM_CHAIN_BENCH_REVERSE_ORDER").as_deref()
            == Some(std::ffi::OsStr::new("1"))
        {
            [("event chain", true), ("two-slot host wait", false)]
        } else {
            [("two-slot host wait", false), ("event chain", true)]
        };
        for (label, event_chaining) in routes {
            group.bench_function(BenchmarkId::new(label, N), |bencher| {
                bencher.iter_custom(|iterations| {
                    let started = std::time::Instant::now();
                    for _ in 0..iterations {
                        run_owned_dependency_chain_once(
                            &producer,
                            &producer_bindings,
                            &consumer,
                            &consumer_bindings,
                            event_chaining,
                        )
                        .expect("owned dependency-chain iteration");
                    }
                    started.elapsed()
                });
            });
        }
        group.finish();
    }
    println!(
        "ROCm owned two-copy dependency chain benchmark preflight passed; dispatch preparation and resident buffers reused"
    );
    Ok(())
}

/// Opt-in RX 6900 XT correctness fixture for dispatch -> SGEMM -> dispatch. Set
/// `FUSION_ROCM_OWNED_SGEMM_EVENT_FIXTURE=1` to run; it is deliberately outside normal sampling.
#[allow(clippy::too_many_lines)]
fn run_owned_execution_sgemm_event_fixture(
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const ELEMENTS: usize = 4;
    const BYTES: usize = ELEMENTS * std::mem::size_of::<f32>();
    let input_values = [1.0_f32, 0.0, 0.0, 1.0];
    let rhs_values = [5.0_f32, 7.0, 11.0, 13.0];
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let input_bytes = encode(&input_values);
    let rhs_bytes = encode(&rhs_values);
    let mut input = backend.allocate(BYTES)?;
    let a = backend.allocate(BYTES)?;
    let mut rhs = backend.allocate(BYTES)?;
    let c = backend.allocate(BYTES)?;
    let output = backend.allocate(BYTES)?;
    input.copy_from(&input_bytes)?;
    rhs.copy_from(&rhs_bytes)?;

    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0051),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_sgemm_copy",
            logical_shape: [u32::try_from(ELEMENTS)?, 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let shape = PcuInvocationShape::invocations(
        NonZeroU32::new(u32::try_from(ELEMENTS)?).ok_or("empty SGEMM fixture")?,
    );
    let copy_in = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let copy_out = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let input_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            input.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            a.clone(),
        )?,
    ];
    let output_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            c.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            output.clone(),
        )?,
    ];
    let sgemm_stream = backend.create_stream()?;
    let mut rocblas = backend.create_rocblas()?;
    rocblas.bind_stream(&sgemm_stream)?;
    let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &copy_in,
                bindings: &input_bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Sgemm {
                handle: &rocblas,
                stream: &sgemm_stream,
                transpose_a: false,
                transpose_b: false,
                m: 2,
                n: 2,
                k: 2,
                alpha: 1.0,
                a: &a,
                lda: 2,
                b: &rhs,
                ldb: 2,
                beta: 0.0,
                c: &c,
                ldc: 2,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[2],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &copy_out,
                bindings: &output_bindings,
            },
        },
    ];
    let mut scratch = [false; 3];
    let mut states = [PcuExecutionNodeState::Pending; 3];
    let mut execution =
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;
    for node in 0..3 {
        if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
            return Err(format!("SGEMM event fixture did not submit node {node}").into());
        }
    }
    if !matches!(
        execution.step()?,
        RocmTwoSlotExecutionStep::Succeeded { node: 2 }
    ) || execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("SGEMM event fixture did not reach terminal success".into());
    }
    if states
        != [
            PcuExecutionNodeState::Succeeded,
            PcuExecutionNodeState::Succeeded,
            PcuExecutionNodeState::Succeeded,
        ]
    {
        return Err(format!("SGEMM event fixture states were {states:?}").into());
    }
    let mut actual = vec![0_u8; BYTES];
    output.copy_to(&mut actual)?;
    if actual != rhs_bytes {
        return Err("dispatch -> SGEMM -> dispatch fixture output mismatch".into());
    }
    println!("ROCm dispatch -> SGEMM -> dispatch event chain verified");
    Ok(())
}

/// Opt-in RX 6900 XT fixture proving dependent SGEMMs and terminal readback share one
/// same-stream batch while the rocBLAS handle remains reserved. Set
/// `FUSION_ROCM_SAME_HANDLE_SGEMM_FIXTURE=1` to run it outside Criterion sampling.
#[allow(clippy::too_many_lines)] // Keep the three-node ownership and result proof in one fixture.
fn run_same_handle_sgemm_queue_fixture(
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const BYTES: usize = 4 * std::mem::size_of::<f32>();
    let identity = [1.0_f32, 0.0, 0.0, 1.0];
    let rhs_values = [5.0_f32, 7.0, 11.0, 13.0];
    let encode = |values: &[f32]| {
        values
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>()
    };
    let identity_bytes = encode(&identity);
    let rhs_bytes = encode(&rhs_values);
    let mut left = backend.allocate(BYTES)?;
    let mut right_identity = backend.allocate(BYTES)?;
    let mut rhs = backend.allocate(BYTES)?;
    let intermediate = backend.allocate(BYTES)?;
    let output = backend.allocate(BYTES)?;
    left.copy_from(&identity_bytes)?;
    right_identity.copy_from(&identity_bytes)?;
    rhs.copy_from(&rhs_bytes)?;

    let stream = backend.create_stream()?;
    let mut handle = backend.create_rocblas()?;
    handle.bind_stream(&stream)?;
    let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Sgemm {
                handle: &handle,
                stream: &stream,
                transpose_a: false,
                transpose_b: false,
                m: 2,
                n: 2,
                k: 2,
                alpha: 1.0,
                a: &left,
                lda: 2,
                b: &rhs,
                ldb: 2,
                beta: 0.0,
                c: &intermediate,
                ldc: 2,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Sgemm {
                handle: &handle,
                stream: &stream,
                transpose_a: false,
                transpose_b: false,
                m: 2,
                n: 2,
                k: 2,
                alpha: 1.0,
                a: &intermediate,
                lda: 2,
                b: &right_identity,
                ldb: 2,
                beta: 0.0,
                c: &output,
                ldc: 2,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[2],
            operation: RocmOwnedExecutionOperation::DeviceReadback {
                stream: &stream,
                source: &output,
                source_offset: 0,
                bytes: BYTES,
            },
        },
    ];
    let mut scratch = [false; 3];
    let mut states = [PcuExecutionNodeState::Pending; 3];
    let mut execution =
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;

    for node in 0..3 {
        if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
            return Err(format!("same-handle SGEMM did not queue node {node}").into());
        }
        if handle.is_usable() {
            return Err("same-handle SGEMM reservation opened before completion".into());
        }
    }
    if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 2 })
        || execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("same-handle queued SGEMM did not complete".into());
    }
    if !handle.is_usable() {
        return Err("same-handle SGEMM reservation remained after completion".into());
    }
    let readback = execution.take_readback(2)?;
    if readback.as_ref() != rhs_bytes {
        return Err("same-handle queued SGEMM readback mismatch".into());
    }
    if states != [PcuExecutionNodeState::Succeeded; 3] {
        return Err(format!("same-handle SGEMM states were {states:?}").into());
    }
    let mut actual = vec![0_u8; BYTES];
    output.copy_to(&mut actual)?;
    if actual != rhs_bytes {
        return Err("same-handle queued SGEMM output mismatch".into());
    }
    println!("ROCm same-handle SGEMM event queue verified");
    Ok(())
}

/// Prove that an owned host payload survives upload -> dispatch event chaining without a host
/// wait between the nodes. This correctness fixture is outside Criterion sampling.
fn run_host_upload_graph_fixture(backend: &RocmOwnedDispatchBackend) -> Result<(), Box<dyn Error>> {
    const ELEMENTS: usize = 65;
    let values = (0..ELEMENTS)
        .map(|index| u32::try_from(index).expect("bounded fixture") * 3 + 7)
        .collect::<Vec<_>>();
    let input_bytes = encode_u32(&values);
    let source = Arc::<[u8]>::from(input_bytes.clone());
    let input = backend.allocate(input_bytes.len())?;
    let output = backend.allocate(input_bytes.len())?;
    let upload_stream = backend.create_stream()?;

    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0052),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_upload_copy",
            logical_shape: [u32::try_from(ELEMENTS)?, 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(u32::try_from(ELEMENTS)?).ok_or("empty upload fixture")?,
        ),
    })?;
    let bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            input.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            output.clone(),
        )?,
    ];
    let dependencies: [&[usize]; 2] = [&[], &[0]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::HostUpload {
                stream: &upload_stream,
                source,
                destination: &input,
                offset: 0,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &prepared,
                bindings: &bindings,
            },
        },
    ];
    let mut scratch = [false; 2];
    let mut states = [PcuExecutionNodeState::Pending; 2];
    let mut execution =
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;
    for node in 0..2 {
        if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
            return Err(format!("upload graph did not submit node {node} before wait").into());
        }
    }
    if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 1 })
        || execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("upload graph did not complete successfully".into());
    }
    if states != [PcuExecutionNodeState::Succeeded; 2] {
        return Err(format!("upload graph states were {states:?}").into());
    }
    let mut actual = vec![0_u8; input_bytes.len()];
    output.copy_to(&mut actual)?;
    if actual != input_bytes {
        return Err("upload -> dispatch graph output mismatch".into());
    }
    println!("ROCm owned upload -> dispatch event chain verified");
    Ok(())
}

/// Verify owned upload -> dispatch -> readback through serial and event-chained result paths.
#[allow(clippy::too_many_lines)] // Keep the three-node ownership fixture visible as one flow.
fn run_device_readback_graph_fixture(
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const ELEMENTS: usize = 65;
    let values = (0..ELEMENTS)
        .map(|index| u32::try_from(index).expect("bounded fixture") * 5 + 11)
        .collect::<Vec<_>>();
    let expected = encode_u32(&values);
    let input = backend.allocate(expected.len())?;
    let output = backend.allocate(expected.len())?;
    let upload_stream = backend.create_stream()?;
    let readback_stream = backend.create_stream()?;
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0054),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_readback_copy",
            logical_shape: [u32::try_from(ELEMENTS)?, 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let prepared = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(
            NonZeroU32::new(u32::try_from(ELEMENTS)?).ok_or("empty readback fixture")?,
        ),
    })?;
    let bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            input.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            output.clone(),
        )?,
    ];
    let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::HostUpload {
                stream: &upload_stream,
                source: Arc::from(expected.clone()),
                destination: &input,
                offset: 0,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &prepared,
                bindings: &bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[2],
            operation: RocmOwnedExecutionOperation::DeviceReadback {
                stream: &readback_stream,
                source: &output,
                source_offset: 0,
                bytes: expected.len(),
            },
        },
    ];
    let mut scratch = [false; 3];
    let mut states = [PcuExecutionNodeState::Pending; 3];
    let mut execution = RocmOwnedExecution::new(&nodes, &[], &mut scratch, &mut states)?;
    if !matches!(
        execution.take_readback(2),
        Err(RocmOwnedExecutionError::ReadbackNotReady { node: 2 })
    ) {
        return Err("readback was visible before graph execution".into());
    }
    for node in 0..3 {
        if execution.step()? != (RocmExecutionStep::Succeeded { node }) {
            return Err(format!("readback graph did not succeed at node {node}").into());
        }
    }
    if execution.step()? != RocmExecutionStep::Complete {
        return Err("readback graph did not complete".into());
    }
    let actual = execution.take_readback(2)?;
    if actual.as_ref() != expected {
        return Err("upload -> dispatch -> readback graph mismatch".into());
    }
    if !matches!(
        execution.take_readback(2),
        Err(RocmOwnedExecutionError::ReadbackAlreadyTaken { node: 2 })
    ) {
        return Err("readback graph allowed duplicate take".into());
    }
    drop(execution);
    if states != [PcuExecutionNodeState::Succeeded; 3] {
        return Err(format!("readback graph states were {states:?}").into());
    }
    let mut two_slot_scratch = [false; 3];
    let mut two_slot_states = [PcuExecutionNodeState::Pending; 3];
    let mut two_slot = RocmOwnedExecutionTwoSlot::new_event_chained(
        &nodes,
        &[],
        &mut two_slot_scratch,
        &mut two_slot_states,
    )?;
    if !matches!(
        two_slot.take_readback(2),
        Err(RocmOwnedExecutionError::ReadbackNotReady { node: 2 })
    ) {
        return Err("two-slot readback was visible before graph execution".into());
    }
    for node in 0..3 {
        if two_slot.step()? != (RocmTwoSlotExecutionStep::Submitted { node }) {
            return Err(format!("readback event chain did not submit node {node}").into());
        }
    }
    if two_slot.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 2 })
        || two_slot.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("readback event chain did not complete successfully".into());
    }
    let actual = two_slot.take_readback(2)?;
    if actual.as_ref() != expected {
        return Err("upload -> dispatch -> readback event chain mismatch".into());
    }
    if !matches!(
        two_slot.take_readback(2),
        Err(RocmOwnedExecutionError::ReadbackAlreadyTaken { node: 2 })
    ) {
        return Err("two-slot readback allowed duplicate take".into());
    }
    drop(two_slot);
    if two_slot_states != [PcuExecutionNodeState::Succeeded; 3] {
        return Err(format!("readback event chain states were {two_slot_states:?}").into());
    }
    let independent = [
        RocmOwnedExecutionNode {
            dependencies: &[],
            operation: RocmOwnedExecutionOperation::DeviceReadback {
                stream: &upload_stream,
                source: &input,
                source_offset: 0,
                bytes: expected.len(),
            },
        },
        RocmOwnedExecutionNode {
            dependencies: &[],
            operation: RocmOwnedExecutionOperation::DeviceReadback {
                stream: &readback_stream,
                source: &output,
                source_offset: 0,
                bytes: expected.len(),
            },
        },
    ];
    let mut independent_scratch = [false; 2];
    let mut independent_states = [PcuExecutionNodeState::Pending; 2];
    let mut independent_execution = RocmOwnedExecutionTwoSlot::new(
        &independent,
        &[],
        &mut independent_scratch,
        &mut independent_states,
    )?;
    if independent_execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 0 })
        || independent_execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 })
    {
        return Err("independent readbacks did not occupy both slots".into());
    }
    for node in 0..2 {
        if independent_execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node }) {
            return Err(format!("independent readback node {node} did not succeed").into());
        }
        if independent_execution.take_readback(node)?.as_ref() != expected {
            return Err(format!("independent readback node {node} output mismatch").into());
        }
    }
    if independent_execution.step()? != RocmTwoSlotExecutionStep::Complete {
        return Err("independent readbacks did not complete".into());
    }
    drop(independent_execution);
    if independent_states != [PcuExecutionNodeState::Succeeded; 2] {
        return Err(format!("independent readback states were {independent_states:?}").into());
    }
    let dispatch_stream = prepared.stream_handle();
    let same_stream_dependencies: [&[usize]; 2] = [&[], &[0]];
    let same_stream = [
        RocmOwnedExecutionNode {
            dependencies: same_stream_dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &prepared,
                bindings: &bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: same_stream_dependencies[1],
            operation: RocmOwnedExecutionOperation::DeviceReadback {
                stream: &dispatch_stream,
                source: &output,
                source_offset: 0,
                bytes: expected.len(),
            },
        },
    ];
    let mut same_stream_scratch = [false; 2];
    let mut same_stream_states = [PcuExecutionNodeState::Pending; 2];
    let mut same_stream_execution = RocmOwnedExecutionTwoSlot::new_event_chained(
        &same_stream,
        &[],
        &mut same_stream_scratch,
        &mut same_stream_states,
    )?;
    if same_stream_execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 0 })
        || same_stream_execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 })
        || same_stream_execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 1 })
        || same_stream_execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("same-stream dispatch -> readback did not complete".into());
    }
    if same_stream_execution.take_readback(1)?.as_ref() != expected {
        return Err("same-stream dispatch -> readback output mismatch".into());
    }
    drop(same_stream_execution);
    if same_stream_states != [PcuExecutionNodeState::Succeeded; 2] {
        return Err(format!("same-stream readback states were {same_stream_states:?}").into());
    }
    println!("ROCm owned serial, event-chain, independent, and same-stream readbacks verified");
    Ok(())
}

fn run_owned_dependency_chain_paired_diagnostic(
    producer: &fusion_pcu_rocm::RocmPreparedDispatch,
    producer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    consumer: &fusion_pcu_rocm::RocmPreparedDispatch,
    consumer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    output: &fusion_pcu_rocm::DeviceBuffer,
    input_bytes: &[u8],
    expected: &[u32],
) -> Result<(), Box<dyn Error>> {
    const PAIRS: usize = 16;
    const ITERATIONS: usize = 100;
    println!(
        "ROCm paired diagnostic: {PAIRS} alternating pairs; one warmup plus {ITERATIONS} timed synchronous runs per route"
    );
    for pair in 0..PAIRS {
        let host_wait_first = pair.is_multiple_of(2);
        let mut per_iteration = [std::time::Duration::ZERO; 2];
        for event_chaining in [!host_wait_first, host_wait_first] {
            run_owned_dependency_chain_once(
                producer,
                producer_bindings,
                consumer,
                consumer_bindings,
                event_chaining,
            )?;
            let started = std::time::Instant::now();
            for _ in 0..ITERATIONS {
                run_owned_dependency_chain_once(
                    producer,
                    producer_bindings,
                    consumer,
                    consumer_bindings,
                    event_chaining,
                )?;
            }
            per_iteration[usize::from(event_chaining)] =
                started.elapsed() / u32::try_from(ITERATIONS).expect("small fixed iteration count");
            verify_owned_dependency_chain_output(
                output,
                input_bytes,
                expected,
                if event_chaining {
                    "paired event-chain"
                } else {
                    "paired host-wait"
                },
            )?;
        }
        let ratio = per_iteration[0].as_secs_f64() / per_iteration[1].as_secs_f64();
        println!(
            "ROCm paired diagnostic pair {}: host-wait/event-chain = {ratio:.4} ({:?} / {:?} per run)",
            pair + 1,
            per_iteration[0],
            per_iteration[1]
        );
    }
    Ok(())
}

fn run_owned_dependency_chain_phase_profile(
    producer: &fusion_pcu_rocm::RocmPreparedDispatch,
    producer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    consumer: &fusion_pcu_rocm::RocmPreparedDispatch,
    consumer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    output: &fusion_pcu_rocm::DeviceBuffer,
    input_bytes: &[u8],
    expected: &[u32],
) -> Result<(), Box<dyn Error>> {
    const PAIRS: usize = 16;
    println!(
        "ROCm host phase profile: {PAIRS} alternating pairs; host-wait second step waits for producer, while event-chain second step submits consumer"
    );
    for pair in 0..PAIRS {
        let host_wait_first = pair.is_multiple_of(2);
        for event_chaining in [!host_wait_first, host_wait_first] {
            let phase = profile_owned_dependency_chain_route(
                producer,
                producer_bindings,
                consumer,
                consumer_bindings,
                event_chaining,
            )?;
            verify_owned_dependency_chain_output(
                output,
                input_bytes,
                expected,
                if event_chaining {
                    "phase-profile event-chain"
                } else {
                    "phase-profile host-wait"
                },
            )?;
            println!(
                "ROCm host phase profile pair {} {}: construction/admission {:?}, first step {:?}, second step {:?}, final drain {:?}",
                pair + 1,
                if event_chaining {
                    "event-chain"
                } else {
                    "host-wait"
                },
                phase[0],
                phase[1],
                phase[2],
                phase[3]
            );
        }
    }
    Ok(())
}

fn profile_owned_dependency_chain_route(
    producer: &fusion_pcu_rocm::RocmPreparedDispatch,
    producer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    consumer: &fusion_pcu_rocm::RocmPreparedDispatch,
    consumer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    event_chaining: bool,
) -> Result<[std::time::Duration; 4], Box<dyn Error>> {
    let dependencies: [&[usize]; 2] = [&[], &[0]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: producer,
                bindings: producer_bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: consumer,
                bindings: consumer_bindings,
            },
        },
    ];
    let mut scratch = [false; 2];
    let mut states = [PcuExecutionNodeState::Pending; 2];
    let started = std::time::Instant::now();
    let mut execution = if event_chaining {
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?
    } else {
        RocmOwnedExecutionTwoSlot::new(&nodes, &[], &mut scratch, &mut states)?
    };
    let mut phase = [std::time::Duration::ZERO; 4];
    phase[0] = started.elapsed();

    let started = std::time::Instant::now();
    if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 0 }) {
        return Err("phase profile did not submit the producer first".into());
    }
    phase[1] = started.elapsed();

    let started = std::time::Instant::now();
    let second = execution.step()?;
    let expected_second = if event_chaining {
        RocmTwoSlotExecutionStep::Submitted { node: 1 }
    } else {
        RocmTwoSlotExecutionStep::Succeeded { node: 0 }
    };
    if second != expected_second {
        return Err(format!("phase profile second step was {second:?}").into());
    }
    phase[2] = started.elapsed();

    let started = std::time::Instant::now();
    if !event_chaining && execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 }) {
        return Err("host-wait phase profile did not submit the consumer".into());
    }
    if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 1 })
        || execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("phase profile did not reach successful completion".into());
    }
    phase[3] = started.elapsed();
    if states
        != [
            PcuExecutionNodeState::Succeeded,
            PcuExecutionNodeState::Succeeded,
        ]
    {
        return Err(format!("phase-profile states were {states:?}").into());
    }
    Ok(phase)
}

fn run_owned_dependency_chain_once(
    producer: &fusion_pcu_rocm::RocmPreparedDispatch,
    producer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    consumer: &fusion_pcu_rocm::RocmPreparedDispatch,
    consumer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    event_chaining: bool,
) -> Result<(), Box<dyn Error>> {
    let dependencies: [&[usize]; 2] = [&[], &[0]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: producer,
                bindings: producer_bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: consumer,
                bindings: consumer_bindings,
            },
        },
    ];
    let mut scratch = [false; 2];
    let mut states = [PcuExecutionNodeState::Pending; 2];
    let mut execution = if event_chaining {
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?
    } else {
        RocmOwnedExecutionTwoSlot::new(&nodes, &[], &mut scratch, &mut states)?
    };
    if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 0 }) {
        return Err("dependency-chain executor did not submit the producer".into());
    }
    if event_chaining {
        if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 }) {
            return Err("event-chain executor did not submit the consumer before waiting".into());
        }
    } else if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 0 })
        || execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 })
    {
        return Err("host-wait executor did not complete the producer before consumer".into());
    }
    if execution.step()? != (RocmTwoSlotExecutionStep::Succeeded { node: 1 })
        || execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("dependency-chain executor did not reach successful completion".into());
    }
    if states
        != [
            PcuExecutionNodeState::Succeeded,
            PcuExecutionNodeState::Succeeded,
        ]
    {
        return Err(format!("dependency-chain states were {states:?}").into());
    }
    Ok(())
}

fn verify_owned_dependency_chain_output(
    output: &fusion_pcu_rocm::DeviceBuffer,
    input_bytes: &[u8],
    expected: &[u32],
    route: &str,
) -> Result<(), Box<dyn Error>> {
    let mut actual = vec![0_u8; input_bytes.len()];
    output.copy_to(&mut actual)?;
    if decode_u32(&actual)?.as_slice() != expected {
        return Err(format!("{route} dependency-chain output mismatch").into());
    }
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
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
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
    if N == 65 && !grid_stride {
        verify_unchecked_predecessor_batch(backend, &prepared, &bindings_owned)?;
    }
    let mut checked_batch = prepared.checked_batch();
    checked_batch.submit_checked_last(&prepared, &bindings_owned)?;
    let mut batch_completion = checked_batch.finish()?;
    if batch_completion.wait()? != fusion_pcu::PcuCompletionOutcome::Succeeded {
        return Err("checked-last batch unexpectedly faulted on valid i32 inputs".into());
    }
    verify_pair(
        "checked-last batch",
        &pcu_q,
        &pcu_r,
        &expected_q,
        &expected_r,
    )?;
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

fn run_owned_execution_gate_preflight(
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const N: usize = 65;
    let left = (0..N)
        .map(|i| i32::try_from(i).map(|index| index + 31))
        .collect::<Result<Vec<_>, _>>()?;
    let right = (0..N)
        .map(|i| if i.is_multiple_of(2) { 3 } else { 7 })
        .collect::<Vec<_>>();
    let lhs_bytes = encode_i32(&left);
    let rhs_bytes = encode_i32(&right);
    let mut lhs = backend.allocate(lhs_bytes.len())?;
    let mut rhs = backend.allocate(rhs_bytes.len())?;
    let quotient = backend.allocate(lhs_bytes.len())?;
    let remainder = backend.allocate(lhs_bytes.len())?;
    lhs.copy_from(&lhs_bytes)?;
    rhs.copy_from(&rhs_bytes)?;

    let (bindings, ops) = build_ops::<N>(false);
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0030),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_checked_div_rem_preflight",
            logical_shape: [65, 1, 1],
        },
        bindings,
        ports: &[],
        parameters: &[],
        ops,
        type_caps: PcuValueTypeCaps::INT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let producer = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(65).expect("nonzero")),
    })?;
    verify_owned_execution_fault_gate(backend, &producer, &lhs, &rhs, &quotient, &remainder)?;
    verify_owned_execution_two_slot(backend)?;
    verify_owned_execution_event_chain(backend)?;
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn verify_owned_execution_event_chain(
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const N: usize = 65;
    let expected = (0..N)
        .map(|value| u32::try_from(value).expect("small fixture") * 5 + 11)
        .collect::<Vec<_>>();
    let input_bytes = encode_u32(&expected);
    let mut input = backend.allocate(input_bytes.len())?;
    let intermediate = backend.allocate(input_bytes.len())?;
    let final_intermediate = backend.allocate(input_bytes.len())?;
    let output = backend.allocate(input_bytes.len())?;
    input.copy_from(&input_bytes)?;

    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0041),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_event_chain_copy",
            logical_shape: [u32::try_from(N).expect("small fixture"), 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let shape = PcuInvocationShape::invocations(
        NonZeroU32::new(u32::try_from(N).expect("small fixture")).expect("nonzero"),
    );
    let copy_stream = backend.create_stream()?;
    let producer = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let final_consumer = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let producer_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            input,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            intermediate.clone(),
        )?,
    ];
    let final_consumer_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            final_intermediate.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            output.clone(),
        )?,
    ];
    let dependencies: [&[usize]; 3] = [&[], &[0], &[1]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &producer,
                bindings: &producer_bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::DeviceCopy {
                stream: &copy_stream,
                source: &intermediate,
                destination: &final_intermediate,
                bytes: input_bytes.len(),
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[2],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &final_consumer,
                bindings: &final_consumer_bindings,
            },
        },
    ];
    let mut scratch = [false; 3];
    let mut states = [PcuExecutionNodeState::Pending; 3];
    let mut execution =
        RocmOwnedExecutionTwoSlot::new_event_chained(&nodes, &[], &mut scratch, &mut states)?;
    if execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 0 })
        || execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 1 })
        || execution.step()? != (RocmTwoSlotExecutionStep::Submitted { node: 2 })
    {
        return Err("event-chain preflight did not enqueue its three-node chain".into());
    }
    if !matches!(
        execution.step()?,
        RocmTwoSlotExecutionStep::Succeeded { node: 2 }
    ) || execution.step()? != RocmTwoSlotExecutionStep::Complete
    {
        return Err("event-chain preflight did not complete its producer lineage".into());
    }
    if states
        != [
            PcuExecutionNodeState::Succeeded,
            PcuExecutionNodeState::Succeeded,
            PcuExecutionNodeState::Succeeded,
        ]
    {
        return Err(format!("event-chain lineage states were {states:?}").into());
    }
    let mut actual = vec![0_u8; input_bytes.len()];
    output.copy_to(&mut actual)?;
    if decode_u32(&actual)?.as_slice() != expected.as_slice() {
        return Err("event-chain consumer output mismatch".into());
    }
    println!("ROCm owned execution event-chain copy verified across three streams");
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn verify_owned_execution_two_slot(
    backend: &RocmOwnedDispatchBackend,
) -> Result<(), Box<dyn Error>> {
    const N: usize = 65;
    let first_input = (0..N)
        .map(|value| u32::try_from(value).expect("small fixture") + 17)
        .collect::<Vec<_>>();
    let second_input = (0..N)
        .map(|value| u32::try_from(value).expect("small fixture") * 3 + 5)
        .collect::<Vec<_>>();
    let first_bytes = encode_u32(&first_input);
    let second_bytes = encode_u32(&second_input);
    let mut first_device_input = backend.allocate(first_bytes.len())?;
    let mut second_device_input = backend.allocate(second_bytes.len())?;
    let first_device_output = backend.allocate(first_bytes.len())?;
    let second_device_output = backend.allocate(second_bytes.len())?;
    first_device_input.copy_from(&first_bytes)?;
    second_device_input.copy_from(&second_bytes)?;

    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0040),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_two_slot_copy",
            logical_shape: [u32::try_from(N).expect("small fixture"), 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let shape = PcuInvocationShape::invocations(
        NonZeroU32::new(u32::try_from(N).expect("small fixture")).expect("nonzero"),
    );
    // Each preparation creates a separate stream on this exact backend runtime.
    let first_dispatch = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let second_dispatch = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &kernel,
        shape,
    })?;
    let first_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            first_device_input,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            first_device_output.clone(),
        )?,
    ];
    let second_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            second_device_input,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            second_device_output.clone(),
        )?,
    ];
    let dependencies: [&[usize]; 2] = [&[], &[]];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &first_dispatch,
                bindings: &first_bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: &second_dispatch,
                bindings: &second_bindings,
            },
        },
    ];
    let mut scratch = [false; 2];
    let mut states = [
        PcuExecutionNodeState::Pending,
        PcuExecutionNodeState::Pending,
    ];
    let mut execution = RocmOwnedExecutionTwoSlot::new(&nodes, &[], &mut scratch, &mut states)?;
    for expected_node in 0..2 {
        if execution.step()?
            != (RocmTwoSlotExecutionStep::Submitted {
                node: expected_node,
            })
        {
            return Err(
                "two-slot ROCm preflight did not submit both dispatches before waiting".into(),
            );
        }
    }
    let mut succeeded = [false; 2];
    loop {
        match execution.step()? {
            RocmTwoSlotExecutionStep::Succeeded { node } => succeeded[node] = true,
            RocmTwoSlotExecutionStep::Complete => break,
            RocmTwoSlotExecutionStep::Failed { node, outcome } => {
                return Err(format!("two-slot ROCm node {node} failed: {outcome:?}").into());
            }
            step => return Err(format!("unexpected two-slot ROCm preflight step: {step:?}").into()),
        }
    }
    if succeeded != [true, true] {
        return Err(
            format!("two-slot ROCm completions did not both succeed: {succeeded:?}").into(),
        );
    }
    for (label, buffer, expected) in [
        ("first", &first_device_output, &first_input),
        ("second", &second_device_output, &second_input),
    ] {
        let mut actual = vec![0_u8; expected.len() * std::mem::size_of::<u32>()];
        buffer.copy_to(&mut actual)?;
        if decode_u32(&actual)?.as_slice() != expected.as_slice() {
            return Err(format!("two-slot ROCm {label} copy output mismatch").into());
        }
    }
    println!("ROCm owned execution two-slot independent copies verified on distinct streams");
    Ok(())
}

static GATE_COPY_BINDINGS: [PcuBinding<'static>; 2] = [
    PcuBinding::value(
        Some("copy_input"),
        0,
        0,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::ReadOnly,
        PcuValueType::u32(),
    ),
    PcuBinding::value(
        Some("copy_output"),
        0,
        1,
        PcuBindingStorageClass::Storage,
        PcuBindingAccess::WriteOnly,
        PcuValueType::u32(),
    ),
];
static GATE_COPY_OPS: [PcuDispatchOp<'static>; 3] = [
    load(1, 0, PcuDispatchIndex::InvocationId),
    store(1, PcuDispatchIndex::InvocationId, 1),
    PcuDispatchOp::Control(fusion_pcu::model::dispatch::PcuDispatchControlOp::Return),
];

fn verify_owned_execution_fault_gate(
    backend: &RocmOwnedDispatchBackend,
    producer: &fusion_pcu_rocm::RocmPreparedDispatch,
    lhs: &fusion_pcu_rocm::DeviceBuffer,
    rhs: &fusion_pcu_rocm::DeviceBuffer,
    quotient: &fusion_pcu_rocm::DeviceBuffer,
    remainder: &fusion_pcu_rocm::DeviceBuffer,
) -> Result<(), Box<dyn Error>> {
    const N: usize = 65;
    let copy_kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0020),
        entry: PcuDispatchEntryPoint {
            name: "owned_execution_gated_copy",
            logical_shape: [65, 1, 1],
        },
        bindings: &GATE_COPY_BINDINGS,
        ports: &[],
        parameters: &[],
        ops: &GATE_COPY_OPS,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let consumer = backend.prepare_dispatch(PcuDispatchSubmission {
        kernel: &copy_kernel,
        shape: PcuInvocationShape::invocations(NonZeroU32::new(65).expect("nonzero")),
    })?;

    let mut copy_output = backend.allocate(N * std::mem::size_of::<i32>())?;
    let valid_bindings = make_owned_gate_producer_bindings(backend, lhs, rhs, quotient, remainder)?;
    run_owned_gate_graph(
        backend,
        producer,
        &consumer,
        &valid_bindings,
        quotient,
        &mut copy_output,
        (
            RocmExecutionStep::Succeeded { node: 0 },
            RocmExecutionStep::Succeeded { node: 1 },
        ),
    )?;
    for (left_values, right_values, kind) in [
        (
            None,
            [0_i32; N],
            fusion_pcu::PcuExecutionFaultKind::DivideByZero,
        ),
        (
            Some([i32::MIN; N]),
            [-1_i32; N],
            fusion_pcu::PcuExecutionFaultKind::SignedDivisionOverflow,
        ),
    ] {
        let fault_lhs = if let Some(values) = left_values {
            let bytes = encode_i32(&values);
            let mut buffer = backend.allocate(bytes.len())?;
            buffer.copy_from(&bytes)?;
            buffer
        } else {
            lhs.clone()
        };
        let right_bytes = encode_i32(&right_values);
        let mut fault_rhs = backend.allocate(right_bytes.len())?;
        fault_rhs.copy_from(&right_bytes)?;
        let fault_bindings = make_owned_gate_producer_bindings(
            backend, &fault_lhs, &fault_rhs, quotient, remainder,
        )?;
        run_owned_gate_graph(
            backend,
            producer,
            &consumer,
            &fault_bindings,
            quotient,
            &mut copy_output,
            (
                RocmExecutionStep::Failed {
                    node: 0,
                    outcome: fusion_pcu::PcuCompletionOutcome::Fault(
                        fusion_pcu::PcuExecutionFault {
                            recovered: false,
                            kind,
                            invocation_id: 0,
                        },
                    ),
                },
                RocmExecutionStep::Blocked { node: 1 },
            ),
        )?;
    }
    println!("ROCm owned execution gate verified success and both checked arithmetic faults");
    Ok(())
}

fn make_owned_gate_producer_bindings(
    backend: &RocmOwnedDispatchBackend,
    lhs: &fusion_pcu_rocm::DeviceBuffer,
    rhs: &fusion_pcu_rocm::DeviceBuffer,
    quotient: &fusion_pcu_rocm::DeviceBuffer,
    remainder: &fusion_pcu_rocm::DeviceBuffer,
) -> Result<[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>; 4], Box<dyn Error>> {
    Ok([
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            lhs.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            rhs.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 2),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            quotient.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 3),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::i32()),
            remainder.clone(),
        )?,
    ])
}

fn run_owned_gate_graph(
    backend: &RocmOwnedDispatchBackend,
    producer: &fusion_pcu_rocm::RocmPreparedDispatch,
    consumer: &fusion_pcu_rocm::RocmPreparedDispatch,
    producer_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
    quotient: &fusion_pcu_rocm::DeviceBuffer,
    copy_output: &mut fusion_pcu_rocm::DeviceBuffer,
    expected_steps: (RocmExecutionStep, RocmExecutionStep),
) -> Result<(), Box<dyn Error>> {
    const N: usize = 65;
    const SENTINEL: i32 = 0x5a5a_5a5a;
    let sentinel_bytes = encode_i32(&[SENTINEL; N]);
    copy_output.copy_from(&sentinel_bytes)?;
    let consumer_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            quotient.clone(),
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::WriteOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            copy_output.clone(),
        )?,
    ];
    let dependencies: [&[usize]; 2] = [&[], &[0]];
    let gates = [PcuExecutionSuccessGate {
        predecessor: 0,
        successor: 1,
    }];
    let nodes = [
        RocmOwnedExecutionNode {
            dependencies: dependencies[0],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: producer,
                bindings: producer_bindings,
            },
        },
        RocmOwnedExecutionNode {
            dependencies: dependencies[1],
            operation: RocmOwnedExecutionOperation::Dispatch {
                prepared: consumer,
                bindings: &consumer_bindings,
            },
        },
    ];
    let mut scratch = [false; 2];
    let mut states = [PcuExecutionNodeState::Pending; 2];
    let mut execution = RocmOwnedExecution::new(&nodes, &gates, &mut scratch, &mut states)?;
    if execution.step()? != expected_steps.0 {
        return Err(format!(
            "owned execution producer step differed from {:?}",
            expected_steps.0
        )
        .into());
    }
    if execution.step()? != expected_steps.1 {
        return Err(format!(
            "owned execution consumer step differed from {:?}",
            expected_steps.1
        )
        .into());
    }
    if execution.step()? != RocmExecutionStep::Complete {
        return Err("owned execution did not reach completion".into());
    }
    let mut actual = vec![0_u8; sentinel_bytes.len()];
    copy_output.copy_to(&mut actual)?;
    let actual = decode_i32(&actual)?;
    match expected_steps.1 {
        RocmExecutionStep::Succeeded { .. } => {
            let mut expected = vec![0_u8; sentinel_bytes.len()];
            quotient.copy_to(&mut expected)?;
            if actual != decode_i32(&expected)? {
                return Err("owned execution gated consumer copy output mismatch".into());
            }
        }
        RocmExecutionStep::Blocked { .. } => {
            if actual != vec![SENTINEL; N] {
                return Err(
                    "owned execution submitted a gated consumer after producer fault".into(),
                );
            }
        }
        _ => return Err("invalid expected consumer step in owned execution smoke".into()),
    }
    Ok(())
}

fn verify_unchecked_predecessor_batch(
    backend: &RocmOwnedDispatchBackend,
    checked: &fusion_pcu_rocm::RocmPreparedDispatch,
    checked_bindings: &[PcuOwnedBinding<fusion_pcu_rocm::DeviceBuffer>],
) -> Result<(), Box<dyn Error>> {
    let source = (0..65_u32).map(|x| x ^ 0xa5a5_5a5a).collect::<Vec<_>>();
    let bytes = source
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    let mut input = backend.allocate(bytes.len())?;
    let output = backend.allocate(bytes.len())?;
    input.copy_from(&bytes)?;
    let bindings = [
        PcuBinding::value(
            Some("copy_input"),
            0,
            0,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadOnly,
            PcuValueType::u32(),
        ),
        PcuBinding::value(
            Some("copy_output"),
            0,
            1,
            PcuBindingStorageClass::Storage,
            PcuBindingAccess::ReadWrite,
            PcuValueType::u32(),
        ),
    ];
    let operations = [
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingLoad {
            result: PcuDispatchValueId(1),
            binding: PcuBindingRef::new(0, 0),
            index: PcuDispatchIndex::InvocationId,
        }),
        PcuDispatchOp::Data(PcuDispatchDataOp::BindingStore {
            binding: PcuBindingRef::new(0, 1),
            index: PcuDispatchIndex::InvocationId,
            value: PcuDispatchValueId(1),
        }),
        PcuDispatchOp::Control(PcuDispatchControlOp::Return),
    ];
    let kernel = PcuDispatchKernelIr {
        numerical_requirements: PcuDispatchKernelIr::DEFAULT_REQUIREMENTS,
        id: PcuKernelId(0xD1_0010),
        entry: PcuDispatchEntryPoint {
            name: "batch_predecessor_copy",
            logical_shape: [65, 1, 1],
        },
        bindings: &bindings,
        ports: &[],
        parameters: &[],
        ops: &operations,
        type_caps: PcuValueTypeCaps::UINT32 | PcuValueTypeCaps::SCALAR_VALUES,
        feature_caps: PcuDispatchFeatureCaps::READ_ONLY_RESOURCES
            .union(PcuDispatchFeatureCaps::MUTABLE_RESOURCES),
    };
    let prepared_copy = backend.prepare_dispatch_on_stream(
        kernel,
        PcuInvocationShape::invocations(NonZeroU32::new(65).expect("nonzero")),
        &checked.stream_handle(),
    )?;
    let copy_bindings = [
        backend.binding(
            PcuBindingRef::new(0, 0),
            PcuBindingAccess::ReadOnly,
            PcuBindingType::Value(PcuValueType::u32()),
            input,
        )?,
        backend.binding(
            PcuBindingRef::new(0, 1),
            PcuBindingAccess::ReadWrite,
            PcuBindingType::Value(PcuValueType::u32()),
            output.clone(),
        )?,
    ];
    let mut batch = checked.checked_batch();
    batch.submit_unchecked(&prepared_copy, &copy_bindings)?;
    batch.submit_checked_last(checked, checked_bindings)?;
    let mut completion = batch.finish()?;
    if completion.wait()? != fusion_pcu::PcuCompletionOutcome::Succeeded {
        return Err("checked-tail batch with an unchecked predecessor faulted".into());
    }
    let mut actual = vec![0_u8; bytes.len()];
    output.copy_to(&mut actual)?;
    if actual != bytes {
        return Err("unchecked predecessor copy output mismatch".into());
    }
    println!("ordered unchecked copy -> checked i32 DivRem batch verified");
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

fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_u32(bytes: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<u32>()) {
        return Err("u32 readback had a partial element".into());
    }
    Ok(bytes
        .as_chunks::<{ std::mem::size_of::<u32>() }>()
        .0
        .iter()
        .map(|chunk| u32::from_ne_bytes(*chunk))
        .collect())
}

fn decode_i32(bytes: &[u8]) -> Result<Vec<i32>, Box<dyn Error>> {
    if !bytes.len().is_multiple_of(std::mem::size_of::<i32>()) {
        return Err("i32 readback had a partial element".into());
    }
    Ok(bytes
        .as_chunks::<{ std::mem::size_of::<i32>() }>()
        .0
        .iter()
        .map(|chunk| i32::from_le_bytes(*chunk))
        .collect())
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| i32::from_le_bytes(*x))
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
        fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
        | fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
        | fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand => {
            unreachable!("division benchmark only creates division faults")
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

    let mut checked_batch = prepared.checked_batch();
    checked_batch.submit_checked_last(prepared, &fault_bindings)?;
    if !matches!(
        checked_batch.submit_unchecked(prepared, &fault_bindings),
        Err(fusion_pcu_rocm::RocmOwnedDispatchError::CheckedArithmeticBatchClosed)
    ) {
        return Err("checked-last batch admitted a subsequent launch".into());
    }
    let mut batch_completion = checked_batch.finish()?;
    let expected_fault = fusion_pcu::PcuExecutionFault {
        recovered: false,
        kind: fault_kind,
        invocation_id: expected_id,
    };
    if batch_completion.wait()? != fusion_pcu::PcuCompletionOutcome::Fault(expected_fault) {
        return Err(format!(
            "checked-last batch did not report {fault_kind:?} at ID {expected_id}"
        )
        .into());
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
        fusion_pcu::PcuExecutionFaultKind::ArithmeticOverflow
        | fusion_pcu::PcuExecutionFaultKind::ArithmeticUnderflow
        | fusion_pcu::PcuExecutionFaultKind::InvalidFloatingOperand => {
            unreachable!("division benchmark only creates division faults")
        }
    };
    let expected = (expected_id << 3) | fault_tag;
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
        "#include <hip/hip_runtime.h>\nextern \"C\" __global__ void native_checked_i32_div_rem(const int* a,const int* b,int* q,int* r,unsigned long long* fault_word) {{\nunsigned int base=blockIdx.x*blockDim.x+threadIdx.x; if (base >= {invocations}u) return; {iteration} if (b[i] == 0) {{ atomicMin(fault_word, (static_cast<unsigned long long>(i) << 3u) | 1ull); }} else if (a[i] == (-2147483647 - 1) && b[i] == -1) {{ atomicMin(fault_word, (static_cast<unsigned long long>(i) << 3u) | 2ull); }} else {{ q[i]=a[i]/b[i]; r[i]=a[i]%b[i]; }} {close} }}\n"
    )
}

criterion_group! { name = benches; config = support::criterion_config(); targets = bench }
criterion_main!(benches);
