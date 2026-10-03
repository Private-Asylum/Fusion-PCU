//! Paired genuine source/ordinary, explicit graph and independent ash checked F64 binaries.
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "../../tests/clamped_binary/source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
use std::hint::black_box;
use std::sync::OnceLock;
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuExecutionFault,
    PcuExecutionFaultKind,
    PcuRangePolicy,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuDispatchFloatBinaryOp,
    PcuFloatUnderflowPolicy,
    PcuDeviceClass,
    PcuDeviceDescriptor,
    PcuObjectKind,
    PcuObjectRef,
    PcuProviderDescriptor,
    PcuProviderId,
    PcuProviderReadiness,
    PcuProviderStatus,
    PcuRuntimeDiscovery,
    PcuStableDeviceIdentity,
    PcuTargetDescriptor,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanDiscovery,
};
static SELECTED_IDENTITY: OnceLock<PcuStableDeviceIdentity> = OnceLock::new();
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);

fn matched_device_score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(
        candidate.facts.stable_identity.as_ref(),
        SELECTED_IDENTITY.get()
    );
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

fn selected_backend() -> PcuVulkanBackend {
    let discovery = PcuVulkanDiscovery::discover().expect("physical Vulkan discovery");
    let reference = PcuObjectRef {
        provider: PcuProviderId(0),
        generation: 0,
        kind: PcuObjectKind::Device,
        id: 0,
    };
    let readiness = PcuProviderReadiness {
        status: PcuProviderStatus::Unavailable,
        reason: None,
    };
    let mut providers = [PcuProviderDescriptor {
        id: PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness,
    }];
    discovery.providers(&mut providers).unwrap();
    let mut targets = [PcuTargetDescriptor {
        reference,
        name: "",
        readiness,
    }];
    discovery
        .targets(providers[0].id, providers[0].generation, &mut targets)
        .unwrap();
    let mut devices = [PcuDeviceDescriptor {
        reference,
        target: reference,
        name: "",
        class: PcuDeviceClass::Other,
        vendor: None,
        architecture: None,
        generation: None,
        location: None,
    }];
    assert!(
        discovery
            .devices(targets[0].reference, &mut devices)
            .unwrap()
            > 0,
        "a physical GPU is required"
    );
    let device = devices[0].reference;
    SELECTED_IDENTITY
        .set(
            discovery
                .device_facts(device)
                .unwrap()
                .stable_identity
                .expect("physical device UUID"),
        )
        .unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        device: Some(device.id),
        score_invocation: Some(matched_device_score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    PcuVulkanBackend::open(&discovery, device).expect("selected physical Vulkan GPU")
}

fn require_gpu_idle() {
    if std::env::args().any(|argument| argument == "--test") {
        println!("Criterion semantic smoke only: no statistical timing samples");
        return;
    }
    let path = std::env::var_os("PCU_VULKAN_GPU_BUSY_PATH").map_or_else(
        || std::path::PathBuf::from("/sys/class/drm/card1/device/gpu_busy_percent"),
        std::path::PathBuf::from,
    );
    let mut idle = 0;
    for _ in 0..200 {
        let busy: u32 = std::fs::read_to_string(&path)
            .expect("GPU idle guard requires a device activity counter")
            .trim()
            .parse()
            .expect("GPU activity counter is an integer percent");
        idle = if busy <= 5 { idle + 1 } else { 0 };
        if idle == 3 {
            println!(
                "GPU idle guard admitted <=5% activity at {}",
                path.display()
            );
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!(
        "GPU idle guard did not reach three <=5% readings in ten seconds at {}",
        path.display()
    );
}

#[rustfmt::skip]
use ffi::{
    bytes,
    bytes_mut,
};

fn fault(error: fusion_pcu_vulkan::PcuVulkanError) -> PcuExecutionFault {
    match error {
        fusion_pcu_vulkan::PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected backend error {other:?}"),
    }
}
fn ordinary_fault(error: global::PcuExecutionError) -> PcuExecutionFault {
    match error {
        global::PcuExecutionError::ArithmeticFault(fault) => fault,
        other => panic!("unexpected source error {other:?}"),
    }
}
macro_rules! workload {
    ($run:ident,$entry:ident,$prepare:ident,$op:ident,$code:literal,$right:expr) => {
        #[allow(clippy::too_many_lines)] // Four matched warm publication boundaries and their census stay together.
        fn $run<const N: usize>(criterion: &mut Criterion, backend: &PcuVulkanBackend) {
            let mut left = vec![f64::MAX; N];
            let right = vec![$right; N];
            let mut output = vec![17.0; N + 3];
            let mut source = source::$prepare::<f64, N, _>(backend).unwrap();
            let mut graph = graph::Graph::new(
                u32::try_from(N).unwrap(),
                PcuDispatchFloatBinaryOp::$op,
                PcuFloatUnderflowPolicy::default(),
            );
            graph.range = PcuRangePolicy::Clamp;
            graph.scalar = pcu_facade::PcuScalarType::F64;
            let mut explicit = graph
                .with(|kernel| backend.prepare_host_kernel(kernel))
                .unwrap();
            let mut native = ffi::NativeBinary::new(
                SELECTED_IDENTITY.get().unwrap().clone(),
                u32::try_from(N).unwrap(),
                $code,
                0,
            )
            .unwrap();
            let expected = PcuExecutionFault {
                recovered: true,
                invocation_id: 0,
                kind: PcuExecutionFaultKind::ArithmeticOverflow,
            };
            for route in 0..4 {
                for _ in 0..2 {
                    left[0] = f64::from_bits(left[0].to_bits() ^ 0x0000_0001);
                    let result = match route {
                        0 => source(&left, &right, &mut output).map_err(fault),
                        1 => source::$entry::<f64, N>(&left, &right, &mut output)
                            .map_err(ordinary_fault),
                        2 => explicit
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                                PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                            ])
                            .map_err(fault),
                        _ => native
                            .call_clamped(bytes(&left), bytes(&right), bytes_mut(&mut output))
                            .unwrap()
                            .map_or(Ok(()), Err),
                    };
                    assert_eq!(result, Err(expected));
                    assert!(
                        output[..N]
                            .iter()
                            .all(|value| value.to_bits() == f64::MAX.to_bits())
                    );
                    assert!(
                        output[N..]
                            .iter()
                            .all(|value| value.to_bits() == 17.0_f64.to_bits())
                    );
                }
            }
            left[N - 1] = f64::NAN;
            let before: Vec<_> = output.iter().map(|value| value.to_bits()).collect();
            let error = native
                .call_clamped(bytes(&left), bytes(&right), bytes_mut(&mut output))
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains(&format!("status1 at invocation{}", N - 1))
            );
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                before
            );
            left[N - 1] = f64::MAX;
            assert_eq!(
                native
                    .call_clamped(bytes(&left), bytes(&right), bytes_mut(&mut output))
                    .unwrap(),
                Some(expected)
            );
            let warm_scores = COLD_SCORES.load(Ordering::Relaxed);
            for route in 0..4 {
                let counts = ffi::count_heap(|| {
                    for _ in 0..64 {
                        left[0] = f64::from_bits(left[0].to_bits() ^ 0x0000_0001);
                        let result = match route {
                            0 => source(&left, &right, &mut output).map_err(fault),
                            1 => source::$entry::<f64, N>(&left, &right, &mut output)
                                .map_err(ordinary_fault),
                            2 => explicit
                                .call(&mut [
                                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                                    PcuHostArgument::read_write(
                                        PcuBindingRef::new(0, 2),
                                        &mut output,
                                    ),
                                ])
                                .map_err(fault),
                            _ => native
                                .call_clamped(bytes(&left), bytes(&right), bytes_mut(&mut output))
                                .unwrap()
                                .map_or(Ok(()), Err),
                        };
                        assert_eq!(result, Err(expected));
                    }
                });
                assert_eq!(
                    (counts.allocations, counts.reallocations, counts.frees),
                    (0, 0, 0)
                );
                println!(
                    "Clamp {} N{} route{} 64 warm calls heap {:?}",
                    stringify!($entry),
                    N,
                    route,
                    (counts.allocations, counts.reallocations, counts.frees)
                );
            }
            assert_eq!(COLD_SCORES.load(Ordering::Relaxed), warm_scores);
            require_gpu_idle();
            {
                let mut group = criterion
                    .benchmark_group(format!("vulkan_clamped_f64_{}_host", stringify!($entry)));
                group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
                for (route, label) in [
                    "pcu_source_prepared",
                    "pcu_source_ordinary",
                    "explicit_graph_diagnostic",
                    "native_ash_integer_kernel",
                ]
                .into_iter()
                .enumerate()
                {
                    group.bench_function(BenchmarkId::new(label, N), |bench| {
                        bench.iter(|| {
                            left[0] = f64::from_bits(left[0].to_bits() ^ 0x0000_0001);
                            let result = match route {
                                0 => source(
                                    black_box(&left),
                                    black_box(&right),
                                    black_box(&mut output),
                                )
                                .map_err(fault),
                                1 => source::$entry::<f64, N>(
                                    black_box(&left),
                                    black_box(&right),
                                    black_box(&mut output),
                                )
                                .map_err(ordinary_fault),
                                2 => explicit
                                    .call(&mut [
                                        PcuHostArgument::read(
                                            PcuBindingRef::new(0, 0),
                                            black_box(&left),
                                        ),
                                        PcuHostArgument::read(
                                            PcuBindingRef::new(0, 1),
                                            black_box(&right),
                                        ),
                                        PcuHostArgument::read_write(
                                            PcuBindingRef::new(0, 2),
                                            black_box(&mut output),
                                        ),
                                    ])
                                    .map_err(fault),
                                _ => native
                                    .call_clamped(
                                        bytes(black_box(&left)),
                                        bytes(black_box(&right)),
                                        bytes_mut(black_box(&mut output)),
                                    )
                                    .unwrap()
                                    .map_or(Ok(()), Err),
                            };
                            assert_eq!(result, Err(expected));
                            black_box(&output);
                        })
                    });
                }
                group.finish();
            }
            assert_eq!(COLD_SCORES.load(Ordering::Relaxed), warm_scores);
        }
    };
}
workload!(add, add, add_prepare, Add, 0, f64::MAX);
workload!(sub, sub, sub_prepare, Sub, 1, -f64::MAX);
workload!(mul, mul, mul_prepare, Mul, 2, 2.0);
workload!(div, div, div_prepare, Div, 3, 0.5);
fn benchmark(criterion: &mut Criterion) {
    let backend = selected_backend();
    add::<1>(criterion, &backend);
    add::<4096>(criterion, &backend);
    sub::<1>(criterion, &backend);
    sub::<4096>(criterion, &backend);
    mul::<1>(criterion, &backend);
    mul::<4096>(criterion, &backend);
    div::<1>(criterion, &backend);
    div::<4096>(criterion, &backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
#[path = "reject/reject.rs"]
mod reject;
criterion_group!(benches, reject::benchmark, benchmark);
criterion_main!(benches);
