//! Paired genuine source/ordinary, explicit graph and independent ash checked F32 binaries.
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../../spirv/tests/checked_binary/support/support.rs"]
mod graph;
#[path = "../../tests/checked_binary/source/source.rs"]
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
    PcuCheckedFloat,
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
macro_rules! workload {
    ($run:ident,$entry:ident,$prepare:ident,$op:ident,$code:literal,$checked:ident) => {
        #[allow(clippy::too_many_lines)] // Four matched cold/warm ownership boundaries stay together.
        fn $run<const N: usize>(criterion: &mut Criterion, backend: &PcuVulkanBackend) {
            let mut left = vec![6.0_f32; N];
            let right = vec![2.0_f32; N];
            let mut output = vec![0.0_f32; N + 3];
            let expected = 6.0_f32.$checked(2.0).unwrap();
            let mut source = source::f32_source::$prepare::<N, _>(backend).unwrap();
            let mut explicit = graph::Graph::new(
                u32::try_from(N).unwrap(),
                PcuDispatchFloatBinaryOp::$op,
                PcuFloatUnderflowPolicy::default(),
            )
            .with(|kernel| backend.prepare_host_kernel(kernel))
            .unwrap();
            let mut native = ffi::NativeBinary::new(
                SELECTED_IDENTITY.get().unwrap().clone(),
                u32::try_from(N).unwrap(),
                $code,
                0,
            )
            .unwrap();
            source(&left, &right, &mut output).unwrap();
            assert!(
                output[..N]
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
            source::f32_source::$entry::<N>(&left, &right, &mut output).unwrap();
            native
                .call(bytes(&left), bytes(&right), bytes_mut(&mut output), None)
                .unwrap();
            assert!(
                output[..N]
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
            for value in [7.0_f32, 6.0] {
                left[0] = value;
                let changed = value.$checked(2.0).unwrap().to_bits();
                source(&left, &right, &mut output).unwrap();
                assert_eq!(output[0].to_bits(), changed);
                source::f32_source::$entry::<N>(&left, &right, &mut output).unwrap();
                assert_eq!(output[0].to_bits(), changed);
                explicit
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                    ])
                    .unwrap();
                assert_eq!(output[0].to_bits(), changed);
                native
                    .call(bytes(&left), bytes(&right), bytes_mut(&mut output), None)
                    .unwrap();
                assert_eq!(output[0].to_bits(), changed);
                assert_eq!(&output[N..], &[0.0; 3]);
            }
            left[0] = f32::NAN;
            let before: Vec<_> = output.iter().map(|value| value.to_bits()).collect();
            let error = native
                .call(bytes(&left), bytes(&right), bytes_mut(&mut output), None)
                .unwrap_err();
            assert!(error.to_string().contains("status1 at invocation0"));
            assert_eq!(
                output
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                before
            );
            left[0] = 6.0;
            native
                .call(bytes(&left), bytes(&right), bytes_mut(&mut output), None)
                .unwrap();
            let mut stages = ffi::Timings::default();
            native
                .call(
                    bytes(&left),
                    bytes(&right),
                    bytes_mut(&mut output),
                    Some(&mut stages),
                )
                .unwrap();
            assert_eq!(
                stages.wall,
                stages.upload
                    + stages.submission
                    + stages.completion
                    + stages.diagnostic_publication
            );
            let warm_scores = COLD_SCORES.load(Ordering::Relaxed);
            let source_heap = ffi::count_heap(|| {
                source(&left, &right, &mut output).unwrap();
            });
            let ordinary_heap = ffi::count_heap(|| {
                source::f32_source::$entry::<N>(&left, &right, &mut output).unwrap();
            });
            let graph_heap = ffi::count_heap(|| {
                explicit
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), &left),
                        PcuHostArgument::read(PcuBindingRef::new(0, 1), &right),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut output),
                    ])
                    .unwrap();
            });
            let native_heap = ffi::count_heap(|| {
                native
                    .call(bytes(&left), bytes(&right), bytes_mut(&mut output), None)
                    .unwrap();
            });
            for counts in [source_heap, ordinary_heap, graph_heap, native_heap] {
                assert_eq!(
                    (counts.allocations, counts.reallocations, counts.frees),
                    (0, 0, 0)
                );
            }
            assert_eq!(COLD_SCORES.load(Ordering::Relaxed), warm_scores);
            println!(
                "binary {} N{} actual allocation metadata {:?}",
                stringify!($entry),
                N,
                explicit.memory_realizations()
            );
            require_gpu_idle();
            {
                let mut group = criterion
                    .benchmark_group(format!("vulkan_checked_f32_{}_host", stringify!($entry)));
                group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
                group.bench_function(BenchmarkId::new("pcu_source_prepared", N), |bench| {
                    bench.iter(|| {
                        left[0] = f32::from_bits(left[0].to_bits() ^ 0x0020_0000);
                        source(black_box(&left), black_box(&right), black_box(&mut output))
                            .unwrap();
                        black_box(&output);
                    })
                });
                group.bench_function(BenchmarkId::new("pcu_source_ordinary", N), |bench| {
                    bench.iter(|| {
                        left[0] = f32::from_bits(left[0].to_bits() ^ 0x0020_0000);
                        source::f32_source::$entry::<N>(
                            black_box(&left),
                            black_box(&right),
                            black_box(&mut output),
                        )
                        .unwrap();
                        black_box(&output);
                    })
                });
                group.bench_function(BenchmarkId::new("explicit_graph_diagnostic", N), |bench| {
                    bench.iter(|| {
                        left[0] = f32::from_bits(left[0].to_bits() ^ 0x0020_0000);
                        explicit
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), black_box(&left)),
                                PcuHostArgument::read(PcuBindingRef::new(0, 1), black_box(&right)),
                                PcuHostArgument::read_write(
                                    PcuBindingRef::new(0, 2),
                                    black_box(&mut output),
                                ),
                            ])
                            .unwrap();
                        black_box(&output);
                    })
                });
                group.bench_function(BenchmarkId::new("native_ash_integer_kernel", N), |bench| {
                    bench.iter(|| {
                        left[0] = f32::from_bits(left[0].to_bits() ^ 0x0020_0000);
                        native
                            .call(
                                bytes(black_box(&left)),
                                bytes(black_box(&right)),
                                bytes_mut(black_box(&mut output)),
                                None,
                            )
                            .unwrap();
                        black_box(&output);
                    })
                });
                group.finish();
            }
            assert_eq!(COLD_SCORES.load(Ordering::Relaxed), warm_scores);
        }
    };
}
workload!(add, add, add_prepare, Add, 0, pcu_checked_add);
workload!(sub, sub, sub_prepare, Sub, 1, pcu_checked_sub);
workload!(mul, mul, mul_prepare, Mul, 2, pcu_checked_mul);
workload!(div, div, div_prepare, Div, 3, pcu_checked_div);
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
criterion_group!(benches, benchmark);
criterion_main!(benches);
