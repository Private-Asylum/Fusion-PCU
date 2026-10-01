//! Matched full synchronous host-boundary source/graph/global/independent native Vulkan comparison.

#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/support/support.rs"]
mod support;
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
    pcu,
    global,
    PcuBindingRef,
    PcuScalar,
    PcuScalarType,
    PcuHostKernelBackend,
    PcuFloatUnderflowPolicy,
    PcuHostArgument,
    PcuPreparedHostKernel,
    PcuRangePolicy,
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
    PcuVulkanCallMeasurements,
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

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate<const N: usize>(input: &[f32], output: &mut [f32]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

fn require_gpu_idle() {
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

trait Sample: PcuScalar {
    const F64: bool;
    const GROUP: &'static str;
    fn seed(index: usize) -> Self;
    fn bits(self) -> u64;
    fn from_bits(bits: u64) -> Self;
    fn change(self) -> Self;
}

impl Sample for f32 {
    const F64: bool = false;
    const GROUP: &'static str = "vulkan_checked_neg_host";
    fn seed(index: usize) -> Self {
        let bits = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x3f80_0000,
            0xc000_0000,
        ];
        Self::from_bits(bits[index % bits.len()])
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(u32::try_from(bits).unwrap())
    }
    fn change(self) -> Self {
        if self.to_bits() == 1.0_f32.to_bits() {
            2.0
        } else {
            1.0
        }
    }
}

impl Sample for f64 {
    const F64: bool = true;
    const GROUP: &'static str = "vulkan_checked_neg_f64_host";
    fn seed(index: usize) -> Self {
        let bits = [
            0,
            0x8000_0000_0000_0000,
            1,
            0x8000_0000_0000_0001,
            0x000f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0xffef_ffff_ffff_ffff,
            0x3ff0_0000_0000_0000,
            0xc000_0000_0000_0000,
        ];
        Self::from_bits(bits[index % bits.len()])
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bits(bits: u64) -> Self {
        Self::from_bits(bits)
    }
    fn change(self) -> Self {
        if self.to_bits() == 1.0_f64.to_bits() {
            2.0
        } else {
            1.0
        }
    }
}

fn verify<T: Sample>(input: &[T], output: &[T]) {
    let sign = if T::F64 {
        0x8000_0000_0000_0000
    } else {
        0x8000_0000
    };
    for (value, actual) in input.iter().zip(output) {
        assert_eq!(actual.bits(), value.bits() ^ sign);
    }
    for actual in &output[input.len()..] {
        assert_eq!(actual.bits(), T::seed(9).bits());
    }
}

fn native_call<T: Sample>(
    native: &mut ffi::NativeNeg,
    input: &[T],
    output: &mut [T],
    timings: Option<&mut ffi::Timings>,
) {
    let input = PcuHostArgument::read(PcuBindingRef::new(0, 0), input);
    let mut output = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output);
    native
        .call(input.bytes(), output.bytes_mut().unwrap(), timings)
        .expect("independent native execution");
}

fn native_exceptional_check<T: Sample>(native: &mut ffi::NativeNeg, extent: usize) {
    let mut input = vec![T::seed(8); extent];
    let mut output = vec![T::seed(9); extent + 3];
    let patterns: &[u64] = if T::F64 {
        &[
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0001,
            0x7ff0_0000_0000_0001,
        ]
    } else {
        &[0x7f80_0000, 0xff80_0000, 0x7fc0_0001, 0x7f80_0001]
    };
    for &bits in patterns {
        input[7] = T::from_bits(bits);
        input[16] = T::from_bits(patterns[2]);
        let source = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
        let mut destination = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output);
        let error = native
            .call(source.bytes(), destination.bytes_mut().unwrap(), None)
            .unwrap_err();
        assert!(error.to_string().contains("status1 at invocation7"));
        assert!(output.iter().all(|value| value.bits() == T::seed(9).bits()));
    }
    input.fill(T::seed(8));
    native_call(native, &input, &mut output, None);
    verify(&input, &output);
    let mut strict = ffi::NativeNeg::new(
        *SELECTED_IDENTITY.get().unwrap(),
        u32::try_from(extent).unwrap(),
        T::F64,
        true,
    )
    .unwrap();
    input[7] = T::from_bits(1);
    output.fill(T::seed(9));
    let source = PcuHostArgument::read(PcuBindingRef::new(0, 0), &input);
    let mut destination = PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output);
    let error = strict
        .call(source.bytes(), destination.bytes_mut().unwrap(), None)
        .unwrap_err();
    assert!(error.to_string().contains("status2 at invocation7"));
    assert!(output.iter().all(|value| value.bits() == T::seed(9).bits()));
    input.fill(T::seed(8));
    native_call(&mut strict, &input, &mut output, None);
    verify(&input, &output);
    println!(
        "native exceptional acceptance {}: nonfinite first fault, subnormal rejection, complete rollback, retry, padded tails",
        T::GROUP
    );
}

fn census<T: Sample>(
    label: &str,
    input: &mut [T],
    output: &mut [T],
    mut run: impl FnMut(&[T], &mut [T]),
) {
    let mut elapsed = std::time::Duration::ZERO;
    let counts = ffi::count_heap(|| {
        for _ in 0..32 {
            input[0] = input[0].change();
            let start = std::time::Instant::now();
            run(input, output);
            elapsed += start.elapsed();
            verify(input, output);
        }
    });
    println!(
        "census {label} elements{} calls32 Rust alloc{} realloc{} free{} full-wall-mean{:?}; Vulkan warm allocations0, resets32, queue-submits32, terminal-fence-waits32 (fixed retained call path)",
        input.len(),
        counts.allocations,
        counts.reallocations,
        counts.frees,
        elapsed / 32
    );
}

#[allow(clippy::too_many_lines)] // All peers keep identical data, policy, boundary and oracle adjacent.
fn compare_typed<const N: usize, T: Sample>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    mut source: impl FnMut(&[T], &mut [T]),
    mut ordinary: impl FnMut(&[T], &mut [T]),
) {
    require_gpu_idle();
    let mut graph = if T::F64 {
        support::with_typed_graph(
            u32::try_from(N).unwrap(),
            PcuFloatUnderflowPolicy::default(),
            PcuRangePolicy::Reject,
            false,
            PcuScalarType::F64,
            |kernel| backend.prepare_host_kernel(kernel),
        )
    } else {
        support::prepare_graph(
            backend,
            u32::try_from(N).unwrap(),
            PcuFloatUnderflowPolicy::default(),
            PcuRangePolicy::Reject,
            false,
        )
    }
    .expect("matched graph prepares");
    println!(
        "{} {N}elements actual input/output/status realization {:?}",
        T::GROUP,
        graph.memory_realizations()
    );
    let mut native = ffi::NativeNeg::new(
        *SELECTED_IDENTITY.get().unwrap(),
        u32::try_from(N).unwrap(),
        T::F64,
        false,
    )
    .expect("independent native Vulkan prepares");
    if N == 17 {
        native_exceptional_check::<T>(&mut native, N);
    }
    let mut input: Vec<T> = (0..N).map(T::seed).collect();
    let mut output = vec![T::seed(9); N + 3];
    source(&input, &mut output);
    verify(&input, &output);
    ordinary(&input, &mut output);
    verify(&input, &output);
    let cold_scores = COLD_SCORES.load(Ordering::Relaxed);
    if std::env::var_os("PCU_VULKAN_CENSUS").is_some() {
        census("source", &mut input, &mut output, &mut source);
        census("graph", &mut input, &mut output, |input, output| {
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                ])
                .unwrap();
        });
        census("global", &mut input, &mut output, &mut ordinary);
        census("native", &mut input, &mut output, |input, output| {
            native_call(&mut native, input, output, None);
        });
        let mut pcu_sum = PcuVulkanCallMeasurements::default();
        for _ in 0..32 {
            input[0] = input[0].change();
            let report = graph
                .call_profiled(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &input),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                ])
                .unwrap();
            verify(&input, &output);
            pcu_sum.upload += report.upload;
            pcu_sum.submission += report.submission;
            pcu_sum.completion += report.completion;
            pcu_sum.diagnostic_publication += report.diagnostic_publication;
            pcu_sum.wall += report.wall;
        }
        println!(
            "PCU graph stage means upload{:?} submission{:?} completion{:?} diagnostic-publication{:?} native-full-wall{:?}",
            pcu_sum.upload / 32,
            pcu_sum.submission / 32,
            pcu_sum.completion / 32,
            pcu_sum.diagnostic_publication / 32,
            pcu_sum.wall / 32
        );
        let mut sum = ffi::Timings::default();
        for _ in 0..32 {
            input[0] = input[0].change();
            let mut report = ffi::Timings::default();
            native_call(&mut native, &input, &mut output, Some(&mut report));
            verify(&input, &output);
            sum.upload += report.upload;
            sum.submission += report.submission;
            sum.completion += report.completion;
            sum.diagnostic_publication += report.diagnostic_publication;
            sum.wall += report.wall;
        }
        println!(
            "native stage means upload{:?} submission{:?} completion{:?} diagnostic-publication{:?} full-wall{:?}",
            sum.upload / 32,
            sum.submission / 32,
            sum.completion / 32,
            sum.diagnostic_publication / 32,
            sum.wall / 32
        );
    }
    let mut group = criterion.benchmark_group(T::GROUP);
    group.throughput(Throughput::Elements(N as u64));
    group.bench_with_input(BenchmarkId::new("source", N), &N, |bencher, _| {
        bencher.iter_custom(|iterations| {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iterations {
                input[0] = input[0].change();
                let start = std::time::Instant::now();
                source(black_box(&input), black_box(&mut output));
                elapsed += start.elapsed();
                verify(&input, &output);
            }
            elapsed
        });
    });
    require_gpu_idle();
    group.bench_with_input(BenchmarkId::new("graph", N), &N, |bencher, _| {
        bencher.iter_custom(|iterations| {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iterations {
                input[0] = input[0].change();
                let start = std::time::Instant::now();
                graph
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), black_box(&input)),
                        PcuHostArgument::read_write(
                            PcuBindingRef::new(0, 1),
                            black_box(&mut output),
                        ),
                    ])
                    .unwrap();
                elapsed += start.elapsed();
                verify(&input, &output);
            }
            elapsed
        });
    });
    require_gpu_idle();
    group.bench_with_input(BenchmarkId::new("global", N), &N, |bencher, _| {
        bencher.iter_custom(|iterations| {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iterations {
                input[0] = input[0].change();
                let start = std::time::Instant::now();
                ordinary(black_box(&input), black_box(&mut output));
                elapsed += start.elapsed();
                verify(&input, &output);
            }
            elapsed
        });
    });
    require_gpu_idle();
    group.bench_with_input(BenchmarkId::new("native", N), &N, |bencher, _| {
        bencher.iter_custom(|iterations| {
            let mut elapsed = std::time::Duration::ZERO;
            for _ in 0..iterations {
                input[0] = input[0].change();
                let start = std::time::Instant::now();
                native_call(&mut native, black_box(&input), black_box(&mut output), None);
                elapsed += start.elapsed();
                verify(&input, &output);
            }
            elapsed
        });
    });
    assert_eq!(
        COLD_SCORES.load(Ordering::Relaxed),
        cold_scores,
        "warm global calls retain cold selection"
    );
    group.finish();
}

#[pcu(invocations = N, crate_path = ::pcu_facade)]
fn negate_f64<const N: usize>(input: &[f64], output: &mut [f64]) {
    let id = context.global_invocation_id;
    output[id] = -input[id];
}

fn compare<const N: usize>(criterion: &mut Criterion, backend: &PcuVulkanBackend) {
    let mut source = negate_prepare::<N, _>(backend).expect("actual F32 source prepares");
    compare_typed::<N, f32>(
        criterion,
        backend,
        |input, output| source(input, output).expect("F32 source"),
        |input, output| negate::<N>(input, output).expect("F32 ordinary global source"),
    );
    let mut source = negate_f64_prepare::<N, _>(backend).expect("actual F64 source prepares");
    compare_typed::<N, f64>(
        criterion,
        backend,
        |input, output| source(input, output).expect("F64 source"),
        |input, output| negate_f64::<N>(input, output).expect("F64 ordinary global source"),
    );
}

fn benchmarks(criterion: &mut Criterion) {
    require_gpu_idle();
    let backend = selected_backend();
    println!("Vulkan benchmark device: {}", backend.name());
    compare::<17>(criterion, &backend);
    compare::<4096>(criterion, &backend);
    compare::<1_048_576>(criterion, &backend);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
