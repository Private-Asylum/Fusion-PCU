//! Cold setup and oracles stay outside timing; warm calls change actual input bytes.
use std::hint::black_box;
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize, Ordering};
#[rustfmt::skip]
use criterion::{BenchmarkId, Criterion, Throughput};
#[rustfmt::skip]
use fusion_pcu::{global, PcuNumericalMode, PcuFloatUnderflowPolicy,
    PcuRangePolicy, PcuDispatchCheckedFloatConversion, PcuHostKernelBackend,
    PcuPreparedHostKernel, PcuHostArgument};
use super::{allocator, graph, native::Native, source};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    global::default_device_score(&candidate.device, candidate.total_memory_bytes.unwrap_or(0))
}
fn configure(backend: global::PcuBackendChoice, mode: PcuNumericalMode) {
    global::configure(global::PcuExecutionPolicy {
        backend,
        numerical_mode: mode,
        score_invocation: Some(score),
        ..global::PcuExecutionPolicy::default()
    })
    .unwrap();
}
macro_rules! direction {
    ($name:ident, $src:ty, $dst:ty, $cast:ident, $strict:ident, $prepare:ident,
        $strict_prepare:ident, $conversion:ident) => {
        fn $name<const N: usize, B: PcuHostKernelBackend>(
            criterion: &mut Criterion, backend: &B, choice: global::PcuBackendChoice,
            native: &Native, mode: PcuNumericalMode,
        ) where B::Error: std::fmt::Debug, <B::Prepared as PcuPreparedHostKernel>::Error: std::fmt::Debug {
            configure(choice, mode);
            let mut captured = source::$prepare::<N, _>(backend).unwrap();
            let mut captured_strict = source::$strict_prepare::<N, _>(backend).unwrap();
            let mut explicit = graph::with(PcuDispatchCheckedFloatConversion::$conversion,
                u32::try_from(N).unwrap(), PcuRangePolicy::Reject,
                PcuFloatUnderflowPolicy::IeeeAfterRounding, false, false, |ir| {
                    let mut ir = ir.clone();
                    ir.numerical_requirements.numerical_mode = mode;
                    backend.prepare_host_kernel(&ir).unwrap()
                });
            let mut input = [<$src>::from(1_u16); N];
            let mut output = vec![<$dst>::from(77_u16); N + 2];
            let mut scratch = vec![0_u8; N * std::mem::size_of::<$dst>()];
            let mut call = |route: usize, phase: u16, verify: bool| {
                input[0] = <$src>::from(phase);
                match route {
                    0 if mode == PcuNumericalMode::Strict => source::$strict(&input, &mut output).unwrap(),
                    0 => source::$cast(&input, &mut output).unwrap(),
                    1 if mode == PcuNumericalMode::Strict => captured_strict(&input, &mut output).unwrap(),
                    1 => captured(&input, &mut output).unwrap(),
                    2 => explicit.call(&mut [PcuHostArgument::read(graph::INPUT, &input),
                        PcuHostArgument::read_write(graph::OUTPUT, &mut output)]).unwrap(),
                    3 => native.call(&input, &mut output, &mut scratch),
                    _ => unreachable!("registered route"),
                }
                black_box(&output);
                if verify {
                assert_eq!(output[0].to_bits(), <$dst>::from(phase).to_bits());
                assert!(output[1..N].iter().all(|x| x.to_bits() == <$dst>::from(1_u16).to_bits()));
                assert!(output[N..].iter().all(|x| x.to_bits() == <$dst>::from(77_u16).to_bits()));
                }
            };
            let mut group = criterion.benchmark_group(format!("apple_conversion/{choice:?}/{mode:?}/{}", stringify!($conversion)));
            group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
            for (route, name) in ["ordinary_pcu", "generated_prepared", "explicit_ir", "detached_native"].into_iter().enumerate() {
                call(route, 1, true);
                let scores = SCORES.load(Ordering::Relaxed);
                let (_, counts) = allocator::observe(|| {
                    for phase in 2..66 { call(route, phase, false); }
                });
                assert_eq!(SCORES.load(Ordering::Relaxed), scores, "warm calls reranked");
                eprintln!("CONVERSION_CENSUS {choice:?} {mode:?} {} {N} {name} calls=64 {counts:?} score_calls=0", stringify!($conversion));
                call(route, 66, true);
                let mut phase = 1_u16;
                group.bench_function(BenchmarkId::new(name, N), |bencher| bencher.iter(|| {
                    phase = if phase == 1024 { 1 } else { phase + 1 };
                    call(route, phase, false);
                }));
            }
            group.finish();
        }
    };
}
direction!(
    narrow,
    f64,
    f32,
    narrow,
    narrow_strict,
    narrow_prepare,
    narrow_strict_prepare,
    F64ToF32
);
direction!(
    widen,
    f32,
    f64,
    widen,
    widen_strict,
    widen_prepare,
    widen_strict_prepare,
    F32ToF64
);
pub fn run<const N: usize>(criterion: &mut Criterion) {
    let metal = fusion_pcu_metal::MetalSession::open(0).unwrap();
    let metal_backend = fusion_pcu_metal::MetalConversionHostBackend::new(metal.clone());
    let mlx = fusion_pcu_mlx::MlxRuntime::load_default()
        .unwrap()
        .open_gpu(0)
        .unwrap();
    let mlx_backend = fusion_pcu_mlx::MlxConversionHostBackend::new(mlx.clone());
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        narrow::<N, _>(
            criterion,
            &metal_backend,
            global::PcuBackendChoice::Metal,
            &Native::metal(&metal, PcuDispatchCheckedFloatConversion::F64ToF32),
            mode,
        );
        widen::<N, _>(
            criterion,
            &metal_backend,
            global::PcuBackendChoice::Metal,
            &Native::metal(&metal, PcuDispatchCheckedFloatConversion::F32ToF64),
            mode,
        );
        narrow::<N, _>(
            criterion,
            &mlx_backend,
            global::PcuBackendChoice::Mlx,
            &Native::mlx(&mlx, PcuDispatchCheckedFloatConversion::F64ToF32, N),
            mode,
        );
        widen::<N, _>(
            criterion,
            &mlx_backend,
            global::PcuBackendChoice::Mlx,
            &Native::mlx(&mlx, PcuDispatchCheckedFloatConversion::F32ToF64, N),
            mode,
        );
    }
}
