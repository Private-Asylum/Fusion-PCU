//! Cold projected ABI; genuine source, prepared IR and same native kernel full-host work.
#[rustfmt::skip]
use std::{
    time::{
        Duration,
        Instant,
    },
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    Throughput,
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuOwnedDispatchBackend,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuFloatUnderflowPolicy,
};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Format,
    },
    source,
};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn guard() {
    if !std::env::args().any(|a| a == "--test") {
        super::activity::guard();
    }
}
#[allow(clippy::too_many_lines)] // One scope matches both observable outputs and complete API boundary.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    kind: u32,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    guard();
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    let mut native = Native::new::<T, N>(backend, ir, 1);
    let banks = [oracle::bank::<T>(N, 0), oracle::bank::<T>(N, 1)];
    let mut stage = vec![T::sentinel(); N + 2];
    let mut output = stage.clone();
    let mut invalid = banks[0].0.clone();
    invalid[2] = T::from(T::MAX + 1);
    assert!(host(&invalid, &mut stage, &mut output).is_err());
    assert!(
        stage
            .iter()
            .chain(&output)
            .all(|value| value.bits() == T::sentinel().bits())
    );
    assert!(
        prepared
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), &mut stage),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                PcuHostArgument::read(PcuBindingRef::new(0, 2), &invalid),
            ])
            .is_err()
    );
    assert!(
        stage
            .iter()
            .chain(&output)
            .all(|value| value.bits() == T::sentinel().bits())
    );
    native.upload(0, &invalid);
    assert_eq!(
        native.submit(0),
        (2 << 3) | 5,
        "first invalid operand lane/status"
    );
    let mut explicit = |input: &[T], stage: &mut [T], output: &mut [T]| {
        prepared
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), stage),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                PcuHostArgument::read(PcuBindingRef::new(0, 2), input),
            ])
            .unwrap();
    };
    for (input, want_stage, want_output) in &banks {
        host(input, &mut stage, &mut output).unwrap();
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        explicit(input, &mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        native.host(input, &mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        native.minimal_host(input, &mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
        native.read_full(&mut stage, &mut output);
        oracle::verify(want_stage, &stage);
        oracle::verify(want_output, &output);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "{}/{:?}/{:?}/{:?}/{}/{kind}/{N}",
        super::WORKLOAD_LABEL,
        ir.numerical_requirements.numerical_mode,
        ir.numerical_requirements.float_underflow,
        ir.numerical_requirements.range_policy,
        T::LABEL
    ));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in [
        "actual_source",
        "prepared_ir",
        "native_matched_boundary",
        "native_minimal_kernel",
    ] {
        guard();
        let mut bank = 0;
        let mut call = || {
            bank ^= 1;
            let (input, want_stage, want_output) = &banks[bank];
            let started = Instant::now();
            match route {
                "actual_source" => host(input, &mut stage, &mut output).unwrap(),
                "prepared_ir" => explicit(input, &mut stage, &mut output),
                "native_matched_boundary" => native.host(input, &mut stage, &mut output),
                _ => native.minimal_host(input, &mut stage, &mut output),
            }
            let elapsed = started.elapsed();
            oracle::verify(want_stage, &stage);
            oracle::verify(want_output, &output);
            assert_eq!(
                SCORES.load(Ordering::Relaxed),
                scores,
                "warm source reranked"
            );
            elapsed
        };
        let _ = call();
        #[cfg(feature = "allocation-census")]
        {
            fusion_pcu_cuda::reset_cuda_api_census();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(call());
                }
            });
            let api = fusion_pcu_cuda::cuda_api_census();
            assert_eq!(api.symbol_resolutions, 0);
            assert_eq!(api.module_loads, 0);
            assert_eq!(api.allocations, 0);
            assert_eq!(api.frees, 0);
            assert_eq!(api.kernel_launches, 64);
            eprintln!(
                "census/{:?}/{:?}/{:?}/{}/{kind}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}",
                ir.numerical_requirements.numerical_mode,
                ir.numerical_requirements.float_underflow,
                ir.numerical_requirements.range_policy,
                T::LABEL,
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        group.bench_function(route, |bench| {
            bench.iter_custom(|iterations| (0..iterations).map(|_| call()).sum::<Duration>());
        });
    }
    group.finish();
}
fn width<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    let bindings = source::direct_bindings::<T>();
    let builder = source::__direct_ir_with_float_underflow_policy::<T, N>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    case::<T, N>(
        criterion,
        backend,
        0,
        &builder.ir(),
        |input, stage, output| source::direct::<T, N>(stage, output, input),
    );
    let builder = source::__grid_ir_with_float_underflow_policy::<T, N>(
        &bindings,
        requirements.float_underflow,
        requirements.range_policy,
        requirements,
    )
    .unwrap();
    builder.with_ir(|ir| {
        case::<T, N>(criterion, backend, 1, ir, |input, stage, output| {
            source::grid::<T, N>(stage, output, input)
        });
    });
}
pub fn run(criterion: &mut Criterion) {
    let (_, backend, _) = super::selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for uf in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
        ] {
            for range in [
                fusion_pcu::PcuRangePolicy::Reject,
                fusion_pcu::PcuRangePolicy::Clamp,
            ] {
                let requirements = PcuImplementationRequirements {
                    numerical_mode: mode,
                    float_underflow: uf,
                    range_policy: range,
                    ..PcuImplementationRequirements::DEFAULT
                };
                global::configure(global::PcuExecutionPolicy {
                    backend: global::PcuBackendChoice::Cuda,
                    device: Some(backend.device_identity().device_id()),
                    numerical_mode: mode,
                    float_underflow: uf,
                    range_policy: range,
                    score_invocation: Some(score),
                    ..Default::default()
                })
                .unwrap();
                macro_rules! ty {
                    ($t:ty) => {
                        width::<$t, 65>(criterion, &backend, requirements);
                        width::<$t, 4096>(criterion, &backend, requirements);
                    };
                }
                ty!(fusion_pcu::PcuF16Bits);
                ty!(fusion_pcu::PcuBf16Bits);
                ty!(fusion_pcu::PcuF8E4M3FnBits);
                ty!(fusion_pcu::PcuF8E5M2Bits);
                ty!(f32);
                ty!(f64);
            }
        }
    }
}
