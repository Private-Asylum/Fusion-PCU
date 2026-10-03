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
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
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
#[allow(clippy::too_many_lines)] // One scope freezes and compares the complete physical boundary.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    kind: u32,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    guard();
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native = Native::new::<T, N>(backend, ir, 1);
    let banks = [
        oracle::inputs::<T>(N, 0, kind, ir.numerical_requirements.float_underflow),
        oracle::inputs::<T>(N, 1, kind, ir.numerical_requirements.float_underflow),
    ];
    let expected = banks.each_ref().map(|(_, result)| result.clone());
    let mut output = vec![T::sentinel(); N + 2];
    let mut invalid = banks[0].0.clone();
    invalid[0] = T::from(T::MAX + 1);
    native.upload(0, &invalid);
    assert_eq!(native.submit(0), 5);
    let mut prepared_call = |input: &[T], output: &mut [T]| {
        explicit
            .call(&mut [
                PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), input),
            ])
            .unwrap();
    };
    for (bank, (input, _)) in banks.iter().enumerate() {
        host(input, &mut output).unwrap();
        oracle::verify(&expected[bank], &output);
        prepared_call(input, &mut output);
        oracle::verify(&expected[bank], &output);
        native.host(input, &mut output);
        oracle::verify(&expected[bank], &output);
        native.read_full(&mut output);
        oracle::verify(&expected[bank], &output);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let name = format!(
        "rocm_portable_unary/{:?}/{:?}/{:?}/{}/{kind}/{N}",
        ir.numerical_requirements.numerical_mode,
        ir.numerical_requirements.float_underflow,
        ir.numerical_requirements.range_policy,
        T::LABEL
    );
    let mut group = criterion.benchmark_group(name);
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in ["actual_source", "prepared_ir", "native_same_kernel"] {
        guard();
        let mut bank = 0;
        let mut call = || {
            bank ^= 1;
            let input = &banks[bank].0;
            let started = Instant::now();
            match route {
                "actual_source" => host(input, &mut output).unwrap(),
                "prepared_ir" => prepared_call(input, &mut output),
                _ => native.host(input, &mut output),
            }
            let elapsed = started.elapsed();
            oracle::verify(&expected[bank], &output);
            assert_eq!(
                SCORES.load(Ordering::Relaxed),
                scores,
                "warm source reranked providers"
            );
            elapsed
        };
        let _ = call();
        #[cfg(feature = "allocation-census")]
        {
            fusion_pcu_rocm::reset_rocm_api_census();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(call());
                }
            });
            let api = fusion_pcu_rocm::rocm_api_census();
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
    backend: &RocmOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    let uf = requirements.float_underflow;
    let range = requirements.range_policy;
    let bindings = source::neg_bindings::<T>();
    let builder =
        source::__neg_ir_with_float_underflow_policy::<T, N>(&bindings, uf, range, requirements)
            .unwrap();
    case::<T, N>(criterion, backend, 0, &builder.ir(), |input, output| {
        source::neg::<T, N>(output, input)
    });
    let bindings = source::relu_bindings::<T>();
    let builder =
        source::__relu_ir_with_float_underflow_policy::<T, N>(&bindings, uf, range, requirements)
            .unwrap();
    case::<T, N>(criterion, backend, 1, &builder.ir(), |input, output| {
        source::relu::<T, N>(output, input)
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
            for range_policy in [
                fusion_pcu::PcuRangePolicy::Reject,
                fusion_pcu::PcuRangePolicy::Clamp,
            ] {
                let numerical_options = fusion_pcu::PcuNumericalOptions {
                    reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
                    ..Default::default()
                };
                let requirements = PcuImplementationRequirements {
                    numerical_mode: mode,
                    float_underflow: uf,
                    range_policy,
                    numerical_options,
                };
                global::configure(global::PcuExecutionPolicy {
                    backend: global::PcuBackendChoice::Rocm,
                    device: Some(backend.device_identity().device_id()),
                    numerical_mode: mode,
                    float_underflow: uf,
                    range_policy,
                    numerical_options,
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
