//! Cold projected ABI; genuine source, prepared IR and same native kernel full-host work.
#[rustfmt::skip]
use std::{
    time::{Duration, Instant},
    sync::atomic::{AtomicUsize, Ordering},
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
use super::{
    native::Native,
    oracle::{self, Format},
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
        oracle::inputs::<T>(if kind == 3 { 1 } else { N }, 0, kind >= 2),
        oracle::inputs::<T>(if kind == 3 { 1 } else { N }, 1, kind >= 2),
    ];
    let expected = banks.each_ref().map(|(_, result)| {
        if kind == 3 {
            vec![result[0]; N]
        } else {
            result.clone()
        }
    });
    let mut output = vec![T::sentinel(); N + 2];
    let empty: &[T] = &[];
    let mut invalid = banks[0].0.clone();
    invalid[0] = T::from(T::MAX + 1);
    native.upload(0, &invalid);
    assert_eq!(native.submit(0), 5);
    let mut prepared_call = |input: &[T], output: &mut [T]| {
        if kind == 0 {
            explicit
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), output),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), input),
                ])
                .unwrap();
        } else {
            explicit
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), empty),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                    PcuHostArgument::read(PcuBindingRef::new(0, 2), input),
                ])
                .unwrap();
        }
    };
    for (bank, (input, _)) in banks.iter().enumerate() {
        host(input, &mut output).unwrap();
        oracle::verify(&expected[bank], &output);
        prepared_call(input, &mut output);
        oracle::verify(&expected[bank], &output);
        native.host(input, &mut output);
        oracle::verify(&expected[bank], &output);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let name = format!(
        "rocm_checked_float_operands/{:?}/{:?}/{}/{kind}/{N}",
        ir.numerical_requirements.numerical_mode,
        ir.numerical_requirements.float_underflow,
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
                "census/{:?}/{:?}/{}/{kind}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}",
                ir.numerical_requirements.numerical_mode,
                ir.numerical_requirements.float_underflow,
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
    let single = source::mul::single_bindings::<T>();
    let repeated = source::mul::unused_bindings::<T>();
    let seed = source::div::seed_bindings::<T>();
    let uf = requirements.float_underflow;
    let range = requirements.range_policy;
    let builder = source::mul::__single_ir_with_float_underflow_policy::<T, N>(
        &single,
        uf,
        range,
        requirements,
    )
    .unwrap();
    case::<T, N>(criterion, backend, 0, &builder.ir(), |input, output| {
        source::mul::single::<T, N>(output, input)
    });
    let builder = source::mul::__unused_ir_with_float_underflow_policy::<T, N>(
        &repeated,
        uf,
        range,
        requirements,
    )
    .unwrap();
    case::<T, N>(criterion, backend, 1, &builder.ir(), |input, output| {
        source::mul::unused::<T, N>(&[], output, input)
    });
    let builder = source::div::__grid_ir_with_float_underflow_policy::<T, N>(
        &repeated,
        uf,
        range,
        requirements,
    )
    .unwrap();
    builder.with_ir(|ir| {
        case::<T, N>(criterion, backend, 2, ir, |input, output| {
            source::div::grid::<T, N>(&[], output, input)
        });
    });
    let builder =
        source::div::__seed_ir_with_float_underflow_policy::<T, N>(&seed, uf, range, requirements)
            .unwrap();
    case::<T, N>(criterion, backend, 3, &builder.ir(), |input, output| {
        source::div::seed::<T, N>(&[], output, &input[0])
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
            let requirements = PcuImplementationRequirements {
                numerical_mode: mode,
                float_underflow: uf,
                ..PcuImplementationRequirements::DEFAULT
            };
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Rocm,
                device: Some(backend.device_identity().device_id()),
                numerical_mode: mode,
                float_underflow: uf,
                score_invocation: Some(score),
                ..Default::default()
            })
            .unwrap();
            macro_rules! ty {
                ($t:ty) => {
                    width::<$t, 65>(criterion, &backend, requirements);
                    width::<$t, 65536>(criterion, &backend, requirements);
                };
            }
            ty!(f32);
            ty!(f64);
            ty!(fusion_pcu::PcuF16Bits);
            ty!(fusion_pcu::PcuBf16Bits);
            ty!(fusion_pcu::PcuF8E4M3FnBits);
            ty!(fusion_pcu::PcuF8E5M2Bits);
        }
    }
}
