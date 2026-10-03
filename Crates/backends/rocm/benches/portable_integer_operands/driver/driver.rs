//! Genuine source, prepared IR and exact projected native ABI with matched full-host work.
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
    PcuRangePolicy,
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
fn prepared<T: Format>(
    explicit: &mut impl PcuPreparedHostKernel,
    kind: u32,
    left: &[T],
    right: &[T],
    out: &mut [T],
) {
    let empty: &[T] = &[];
    let result = match kind {
        0 | 3 => explicit.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), out),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), left),
        ]),
        1 | 4 => explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), empty),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), out),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), left),
        ]),
        2 => explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), right),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), out),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), left),
        ]),
        _ => unreachable!(),
    };
    assert!(result.is_ok());
}
#[allow(clippy::too_many_lines)] // Matched source/IR/native setup, verification and physical census stay together.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    kind: u32,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    guard();
    global::clear_thread_cache().unwrap();
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native = Native::new::<T, N>(backend, ir, 1);
    let banks = [
        oracle::inputs::<T>(N, 0, kind),
        oracle::inputs::<T>(N, 1, kind),
    ];
    let mut out = vec![T::sentinel(); N + 2];
    for (left, right, want) in &banks {
        host(left, right, &mut out).unwrap();
        oracle::verify(want, &out);
        prepared(&mut explicit, kind, left, right, &mut out);
        oracle::verify(want, &out);
        if kind == 2 {
            native.host(right, left, &mut out);
        } else {
            native.host(left, right, &mut out);
        }
        oracle::verify(want, &out);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "rocm_portable_integer_operands/{:?}/{:?}/{}/{kind}/{N}",
        ir.numerical_requirements.numerical_mode,
        ir.numerical_requirements.range_policy,
        T::LABEL
    ));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in ["actual_source", "prepared_ir", "native_same_kernel"] {
        guard();
        let mut bank = 0;
        let mut call = || {
            bank ^= 1;
            let (left, right, want) = &banks[bank];
            let started = Instant::now();
            match route {
                "actual_source" => host(left, right, &mut out).unwrap(),
                "prepared_ir" => prepared(&mut explicit, kind, left, right, &mut out),
                _ => {
                    if kind == 2 {
                        native.host(right, left, &mut out);
                    } else {
                        native.host(left, right, &mut out);
                    }
                }
            }
            let elapsed = started.elapsed();
            oracle::verify(want, &out);
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
            bench.iter_custom(|iterations| (0..iterations).map(|_| call()).sum::<Duration>())
        });
    }
    group.finish();
}
#[inline(never)] // Each cold arena is independent of the enumeration frame.
fn width<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    macro_rules! entry {
        ($name:ident,$bindings:ident,$builder:ident,$kind:literal,$call:expr) => {{
            let bindings = source::$bindings::<T>();
            let builder = source::$builder::<T, N>(
                &bindings,
                requirements.float_underflow,
                requirements.range_policy,
                requirements,
            )
            .unwrap();
            builder.with_ir(|ir| case::<T, N>(criterion, backend, $kind, ir, $call));
        }};
    }
    entry!(
        repeated_add,
        repeated_add_bindings,
        __repeated_add_ir_with_float_underflow_policy,
        0,
        |a, _b, out| source::repeated_add::<T, N>(out, a)
    );
    entry!(
        unused_mul,
        unused_mul_bindings,
        __unused_mul_ir_with_float_underflow_policy,
        1,
        |a, _b, out| source::unused_mul::<T, N>(&[], out, a)
    );
    entry!(
        reordered_sub,
        reordered_sub_bindings,
        __reordered_sub_ir_with_float_underflow_policy,
        2,
        |a, b, out| source::reordered_sub::<T, N>(b, out, a)
    );
    entry!(
        indexed_zero_sub,
        indexed_zero_sub_bindings,
        __indexed_zero_sub_ir_with_float_underflow_policy,
        3,
        |a, _b, out| source::indexed_zero_sub::<T, N>(out, a)
    );
    entry!(
        grid_zero_sub,
        grid_zero_sub_bindings,
        __grid_zero_sub_ir_with_float_underflow_policy,
        4,
        |a, _b, out| source::grid_zero_sub::<T, N>(&[], out, a)
    );
}
pub fn run(criterion: &mut Criterion) {
    let (_, backend, _) = super::selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
            let requirements = PcuImplementationRequirements {
                numerical_mode: mode,
                range_policy: range,
                numerical_options: fusion_pcu::PcuNumericalOptions {
                    reproducibility: fusion_pcu::PcuReproducibility::PortableV1,
                    ..fusion_pcu::PcuNumericalOptions::default()
                },
                ..PcuImplementationRequirements::DEFAULT
            };
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Rocm,
                device: Some(backend.device_identity().device_id()),
                numerical_mode: mode,
                range_policy: range,
                numerical_options: requirements.numerical_options,
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
            ty!(i8);
            ty!(u8);
            ty!(i16);
            ty!(u16);
            ty!(i32);
            ty!(u32);
            ty!(i64);
            ty!(u64);
            ty!(i128);
            ty!(u128);
            ty!(fusion_pcu::PcuI256);
            ty!(fusion_pcu::PcuU256);
            ty!(fusion_pcu::PcuI512);
            ty!(fusion_pcu::PcuU512);
        }
    }
    global::clear_thread_cache().unwrap();
}
