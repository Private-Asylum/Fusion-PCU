//! Genuine source, prepared IR and exact native joint ABI at the same full-host boundary.
#[rustfmt::skip]
use std::{
    sync::atomic::{
        AtomicUsize,
        Ordering,
    },
    time::{
        Duration,
        Instant,
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
    PcuReproducibility,
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
fn prepared<T: Format>(
    explicit: &mut impl PcuPreparedHostKernel,
    kind: u32,
    a: &[T],
    b: &[T],
    q: &mut [T],
    r: &mut [T],
) {
    let result = match kind {
        0 | 3 => explicit.call(&mut [
            PcuHostArgument::read_write(PcuBindingRef::new(0, 0), q),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
        ]),
        1 | 4 => explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[T]),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), q),
        ]),
        2 => explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), b),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
            PcuHostArgument::read(PcuBindingRef::new(0, 2), a),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), q),
        ]),
        _ => unreachable!(),
    };
    assert!(result.is_ok());
}
#[allow(clippy::too_many_lines)] // Keep genuinely matched source/IR/native boundaries and census together.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    kind: u32,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    guard();
    global::clear_thread_cache().unwrap();
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native = Native::new::<T, N>(backend, ir, 1);
    let banks = [
        oracle::inputs::<T>(N, 1, kind),
        oracle::inputs::<T>(N, 17, kind),
    ];
    let mut q = vec![T::SENTINEL; N + 2];
    let mut r = q.clone();
    for (a, b, wq, wr) in &banks {
        host(a, b, &mut q, &mut r).unwrap();
        oracle::verify(wq, wr, &q, &r);
        prepared(&mut explicit, kind, a, b, &mut q, &mut r);
        oracle::verify(wq, wr, &q, &r);
        native.host(a, b, &mut q, &mut r);
        oracle::verify(wq, wr, &q, &r);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "cuda_joint_div_rem_operands/{:?}/{:?}/{}/{kind}/{N}",
        ir.numerical_requirements.numerical_mode,
        ir.numerical_requirements.numerical_options.reproducibility,
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
            let (a, b, wq, wr) = &banks[bank];
            let started = Instant::now();
            match route {
                "actual_source" => host(a, b, &mut q, &mut r).unwrap(),
                "prepared_ir" => prepared(&mut explicit, kind, a, b, &mut q, &mut r),
                _ => native.host(a, b, &mut q, &mut r),
            }
            let elapsed = started.elapsed();
            oracle::verify(wq, wr, &q, &r);
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
                "census/{:?}/{:?}/{}/{kind}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}",
                ir.numerical_requirements.numerical_mode,
                ir.numerical_requirements.numerical_options.reproducibility,
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
#[inline(never)] // Per-width cold isolation preserves the default cache bound.
fn width<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    requirements: PcuImplementationRequirements,
) {
    macro_rules! entry {
        ($bindings:ident,$builder:ident,$kind:literal,$call:expr) => {{
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
        repeated_bindings,
        __repeated_ir_with_float_underflow_policy,
        0,
        |a, _b, q, r| source::repeated::<T, N>(q, r, a)
    );
    entry!(
        unread_bindings,
        __unread_ir_with_float_underflow_policy,
        1,
        |a, _b, q, r| source::unread::<T, N>(&[], r, a, q)
    );
    entry!(
        reordered_bindings,
        __reordered_ir_with_float_underflow_policy,
        2,
        |a, b, q, r| source::reordered::<T, N>(b, r, a, q)
    );
    entry!(
        indexed_zero_bindings,
        __indexed_zero_ir_with_float_underflow_policy,
        3,
        |a, _b, q, r| source::indexed_zero::<T, N>(q, r, a)
    );
    entry!(
        grid_zero_bindings,
        __grid_zero_ir_with_float_underflow_policy,
        4,
        |a, _b, q, r| source::grid_zero::<T, N>(&[], r, a, q)
    );
}
pub fn run(criterion: &mut Criterion) {
    let (_, backend, _) = super::selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        for reproducibility in [
            PcuReproducibility::Unspecified,
            PcuReproducibility::PortableV1,
        ] {
            let requirements = PcuImplementationRequirements {
                numerical_mode: mode,
                numerical_options: fusion_pcu::PcuNumericalOptions {
                    reproducibility,
                    ..Default::default()
                },
                ..PcuImplementationRequirements::DEFAULT
            };
            global::configure(global::PcuExecutionPolicy {
                backend: global::PcuBackendChoice::Cuda,
                device: Some(backend.device_identity().device_id()),
                numerical_mode: mode,
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
