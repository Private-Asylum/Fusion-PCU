//! Actual source / explicit IR / same native kernel, changing complete host/resident calls.
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
    PcuTensor,
};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
#[rustfmt::skip]
use super::{
    native::Native,
    oracle::{
        self,
        Integer,
    },
    source,
};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn guard() {
    // Semantic --test runs emit no estimates; statistical runs retain the unchanged activity gate.
    if !std::env::args().any(|argument| argument == "--test") {
        super::activity::activity_guard();
    }
}
#[allow(clippy::too_many_lines)] // Keep all matched physical boundaries together.
fn case<T: Integer, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
    mut resident: impl FnMut(
        &PcuTensor<T>,
        &PcuTensor<T>,
        &mut PcuTensor<T>,
        &mut PcuTensor<T>,
    ) -> Result<(), PcuExecutionError>,
) {
    guard();
    #[cfg(feature = "allocation-census")]
    fusion_pcu_cuda::reset_cuda_api_census();
    let cold = Instant::now();
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native_host = Native::new::<T, N>(backend, ir, 1);
    let mut native_resident = Native::new::<T, N>(backend, ir, 2);
    eprintln!(
        "cold/{}/{N}: explicit+two native fixtures={:?}",
        T::LABEL,
        cold.elapsed()
    );
    #[cfg(feature = "allocation-census")]
    eprintln!(
        "cold preparation/{}/N{N} API={:?}; private status word8bytes per prepared/native instance",
        T::LABEL,
        fusion_pcu_cuda::cuda_api_census()
    );
    let banks = [oracle::inputs::<T>(N, 1), oracle::inputs::<T>(N, 17)];
    let inputs = banks.each_ref().map(|(lhs, rhs)| {
        [
            source::identity(lhs).unwrap(),
            source::identity(rhs).unwrap(),
        ]
    });
    let mut q = vec![T::SENTINEL; N + 2];
    let mut r = q.clone();
    let mut rq = source::identity(q.as_slice()).unwrap();
    let mut rr = source::identity(r.as_slice()).unwrap();
    let mut bad_divisor = banks[0].1.clone();
    bad_divisor[5] = T::ZERO;
    native_host.upload(0, &banks[0].0, &bad_divisor);
    assert_eq!(native_host.submit(0), (5 << 3) | 1);
    native_resident.upload(0, &banks[0].0, &bad_divisor);
    assert_eq!(native_resident.submit(0), (5 << 3) | 1);
    for (bank, (lhs, rhs)) in banks.iter().enumerate() {
        native_resident.upload(bank, lhs, rhs);
        host(lhs, rhs, &mut q, &mut r).unwrap();
        oracle::verify(lhs, rhs, &q, &r);
        explicit
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
                PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut r),
            ])
            .unwrap();
        oracle::verify(lhs, rhs, &q, &r);
        native_host.host(lhs, rhs, &mut q, &mut r);
        oracle::verify(lhs, rhs, &q, &r);
        resident(&inputs[bank][0], &inputs[bank][1], &mut rq, &mut rr).unwrap();
        rq.read_into(&mut q).unwrap();
        rr.read_into(&mut r).unwrap();
        oracle::verify(lhs, rhs, &q, &r);
        assert_eq!(native_resident.submit(bank), u64::MAX);
        native_resident.read_resident(&mut q, &mut r);
        oracle::verify(lhs, rhs, &q, &r);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    for resident_boundary in [false, true] {
        guard();
        let mut group = criterion.benchmark_group(format!(
            "cuda_wide_div_rem/{}/{:?}/inv{}/{}",
            T::LABEL,
            ir.numerical_requirements.numerical_mode,
            ir.entry.logical_shape[0],
            if resident_boundary {
                "resident"
            } else {
                "full_host"
            }
        ));
        group.sample_size(20);
        group.warm_up_time(Duration::from_millis(500));
        group.measurement_time(Duration::from_secs(2));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for route in 0..if resident_boundary { 2 } else { 3 } {
            let label = if route == 0 {
                "source"
            } else if !resident_boundary && route == 1 {
                "explicit_ir"
            } else {
                "native_same_kernel"
            };
            let mut bank = 0;
            let mut call = || {
                let mut elapsed = Duration::ZERO;
                {
                    bank ^= 1;
                    let (lhs, rhs) = &banks[bank];
                    let started = Instant::now();
                    if resident_boundary {
                        if route == 0 {
                            resident(&inputs[bank][0], &inputs[bank][1], &mut rq, &mut rr).unwrap();
                        } else {
                            assert_eq!(native_resident.submit(bank), u64::MAX);
                        }
                    } else if route == 0 {
                        host(lhs, rhs, &mut q, &mut r).unwrap();
                    } else if route == 1 {
                        explicit
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
                                PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut r),
                            ])
                            .unwrap();
                    } else {
                        native_host.host(lhs, rhs, &mut q, &mut r);
                    }
                    elapsed += started.elapsed();
                    if resident_boundary {
                        if route == 0 {
                            rq.read_into(&mut q).unwrap();
                            rr.read_into(&mut r).unwrap();
                        } else {
                            native_resident.read_resident(&mut q, &mut r);
                        }
                    }
                    oracle::verify(lhs, rhs, &q, &r);
                }
                elapsed
            };
            let _ = call();
            #[cfg(feature = "allocation-census")]
            {
                fusion_pcu_cuda::reset_cuda_api_census();
                let ((), counts) = super::allocations::measure(|| {
                    for _ in 0..64 {
                        let _ = call();
                    }
                });
                let api = fusion_pcu_cuda::cuda_api_census();
                assert_eq!(api.symbol_resolutions, 0);
                assert_eq!(api.module_loads, 0);
                assert_eq!(api.allocations, 0);
                assert_eq!(api.frees, 0);
                assert_eq!(api.kernel_launches, 64);
                eprintln!(
                    "census/{}/{resident_boundary}/{N}/{label}: 64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}",
                    T::LABEL,
                    counts.alloc_calls,
                    counts.realloc_calls,
                    counts.dealloc_calls,
                    counts.requested_bytes
                );
            }
            #[cfg(not(feature = "allocation-census"))]
            group.bench_function(format!("{label}/{N}"), |bench| {
                bench.iter_custom(|iterations| (0..iterations).map(|_| call()).sum::<Duration>());
            });
        }
        group.finish();
    }
    assert_eq!(
        SCORES.load(Ordering::Relaxed),
        scores,
        "warm source reranked providers"
    );
}
pub fn run(criterion: &mut Criterion) {
    let (_, backend, _) = super::selection::selected_device();
    for mode in [
        fusion_pcu::PcuNumericalMode::Boundary,
        fusion_pcu::PcuNumericalMode::Strict,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cuda,
            device: Some(backend.device_identity().device_id()),
            block_size: 256,
            numerical_mode: mode,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        let requirements = fusion_pcu::PcuImplementationRequirements {
            numerical_mode: mode,
            ..fusion_pcu::PcuImplementationRequirements::DEFAULT
        };
        macro_rules! width {
            ($ty:ty, $count:expr) => {{
                macro_rules! entry {
                    ($builder:ident,$function:ident) => {{
                        let bindings = source::direct_bindings::<$ty>();
                        let builder = source::$builder::<$ty, $count>(
                            &bindings,
                            requirements.float_underflow,
                            requirements.range_policy,
                            requirements,
                        )
                        .unwrap();
                        builder.with_ir(|ir| {
                            case::<$ty, $count>(
                                criterion,
                                &backend,
                                ir,
                                |lhs, rhs, q, r| source::$function::<$ty, $count>(lhs, rhs, q, r),
                                |lhs, rhs, q, r| source::$function::<$ty, $count>(lhs, rhs, q, r),
                            )
                        });
                    }};
                }
                entry!(__direct_ir_with_float_underflow_policy, direct);
                entry!(__grid_ir_with_float_underflow_policy, grid);
            }};
        }
        macro_rules! sizes {
            ($ty:ty) => {
                width!($ty, 65);
                width!($ty, 4096);
            };
        }
        sizes!(i128);
        sizes!(u128);
        sizes!(fusion_pcu::PcuI256);
        sizes!(fusion_pcu::PcuU256);
        sizes!(fusion_pcu::PcuI512);
        sizes!(fusion_pcu::PcuU512);
        global::clear_thread_cache().unwrap();
    }
}
