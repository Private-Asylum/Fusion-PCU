//! Actual source / explicit IR / same native kernel, changing complete host/resident calls.
#[rustfmt::skip]
use std::{
    time::{Duration, Instant},
    sync::atomic::{AtomicUsize, Ordering},
};
#[rustfmt::skip]
use criterion::{Criterion, Throughput};
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
use super::{native::Native, oracle::{self, Integer}, source};
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
    let cold = Instant::now();
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native_host = Native::new::<T, N>(backend, ir, 1);
    let mut native_resident = Native::new::<T, N>(backend, ir, 2);
    eprintln!(
        "cold/{}/{N}: explicit+two native fixtures={:?}",
        T::LABEL,
        cold.elapsed()
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
            "cuda_checked_div_rem/{}/{}",
            T::LABEL,
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
                let (_, counts) = super::allocations::measure(&mut call);
                let api = fusion_pcu_cuda::cuda_api_census();
                assert_eq!(api.symbol_resolutions, 0);
                assert_eq!(api.module_loads, 0);
                assert_eq!(api.allocations, 0);
                assert_eq!(api.frees, 0);
                assert_eq!(api.kernel_launches, 1);
                eprintln!("api/{}/{resident_boundary}/{N}/{label}: {api:?}", T::LABEL);
                eprintln!(
                    "census/{}/{resident_boundary}/{N}/{label}: alloc={} realloc={} frees={} bytes={}",
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
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cuda,
        device: Some(backend.device_identity().device_id()),
        block_size: 256,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    macro_rules! width {
        ($module:ident, $ty:ty, $count:expr) => {{
            let bindings = source::$module::direct_bindings();
            let builder = source::$module::direct_ir::<$count>(&bindings).unwrap();
            case::<$ty, $count>(
                criterion,
                &backend,
                &builder.ir(),
                |lhs, rhs, q, r| source::$module::direct::<$count>(lhs, rhs, q, r),
                |lhs, rhs, q, r| source::$module::direct::<$count>(lhs, rhs, q, r),
            );
        }};
    }
    macro_rules! sizes {
        ($module:ident, $ty:ty) => {
            width!($module, $ty, 65);
            width!($module, $ty, 65536);
        };
    }
    sizes!(signed8, i8);
    sizes!(unsigned8, u8);
    sizes!(signed16, i16);
    sizes!(unsigned16, u16);
    sizes!(signed32, i32);
    sizes!(unsigned32, u32);
    sizes!(signed64, i64);
    sizes!(unsigned64, u64);
    global::clear_thread_cache().unwrap();
}
