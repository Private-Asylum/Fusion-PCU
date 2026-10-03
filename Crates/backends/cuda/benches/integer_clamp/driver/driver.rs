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
use super::{native::Native, oracle::{self, Format}, source};
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
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    operation: u32,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionError>,
    mut resident: impl FnMut(
        &PcuTensor<T>,
        &PcuTensor<T>,
        &mut PcuTensor<T>,
    ) -> Result<(), PcuExecutionError>,
) {
    guard();
    // This sweep has more than the default64 specializations. Isolate each cold case;
    // measured calls retain the default bounded cache and may never silently rerank.
    global::clear_thread_cache().unwrap();
    let cold = Instant::now();
    let mut explicit = backend.prepare_host_kernel(ir).unwrap();
    let mut native_host = Native::new::<T, N>(backend, ir, 1);
    let mut native_resident = Native::new::<T, N>(backend, ir, 2);
    eprintln!(
        "cold/{}/{N}: explicit+two native fixtures={:?}",
        T::LABEL,
        cold.elapsed()
    );
    let banks = [
        oracle::inputs::<T>(N, 1, operation),
        oracle::inputs::<T>(N, 17, operation),
    ];
    let inputs = banks.each_ref().map(|(lhs, rhs, _, _)| {
        [
            source::identity(lhs).unwrap(),
            source::identity(rhs).unwrap(),
        ]
    });
    let mut q = vec![T::sentinel(); N + 2];
    let mut rq = source::identity(q.as_slice()).unwrap();
    for (bank, (lhs, rhs, expected, status)) in banks.iter().enumerate() {
        native_resident.upload(bank, lhs, rhs);
        oracle::result(&host(lhs, rhs, &mut q), *status);
        oracle::verify(expected, &q);
        let result = explicit.call(&mut [
            PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
            PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
        ]);
        oracle::result(
            &result.map_err(|error| match error {
                fusion_pcu_cuda::CudaHostKernelError::CheckedExecutionFault(fault) => {
                    fusion_pcu::PcuExecutionError::ArithmeticFault(fault)
                }
                other => panic!("{other:?}"),
            }),
            *status,
        );
        oracle::verify(expected, &q);
        native_host.host(lhs, rhs, &mut q, *status);
        oracle::verify(expected, &q);
        oracle::result(
            &resident(&inputs[bank][0], &inputs[bank][1], &mut rq),
            *status,
        );
        rq.read_into(&mut q).unwrap();
        oracle::verify(expected, &q);
        assert_eq!(native_resident.submit(bank), *status);
        native_resident.read_resident(&mut q);
        oracle::verify(expected, &q);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    for resident_boundary in [false, true] {
        guard();
        let mut group = criterion.benchmark_group(format!(
            "cuda_integer_clamp/{}/{}/{}",
            T::LABEL,
            ["add", "sub", "mul", "div"][usize::try_from(operation).unwrap()],
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
                    let (lhs, rhs, expected, status) = &banks[bank];
                    let started = Instant::now();
                    if resident_boundary {
                        if route == 0 {
                            oracle::result(
                                &resident(&inputs[bank][0], &inputs[bank][1], &mut rq),
                                *status,
                            );
                        } else {
                            assert_eq!(native_resident.submit(bank), *status);
                        }
                    } else if route == 0 {
                        oracle::result(&host(lhs, rhs, &mut q), *status);
                    } else if route == 1 {
                        let result = explicit.call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut q),
                        ]);
                        oracle::result(
                            &result.map_err(|error| match error {
                                fusion_pcu_cuda::CudaHostKernelError::CheckedExecutionFault(
                                    fault,
                                ) => fusion_pcu::PcuExecutionError::ArithmeticFault(fault),
                                other => panic!("{other:?}"),
                            }),
                            *status,
                        );
                    } else {
                        native_host.host(lhs, rhs, &mut q, *status);
                    }
                    elapsed += started.elapsed();
                    if resident_boundary {
                        if route == 0 {
                            rq.read_into(&mut q).unwrap();
                        } else {
                            native_resident.read_resident(&mut q);
                        }
                    }
                    oracle::verify(expected, &q);
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
                    "api/{}/{operation}/{resident_boundary}/{N}/{label}: {api:?}",
                    T::LABEL
                );
                eprintln!(
                    "census/{}/{operation}/{resident_boundary}/{N}/{label}: calls=64 alloc={} realloc={} frees={} bytes={}",
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
    macro_rules! widths {
        ($ty:ty) => {{
            add::<$ty, 65>(criterion, &backend);
            add::<$ty, 65536>(criterion, &backend);
            sub::<$ty, 65>(criterion, &backend);
            sub::<$ty, 65536>(criterion, &backend);
            mul::<$ty, 65>(criterion, &backend);
            mul::<$ty, 65536>(criterion, &backend);
        }};
    }
    widths!(i8);
    widths!(u8);
    widths!(i16);
    widths!(u16);
    widths!(i32);
    widths!(u32);
    widths!(i64);
    widths!(u64);
    widths!(i128);
    widths!(u128);
    widths!(fusion_pcu::PcuI256);
    widths!(fusion_pcu::PcuU256);
    widths!(fusion_pcu::PcuI512);
    widths!(fusion_pcu::PcuU512);
    global::clear_thread_cache().unwrap();
}
#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn add<T: Format, const N: usize>(criterion: &mut Criterion, backend: &CudaOwnedDispatchBackend) {
    let bindings = source::add::direct_bindings::<T>();
    let builder = source::add::direct_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        0,
        &builder.ir(),
        |a, b, out| source::add::direct::<T, N>(a, b, out),
        |a, b, out| source::add::direct::<T, N>(a, b, out),
    );
}
#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn sub<T: Format, const N: usize>(criterion: &mut Criterion, backend: &CudaOwnedDispatchBackend) {
    let bindings = source::sub::direct_bindings::<T>();
    let builder = source::sub::direct_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        1,
        &builder.ir(),
        |a, b, out| source::sub::direct::<T, N>(a, b, out),
        |a, b, out| source::sub::direct::<T, N>(a, b, out),
    );
}
#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn mul<T: Format, const N: usize>(criterion: &mut Criterion, backend: &CudaOwnedDispatchBackend) {
    let bindings = source::mul::direct_bindings::<T>();
    let builder = source::mul::direct_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        2,
        &builder.ir(),
        |a, b, out| source::mul::direct::<T, N>(a, b, out),
        |a, b, out| source::mul::direct::<T, N>(a, b, out),
    );
}
