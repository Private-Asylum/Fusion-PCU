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
    grid: bool,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionError>,
    mut resident: impl FnMut(&PcuTensor<T>, &mut PcuTensor<T>) -> Result<(), PcuExecutionError>,
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
    let inputs = banks
        .each_ref()
        .map(|(input, _)| source::identity(input).unwrap());
    let mut q = vec![T::sentinel(); N + 2];
    let mut rq = source::identity(q.as_slice()).unwrap();
    for (bank, (input, expected)) in banks.iter().enumerate() {
        native_resident.upload(bank, input);
        host(input, &mut q).unwrap();
        oracle::verify(expected, &q);
        explicit
            .call(&mut [
                PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut q),
            ])
            .unwrap();
        oracle::verify(expected, &q);
        native_host.host(input, &mut q);
        oracle::verify(expected, &q);
        resident(&inputs[bank], &mut rq).unwrap();
        rq.read_into(&mut q).unwrap();
        oracle::verify(expected, &q);
        native_resident.submit(bank);
        native_resident.read_resident(&mut q);
        oracle::verify(expected, &q);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    for resident_boundary in [false, true] {
        guard();
        let mut group = criterion.benchmark_group(format!(
            "cuda_wide_transport/{}/{}/{N}/{}",
            T::LABEL,
            if grid { "grid" } else { "direct" },
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
                    let (input, expected) = &banks[bank];
                    let started = Instant::now();
                    if resident_boundary {
                        if route == 0 {
                            resident(&inputs[bank], &mut rq).unwrap();
                        } else {
                            native_resident.submit(bank);
                        }
                    } else if route == 0 {
                        host(input, &mut q).unwrap();
                    } else if route == 1 {
                        explicit
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut q),
                            ])
                            .unwrap();
                    } else {
                        native_host.host(input, &mut q);
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
                let (_, counts) = super::allocations::measure(&mut call);
                let api = fusion_pcu_cuda::cuda_api_census();
                assert_eq!(api.symbol_resolutions, 0);
                assert_eq!(api.module_loads, 0);
                assert_eq!(api.allocations, 0);
                assert_eq!(api.frees, 0);
                assert_eq!(api.kernel_launches, 1);
                eprintln!(
                    "api/{}/{grid}/{resident_boundary}/{N}/{label}: {api:?}",
                    T::LABEL
                );
                eprintln!(
                    "census/{}/{grid}/{resident_boundary}/{N}/{label}: alloc={} realloc={} frees={} bytes={}",
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
        ($ty:ty) => {{
            direct::<$ty, 65>(criterion, &backend);
            grid::<$ty, 65>(criterion, &backend);
            direct::<$ty, 65536>(criterion, &backend);
            grid::<$ty, 65536>(criterion, &backend);
        }};
    }
    width!(i128);
    width!(u128);
    width!(fusion_pcu::PcuI256);
    width!(fusion_pcu::PcuU256);
    width!(fusion_pcu::PcuI512);
    width!(fusion_pcu::PcuU512);
    width!(fusion_pcu::PcuF128Bits);
    width!(fusion_pcu::PcuF256Bits);
    global::clear_thread_cache().unwrap();
}
#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn direct<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
) {
    let bindings = source::direct_bindings::<T>();
    let builder = source::direct_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        false,
        &builder.ir(),
        |a, out| source::direct::<T, N>(a, out),
        |a, out| source::direct::<T, N>(a, out),
    );
}
#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn grid<T: Format, const N: usize>(criterion: &mut Criterion, backend: &CudaOwnedDispatchBackend) {
    let bindings = source::grid_bindings::<T>();
    let builder = source::grid_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        true,
        &builder.ir(),
        |a, out| source::grid::<T, N>(a, out),
        |a, out| source::grid::<T, N>(a, out),
    );
}
