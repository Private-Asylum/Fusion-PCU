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
use fusion_pcu_rocm::RocmOwnedDispatchBackend;
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
        super::activity::guard();
    }
}
#[allow(clippy::too_many_lines)] // Keep all matched physical boundaries together.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
    operation: u32,
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
    let banks = [
        oracle::inputs::<T>(N, 1, operation),
        oracle::inputs::<T>(N, 17, operation),
    ];
    let inputs = banks
        .each_ref()
        .map(|(input, _)| source::identity(input).unwrap());
    let mut q = vec![T::sentinel(); N + 2];
    let mut rq = source::identity(q.as_slice()).unwrap();
    let mut bad_divisor = banks[0].0.clone();
    bad_divisor[5] = T::from(T::MAX + 1);
    native_host.upload(0, &bad_divisor);
    assert_eq!(native_host.submit(0), (5 << 3) | 5);
    native_resident.upload(0, &bad_divisor);
    assert_eq!(native_resident.submit(0), (5 << 3) | 5);
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
        assert_eq!(native_resident.submit(bank), u64::MAX);
        native_resident.read_resident(&mut q);
        oracle::verify(expected, &q);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    for resident_boundary in [false, true] {
        guard();
        let mut group = criterion.benchmark_group(format!(
            "rocm_low_unary/{}/{}/{:?}/{}",
            T::LABEL,
            ["neg", "relu"][usize::try_from(operation).unwrap()],
            ir.numerical_requirements.range_policy,
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
                            assert_eq!(native_resident.submit(bank), u64::MAX);
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
                fusion_pcu_rocm::reset_rocm_api_census();
                let (_, counts) = super::allocations::measure(&mut call);
                let api = fusion_pcu_rocm::rocm_api_census();
                assert_eq!(api.symbol_resolutions, 0);
                assert_eq!(api.module_loads, 0);
                assert_eq!(api.allocations, 0);
                assert_eq!(api.frees, 0);
                assert_eq!(api.kernel_launches, 1);
                assert_eq!(api.device_selections * 2, api.runtime_calls);
                eprintln!(
                    "api/{}/{operation}/{:?}/{resident_boundary}/{N}/{label}: {api:?}",
                    T::LABEL,
                    ir.numerical_requirements.range_policy
                );

                eprintln!(
                    "census/{}/{operation}/{:?}/{resident_boundary}/{N}/{label}: alloc={} realloc={} frees={} bytes={}",
                    T::LABEL,
                    ir.numerical_requirements.range_policy,
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
        backend: global::PcuBackendChoice::Rocm,
        device: Some(backend.device_identity().device_id()),
        block_size: 256,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    macro_rules! width {
        ($ty:ty,$n:expr) => {{
            neg_reject_ieee::<$ty, $n>(criterion, &backend);
            relu_reject_ieee::<$ty, $n>(criterion, &backend);
            neg_clamp_strict::<$ty, $n>(criterion, &backend);
            relu_clamp_strict::<$ty, $n>(criterion, &backend);
        }};
    }
    width!(fusion_pcu::PcuF16Bits, 65);
    width!(fusion_pcu::PcuF16Bits, 65536);
    width!(fusion_pcu::PcuBf16Bits, 65);
    width!(fusion_pcu::PcuBf16Bits, 65536);
    width!(fusion_pcu::PcuF8E4M3FnBits, 65);
    width!(fusion_pcu::PcuF8E4M3FnBits, 65536);
    width!(fusion_pcu::PcuF8E5M2Bits, 65);
    width!(fusion_pcu::PcuF8E5M2Bits, 65536);
    global::clear_thread_cache().unwrap();
}

#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn neg_reject_ieee<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) {
    let bindings = source::neg_reject_ieee_bindings::<T>();
    let builder = source::neg_reject_ieee_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        0,
        &builder.ir(),
        |a, out| source::neg_reject_ieee::<T, N>(a, out),
        |a, out| source::neg_reject_ieee::<T, N>(a, out),
    );
}

#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn relu_reject_ieee<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) {
    let bindings = source::relu_reject_ieee_bindings::<T>();
    let builder = source::relu_reject_ieee_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        1,
        &builder.ir(),
        |a, out| source::relu_reject_ieee::<T, N>(a, out),
        |a, out| source::relu_reject_ieee::<T, N>(a, out),
    );
}

#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn neg_clamp_strict<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) {
    let bindings = source::neg_clamp_strict_bindings::<T>();
    let builder = source::neg_clamp_strict_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        0,
        &builder.ir(),
        |a, out| source::neg_clamp_strict::<T, N>(a, out),
        |a, out| source::neg_clamp_strict::<T, N>(a, out),
    );
}

#[inline(never)] // Keep each cold IR arena out of the enumeration frame.
fn relu_clamp_strict<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &RocmOwnedDispatchBackend,
) {
    let bindings = source::relu_clamp_strict_bindings::<T>();
    let builder = source::relu_clamp_strict_ir::<T, N>(&bindings).unwrap();
    case::<T, N>(
        criterion,
        backend,
        1,
        &builder.ir(),
        |a, out| source::relu_clamp_strict::<T, N>(a, out),
        |a, out| source::relu_clamp_strict::<T, N>(a, out),
    );
}
