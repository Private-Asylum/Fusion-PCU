//! Complete source/prepared/minimal-native host boundaries with changing exact bytes.
#[rustfmt::skip]
use std::{time::{Duration,Instant},sync::atomic::{AtomicUsize,Ordering}};
#[rustfmt::skip]
use fusion_pcu::{global,PcuBindingRef,PcuDispatchKernelIr,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel, PcuOwnedDispatchBackend,PcuExecutionError,PcuImplementationRequirements,PcuNumericalMode};
#[rustfmt::skip]
use criterion::{Criterion,Throughput};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
#[rustfmt::skip]
use super::{native::Native,oracle::{self,Format},source};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    geometry: u8,
    ir: &PcuDispatchKernelIr<'_>,
    mut host: impl FnMut(&[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    let mut native = Native::new::<T, N>(backend, ir);
    let banks = [3_u8, 127].map(|seed| {
        (0..N)
            .map(|lane| T::pattern(seed.wrapping_add(u8::try_from(lane % 256).unwrap())))
            .collect::<Vec<_>>()
    });
    let mut stage = vec![T::pattern(251); N + 2];
    let mut output = stage.clone();
    host(&banks[0], &mut stage, &mut output).unwrap();
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "cuda_ordered_transport/{:?}/{}/{geometry}/{N}",
        ir.numerical_requirements.numerical_mode,
        T::LABEL
    ));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in ["actual_source", "prepared_ir", "native_minimal_kernel"] {
        super::guard_owner();
        let mut bank = 0;
        let mut call = || {
            bank ^= 1;
            let input = &banks[bank];
            let started = Instant::now();
            match route {
                "actual_source" => host(input, &mut stage, &mut output).unwrap(),
                "prepared_ir" => prepared
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut [] as &mut [T]),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), &mut stage),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), &mut output),
                    ])
                    .unwrap(),
                _ => native.host(input, &mut stage, &mut output),
            }
            let elapsed = started.elapsed();
            oracle::verify(input, &stage);
            oracle::verify(input, &output);
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
                "census/{:?}/{}/{geometry}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; API={api:?}",
                ir.numerical_requirements.numerical_mode,
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
    request: PcuImplementationRequirements,
) {
    let bindings = source::direct_bindings::<T>();
    source::__direct_ir_with_float_underflow_policy::<T, N>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap()
    .with_ir(|ir| {
        case::<T, N>(criterion, backend, 0, ir, |input, stage, output| {
            source::direct::<T, N>(input, &mut [], stage, output)
        });
    });
    source::__grid_ir_with_float_underflow_policy::<T, N>(
        &bindings,
        request.float_underflow,
        request.range_policy,
        request,
    )
    .unwrap()
    .with_ir(|ir| {
        case::<T, N>(criterion, backend, 1, ir, |input, stage, output| {
            source::grid::<T, N>(input, &mut [], stage, output)
        });
    });
}
pub fn run(criterion: &mut Criterion) {
    let (_, backend, _) = super::selection::selected_device();
    for mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cuda,
            device: Some(backend.device_identity().device_id()),
            numerical_mode: mode,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        let request = PcuImplementationRequirements {
            numerical_mode: mode,
            ..PcuImplementationRequirements::DEFAULT
        };
        macro_rules! widths {($($ty:ty),+)=>{$(width::<$ty,65>(criterion,&backend,request);width::<$ty,4096>(criterion,&backend,request);)+};}
        widths!(
            i8,
            u8,
            i16,
            u16,
            i32,
            u32,
            i64,
            u64,
            i128,
            u128,
            fusion_pcu::PcuI256,
            fusion_pcu::PcuU256,
            fusion_pcu::PcuI512,
            fusion_pcu::PcuU512,
            fusion_pcu::PcuF16Bits,
            fusion_pcu::PcuBf16Bits,
            fusion_pcu::PcuF8E4M3FnBits,
            fusion_pcu::PcuF8E5M2Bits,
            f32,
            f64,
            fusion_pcu::PcuF128Bits,
            fusion_pcu::PcuF256Bits
        );
    }
    global::clear_thread_cache().unwrap();
}
