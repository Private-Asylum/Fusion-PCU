//! Matched retained source/prepared calls and independent owned raw SDK broadcasts.
#[rustfmt::skip]
use std::{
    time::{Duration,Instant},
    sync::atomic::{AtomicUsize,Ordering},
};
#[rustfmt::skip]
use fusion_pcu::{
    global,
    PcuBindingRef,
    PcuDispatchKernelIr,
    PcuExecutionError,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuNumericalMode,
    PcuOwnedDispatchBackend,
    PcuPreparedHostKernel,
};
#[rustfmt::skip]
use criterion::{Criterion,Throughput};
use fusion_pcu_cuda::CudaOwnedDispatchBackend;
#[rustfmt::skip]
use super::{native::Native,oracle::Format,source};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn verify<T: Format, const N: usize>(value: T, actual: &[T]) {
    assert_eq!(actual.len(), N + 2);
    let expected = value.encode_le();
    let sentinel = T::pattern(251).encode_le();
    for scalar in &actual[..N] {
        assert_eq!(scalar.encode_le().as_ref(), expected.as_ref());
    }
    for scalar in &actual[N..] {
        assert_eq!(scalar.encode_le().as_ref(), sentinel.as_ref());
    }
}
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Criterion group is explicitly finished before terminal owner witness.
fn case<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    geometry: u8,
    ir: &PcuDispatchKernelIr<'_>,
    semantics: bool,
    mut host: impl FnMut(&[T], &mut [T]) -> Result<(), PcuExecutionError>,
) {
    let resources = fusion_pcu::describe_scalar_transport_map::<4>(ir, T::TYPE).unwrap();
    assert_eq!(
        resources
            .resource(PcuBindingRef::new(0, 0))
            .unwrap()
            .minimum_read_elements,
        1
    );
    assert_eq!(
        resources
            .resource(PcuBindingRef::new(0, 1))
            .unwrap()
            .minimum_write_elements,
        u32::try_from(N).unwrap()
    );
    let mut prepared = backend.prepare_host_kernel(ir).unwrap();
    // All-one float payloads are NaNs; transport must retain every bit without arithmetic.
    let banks = [T::pattern(3), T::filled(255)]
        .map(|value| vec![value, T::filled(255), T::pattern(0), T::pattern(127)]);
    let mut native = Native::new::<T, N>(backend, ir, [&banks[0], &banks[1]]);
    #[cfg(feature = "allocation-census")]
    let direct_counter = native.counter();
    let mut output = vec![T::pattern(251); N + 2];
    host(&banks[0], &mut output).unwrap();
    verify::<T, N>(banks[0][0], &output);
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "cuda_flat_broadcast/{:?}/{}/{geometry}/{N}",
        ir.numerical_requirements.numerical_mode,
        T::LABEL
    ));
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for route in ["actual_source", "prepared_ir", "independent_pinned_sdk"] {
        if !semantics {
            super::activity::activity_guard();
        }
        #[cfg(feature = "allocation-census")]
        {
            fusion_pcu_cuda::reset_cuda_api_census();
            direct_counter.set(super::native::Api::default());
        }
        let mut bank = 0;
        let mut call = || {
            bank ^= 1;
            let input = &banks[bank];
            let started = Instant::now();
            match route {
                "actual_source" => host(input, &mut output).unwrap(),
                "prepared_ir" => prepared
                    .call(&mut [
                        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
                        PcuHostArgument::read_write(PcuBindingRef::new(0, 1), &mut output),
                    ])
                    .unwrap(),
                _ => native.call(bank),
            }
            let elapsed = started.elapsed();
            if route == "independent_pinned_sdk" {
                native.verify(input[0]);
            } else {
                verify::<T, N>(input[0], &output);
            }
            assert_eq!(
                SCORES.load(Ordering::Relaxed),
                scores,
                "warm source reranked"
            );
            elapsed
        };
        let _ = call();
        if semantics {
            for _ in 0..3 {
                let _ = call();
            }
            println!(
                "semantic/{:?}/{}/{geometry}/{N}/{route}: changing full bits, scalar span1, output tails, no warm scoring PASS",
                ir.numerical_requirements.numerical_mode,
                T::LABEL
            );
        }
        #[cfg(feature = "allocation-census")]
        {
            // Reset happened before warmup; use a separate 64-call exact delta below.
            let before = fusion_pcu_cuda::cuda_api_census();
            let direct_before = direct_counter.get();
            let ((), counts) = super::allocations::measure(|| {
                for _ in 0..64 {
                    std::hint::black_box(call());
                }
            });
            let after = fusion_pcu_cuda::cuda_api_census();
            assert_eq!(after.symbol_resolutions, before.symbol_resolutions);
            assert_eq!(after.module_loads, before.module_loads);
            assert_eq!(after.allocations, before.allocations);
            assert_eq!(after.frees, before.frees);
            eprintln!(
                "census/{:?}/{}/{geometry}/{N}/{route}/64-changing-calls: alloc={} realloc={} frees={} bytes={}; wrapped-before={before:?}; wrapped-after={after:?}; direct-SDK={:?}",
                ir.numerical_requirements.numerical_mode,
                T::LABEL,
                counts.alloc_calls,
                counts.realloc_calls,
                counts.dealloc_calls,
                counts.requested_bytes,
                direct_counter.get().delta(direct_before)
            );
        }
        #[cfg(not(feature = "allocation-census"))]
        if !semantics {
            group.bench_function(route, |bench| {
                bench.iter_custom(|iterations| (0..iterations).map(|_| call()).sum::<Duration>());
            });
        }
    }
    group.finish();
    if T::LABEL == "u8"
        && N == 65
        && geometry == 0
        && ir.numerical_requirements.numerical_mode == PcuNumericalMode::Boundary
        && semantics
    {
        native.known_terminal_retirement_witness();
    }
}
fn width<T: Format, const N: usize>(
    criterion: &mut Criterion,
    backend: &CudaOwnedDispatchBackend,
    request: PcuImplementationRequirements,
    semantics: bool,
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
        assert_eq!(ir.numerical_requirements, request);
        case::<T, N>(criterion, backend, 0, ir, semantics, |input, output| {
            source::direct::<T, N>(input, output)
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
        assert_eq!(ir.numerical_requirements, request);
        case::<T, N>(criterion, backend, 1, ir, semantics, |input, output| {
            source::grid::<T, N>(input, output)
        });
    });
}
pub fn run(criterion: &mut Criterion) {
    let semantics = std::env::var_os("PCU_FLAT_BROADCAST_SEMANTICS").is_some();
    if !semantics {
        super::activity::activity_guard();
    }
    let (_, backend, _) = super::selection::selected_device();
    for numerical_mode in [PcuNumericalMode::Boundary, PcuNumericalMode::Strict] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cuda,
            device: Some(backend.device_identity().device_id()),
            numerical_mode,
            score_invocation: Some(score),
            ..Default::default()
        })
        .unwrap();
        let request = PcuImplementationRequirements {
            numerical_mode,
            ..PcuImplementationRequirements::DEFAULT
        };
        macro_rules! widths {($($ty:ty),+)=>{$(width::<$ty,65>(criterion,&backend,request,semantics);width::<$ty,4096>(criterion,&backend,request,semantics);)+};}
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
