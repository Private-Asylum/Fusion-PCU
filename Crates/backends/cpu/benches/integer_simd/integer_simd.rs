//! Eight native checked Add/Sub formats, genuine source and complete independent native transactions.
#[path = "../low_precision/ffi/ffi.rs"]
mod heap;
#[path = "native/native.rs"]
mod native;
#[path = "source/source.rs"]
#[allow(dead_code)] // Grid/broadcast source is exercised by the integration target.
mod source;
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,criterion_group,criterion_main};
#[rustfmt::skip]
use fusion_pcu_cpu::{PcuCpuProcessor,PcuCpuImplementation,PcuCpuCheckedInteger,PcuCpuHostBackend};
#[rustfmt::skip]
use pcu_facade::{global,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuDispatchIntegerBinaryOp,PcuRangePolicy,PcuExecutionFault};
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
fn implementation(processor: PcuCpuProcessor) -> PcuCpuImplementation {
    if processor.features().avx512f && processor.features().avx512bw {
        PcuCpuImplementation::Avx512
    } else if processor.features().avx2 {
        PcuCpuImplementation::Avx2
    } else if processor.features().sse2 {
        PcuCpuImplementation::Sse2
    } else {
        assert!(processor.features().neon);
        PcuCpuImplementation::Neon
    }
}
#[allow(clippy::too_many_lines)] // Full transaction, actual changing banks and all four source/native call boundaries remain adjacent.
fn compare<T: native::Native, const N: usize>(
    c: &mut Criterion,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut scalar: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let mut left = [T::small(10); N];
    let right = [T::small(2); N];
    let sentinel = T::small(77);
    let endpoint = if op == PcuDispatchIntegerBinaryOp::Add {
        T::maximum()
    } else {
        T::minimum()
    };
    let changed = endpoint
        .evaluate(
            T::small(1),
            if op == PcuDispatchIntegerBinaryOp::Add {
                PcuDispatchIntegerBinaryOp::Sub
            } else {
                PcuDispatchIntegerBinaryOp::Add
            },
        )
        .unwrap();
    let mut output = std::vec![sentinel;N+3];
    let mut expected = output.clone();
    let mut invoke = |route, left: &[T], out: &mut [T]| match route {
        0 => prepared(left, &right, out),
        1 => ordinary(left, &right, out),
        2 => scalar(left, &right, out),
        _ => native::execute::<T, N>(left, &right, out, op, range),
    };
    for phase in 0..2 {
        left[0] = if range == PcuRangePolicy::Clamp {
            if phase == 0 { endpoint } else { changed }
        } else {
            T::small(10 + phase)
        };
        if N > 1 {
            left[N - 1] = T::small(10 + phase);
        }
        expected.fill(sentinel);
        let notice = native::execute::<T, N>(&left, &right, &mut expected, op, range);
        for route in 0..4 {
            output.fill(sentinel);
            assert_eq!(
                std::hint::black_box(invoke(
                    route,
                    std::hint::black_box(&left),
                    std::hint::black_box(&mut output)
                )),
                notice
            );
            assert_eq!(output, expected);
        }
    }
    let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed);
    let mut group = c.benchmark_group(format!("cpu_integer_simd/{:?}/{op:?}/{range:?}", T::TYPE));
    for (route, label) in [
        "annotated_prepared_simd",
        "ordinary_source_simd",
        "explicit_scalar_ir",
        "native_checked_transaction",
    ]
    .into_iter()
    .enumerate()
    {
        let counts = heap::count_heap(|| {
            for phase in 0..64 {
                left[0] = if range == PcuRangePolicy::Clamp {
                    if phase % 2 == 0 { endpoint } else { changed }
                } else {
                    T::small(10 + u8::try_from(phase % 2).unwrap())
                };
                if N > 1 {
                    left[N - 1] = T::small(10 + u8::try_from(phase % 2).unwrap());
                }
                expected.fill(sentinel);
                let notice = std::hint::black_box(native::execute::<T, N>(
                    std::hint::black_box(&left),
                    std::hint::black_box(&right),
                    std::hint::black_box(&mut expected),
                    op,
                    range,
                ));
                output.fill(sentinel);
                assert_eq!(
                    std::hint::black_box(invoke(
                        route,
                        std::hint::black_box(&left),
                        std::hint::black_box(&mut output)
                    )),
                    notice
                );
                assert_eq!(output, expected);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(std::sync::atomic::Ordering::Relaxed), scores);
        println!(
            "census {:?}/{op:?}/{range:?}/{N}/{label}:64changing calls0Rustheap/norescore/fullpayload/tails",
            T::TYPE
        );
        let mut phase = 0;
        group.bench_function(BenchmarkId::new(label, N), |b| {
            b.iter(|| {
                phase ^= 1;
                left[0] = if range == PcuRangePolicy::Clamp {
                    if phase == 0 { endpoint } else { changed }
                } else {
                    T::small(10 + phase)
                };
                if N > 1 {
                    left[N - 1] = T::small(10 + phase);
                }
                std::hint::black_box(invoke(
                    route,
                    std::hint::black_box(&left),
                    std::hint::black_box(&mut output),
                ))
                .unwrap_or_else(|fault| {
                    assert!(fault.recovered);
                });
            });
        });
    }
    group.finish();
}
fn width<T: native::Native, const N: usize>(c: &mut Criterion) {
    let processor = PcuCpuProcessor::detect();
    let simd = PcuCpuCheckedInteger::<T>::with_implementation(processor, implementation(processor))
        .unwrap();
    macro_rules! workload {
        ($entry:ident,$prepare:ident,$bindings:ident,$ir:ident,$op:ident,$range:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(&simd).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut scalar = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            compare::<T, N>(
                c,
                PcuDispatchIntegerBinaryOp::$op,
                PcuRangePolicy::$range,
                move |a, b, o| {
                    prepared(a, b, o).map_err(|error| match error {
                        fusion_pcu_cpu::PcuCpuCheckedIntegerError::Fault(f) => f,
                        other => panic!("unexpected SIMD failure{other:?}"),
                    })
                },
                |a, b, o| {
                    source::$entry::<T, N>(a, b, o)
                        .map_err(|error| error.arithmetic_fault().unwrap())
                },
                move |a, b, o| {
                    scalar
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), a),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), b),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), o),
                        ])
                        .map_err(|error| error.fault().unwrap())
                },
            );
        }};
    }
    workload!(add, add_prepare, add_bindings, add_ir, Add, Reject);
    workload!(sub, sub_prepare, sub_bindings, sub_ir, Sub, Reject);
    workload!(
        add_clamp,
        add_clamp_prepare,
        add_clamp_bindings,
        add_clamp_ir,
        Add,
        Clamp
    );
    workload!(
        sub_clamp,
        sub_clamp_prepare,
        sub_clamp_bindings,
        sub_clamp_ir,
        Sub,
        Clamp
    );
}
fn benchmark(c: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    macro_rules! widths{($($ty:ty),*)=>{$(width::<$ty,1>(c);width::<$ty,257>(c);)*};}
    widths!(i8, u8, i16, u16, i32, u32, i64, u64);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
