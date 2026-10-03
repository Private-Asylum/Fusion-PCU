//! Four genuine caller routes over cold-frozen read roles; no unneeded input storage.
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/operand_schema/graph/graph.rs"]
mod graph;
#[path = "../../tests/operand_schema/source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use pcu_facade::{global,PcuCheckedFloat,PcuFloatUnderflowPolicy,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuHostKernelBackend,PcuHostArgument,PcuPreparedHostKernel};
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,criterion_group,criterion_main};
use fusion_pcu_cpu::PcuCpuHostBackend;
use std::sync::atomic::{AtomicUsize, Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn bits<T: PcuCheckedFloat>(actual: &[T], expected: &[T]) {
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.encode_le().as_ref(), b.encode_le().as_ref());
    }
}
fn operation<T: PcuCheckedFloat>(
    schema: usize,
    lhs: T,
    rhs: T,
) -> Result<T, pcu_facade::PcuExecutionFaultKind> {
    let uf = PcuFloatUnderflowPolicy::IeeeAfterRounding;
    match schema {
        0 => lhs.pcu_checked_add_with_policy(rhs, uf),
        1 => lhs.pcu_checked_mul_with_policy(rhs, uf),
        2 => lhs.pcu_checked_sub_with_policy(rhs, uf),
        _ => lhs.pcu_checked_div_with_policy(rhs, uf),
    }
}
#[allow(clippy::too_many_lines)] // Five real source signatures share one matched validation/publication/census boundary.
fn width<T: PcuCheckedFloat, const N: usize>(
    criterion: &mut Criterion,
    finite: impl Fn(f32) -> T + Copy,
) {
    let backend = PcuCpuHostBackend::scalar();
    let mut one = source::one_prepare::<T, N, _>(&backend).unwrap();
    let mut repeated = source::repeated_prepare::<T, N, _>(&backend).unwrap();
    let mut reordered = source::reordered_prepare::<T, N, _>(&backend).unwrap();
    let mut independent = source::independent_prepare::<T, N, _>(&backend).unwrap();
    let mut grid = source::grid_prepare::<T, N, _>(&backend).unwrap();
    let empty: [T; 0] = [];
    let sentinel = finite(16.0);
    let raw: [Vec<f32>; 2] = std::array::from_fn(|bank| {
        (0..N)
            .map(|i| [1.0, 0.5, 2.0, 4.0][i % 4] * if bank == 0 { 1.0 } else { 0.5 })
            .collect()
    });
    let inputs = raw
        .each_ref()
        .map(|v| v.iter().copied().map(finite).collect::<Vec<_>>());
    for schema in 0..5 {
        let wants = raw.each_ref().map(|v| {
            v.iter()
                .map(|x| {
                    finite(match schema {
                        0 => x + x,
                        1 => x * x,
                        2 => 0.0,
                        3 => x / v[0],
                        _ => 1.0,
                    })
                })
                .collect::<Vec<_>>()
        });
        let mut plan = graph::with(T::TYPE, u32::try_from(N).unwrap(), schema, |kernel| {
            backend.prepare_host_kernel(kernel).unwrap()
        });
        let mut output = vec![sentinel; N + 3];
        let mut run = |route: usize, bank: usize| {
            let input = &inputs[bank];
            output.fill(sentinel);
            match route {
                0 => match schema {
                    0 => one(input, &mut output),
                    1 => repeated(input, &empty, &mut output),
                    2 => reordered(&mut output, &empty, input),
                    3 => independent(input, &mut output),
                    _ => grid(&empty, &mut output, input),
                }
                .unwrap(),
                1 => match schema {
                    0 => source::one::<T, N>(input, &mut output),
                    1 => source::repeated::<T, N>(input, &empty, &mut output),
                    2 => source::reordered::<T, N>(&mut output, &empty, input),
                    3 => source::independent::<T, N>(input, &mut output),
                    _ => source::grid::<T, N>(&empty, &mut output, input),
                }
                .unwrap(),
                2 => {
                    if schema == 0 || schema == 3 {
                        plan.call(&mut [
                            PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                            PcuHostArgument::read(graph::INPUT, input),
                        ])
                    } else {
                        plan.call(&mut [
                            PcuHostArgument::read(graph::UNUSED, &empty),
                            PcuHostArgument::read_write(graph::OUTPUT, &mut output),
                            PcuHostArgument::read(graph::INPUT, input),
                        ])
                    }
                    .unwrap();
                }
                _ => {
                    // Match the checked executor's two-pass host publication, even for valid controls.
                    for (index, lhs) in input.iter().copied().enumerate() {
                        let rhs = if schema == 3 { input[0] } else { input[index] };
                        operation(schema, lhs, rhs).unwrap();
                    }
                    for (index, lhs) in input.iter().copied().enumerate() {
                        let rhs = if schema == 3 { input[0] } else { input[index] };
                        output[index] = operation(schema, lhs, rhs).unwrap();
                    }
                }
            }
            bits(&output[..N], &wants[bank]);
            bits(&output[N..], &[sentinel; 3]);
        };
        for route in 0..4 {
            run(route, 0);
            run(route, 1);
        }
        let scores = SCORES.load(Ordering::Relaxed);
        let mut group =
            criterion.benchmark_group(format!("cpu_operand_schema/{:?}/{schema}", T::TYPE));
        for (route, label) in [
            "prepared_annotated",
            "ordinary_annotated",
            "explicit_graph",
            "native_checked",
        ]
        .into_iter()
        .enumerate()
        {
            let counts = ffi::count_heap(|| {
                for i in 0..64 {
                    run(route, i % 2);
                }
            });
            assert_eq!(
                (counts.allocations, counts.reallocations, counts.frees),
                (0, 0, 0)
            );
            assert_eq!(SCORES.load(Ordering::Relaxed), scores);
            println!(
                "census {:?}/{N}/{schema}/{label}:64 changing calls 0alloc 0realloc 0free",
                T::TYPE
            );
            group.bench_function(BenchmarkId::new(label, N), |bench| {
                let mut bank = 0;
                bench.iter(|| {
                    bank ^= 1;
                    run(route, std::hint::black_box(bank));
                });
            });
        }
        group.finish();
    }
}
fn benchmarks(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    macro_rules! sizes {
        ($t:ty,$make:expr) => {
            width::<$t, 1>(criterion, $make);
            width::<$t, 65>(criterion, $make);
        };
    }
    sizes!(f32, |v| v);
    sizes!(f64, f64::from);
    sizes!(PcuF16Bits, |v| PcuF16Bits::pcu_checked_from_f32(v).unwrap());
    sizes!(PcuBf16Bits, |v| PcuBf16Bits::pcu_checked_from_f32(v)
        .unwrap());
    sizes!(PcuF8E4M3FnBits, |v| PcuF8E4M3FnBits::pcu_checked_from_f32(
        v
    )
    .unwrap());
    sizes!(PcuF8E5M2Bits, |v| PcuF8E5M2Bits::pcu_checked_from_f32(v)
        .unwrap());
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
