//! Matched escaping source/graph/native ownership; one fresh output allocation, zero warm scratch allocation.
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/wide_integer/oracle/oracle.rs"]
#[allow(dead_code)]
mod oracle;
#[path = "../../tests/integer_tensor/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use std::{hint::black_box,rc::Rc,sync::atomic::{AtomicUsize,Ordering}};
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuDispatchIntegerBinaryOp,PcuI256,PcuU256,PcuI512,PcuU512};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorElement};
use fusion_pcu_cpu::PcuCpuPreparedIntegerTensorGraph;
use oracle::Wide;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
struct Output<T> {
    data: Vec<T>,
    _shape: Rc<[usize]>,
}
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Matched escaping boundaries share their warm census; finish consumes the Criterion group.
fn cases<T: Wide + TensorElement, const N: usize>(criterion: &mut Criterion) {
    for op in [
        PcuDispatchIntegerBinaryOp::Add,
        PcuDispatchIntegerBinaryOp::Sub,
        PcuDispatchIntegerBinaryOp::Mul,
    ] {
        let mut input = vec![oracle::small::<T>(3); N];
        let right = vec![oracle::small::<T>(1); N];
        let sentinel = oracle::small::<T>(77);
        let mut observed = vec![sentinel; N + 3];
        let mut g = Graph::default();
        let a = g.input([N], T::TYPE).unwrap();
        let b = g.input([N], T::TYPE).unwrap();
        let out = match op {
            PcuDispatchIntegerBinaryOp::Add => g.add(a, b),
            PcuDispatchIntegerBinaryOp::Sub => g.sub(a, b),
            PcuDispatchIntegerBinaryOp::Mul => g.mul(a, b),
        }
        .unwrap();
        let mut graph = PcuCpuPreparedIntegerTensorGraph::<T>::prepare(&g, &[out]).unwrap();
        let shape: Rc<[usize]> = Rc::from(graph.output_bindings()[0].shape.as_slice());
        let mut run = |route: usize, input: &[T], observed: &mut [T]| {
            if route == 0 {
                let owner = match op {
                    PcuDispatchIntegerBinaryOp::Add => source::add(input, &right),
                    PcuDispatchIntegerBinaryOp::Sub => source::sub(input, &right),
                    PcuDispatchIntegerBinaryOp::Mul => source::mul(input, &right),
                }
                .unwrap();
                owner.read_into(observed).unwrap();
                drop(owner);
            } else {
                let data = if route == 1 {
                    graph.execute(&[input, &right]).unwrap();
                    graph.output(0).unwrap().to_vec()
                } else {
                    input
                        .iter()
                        .zip(&right)
                        .map(|(&a, &b)| {
                            match op {
                                PcuDispatchIntegerBinaryOp::Add => a.pcu_checked_add(b),
                                PcuDispatchIntegerBinaryOp::Sub => a.pcu_checked_sub(b),
                                PcuDispatchIntegerBinaryOp::Mul => a.pcu_checked_mul(b),
                            }
                            .unwrap()
                        })
                        .collect()
                };
                let owner = Output {
                    data,
                    _shape: Rc::clone(&shape),
                };
                observed[..N].copy_from_slice(&owner.data);
                drop(owner);
            }
        };
        for route in 0..3 {
            for raw in [3, 4] {
                input[0] = oracle::small(raw);
                run(route, &input, &mut observed);
                assert_eq!(
                    observed[0],
                    oracle::evaluate(input[0], right[0], op).unwrap()
                );
                assert_eq!(observed[N..], [sentinel; 3]);
            }
        }
        let scores = SCORES.load(Ordering::Relaxed);
        let mut group =
            criterion.benchmark_group(format!("cpu_integer_tensor/{:?}/{op:?}", T::TYPE));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for (route, label) in [
            "ordinary_annotated_owned_read",
            "explicit_graph_owned_read",
            "native_checked_loop_owned_read",
        ]
        .into_iter()
        .enumerate()
        {
            let counts = ffi::count_heap(|| {
                for bank in 0..64 {
                    input[0] = oracle::small(3 + u8::try_from(bank % 2).unwrap());
                    run(route, &input, &mut observed);
                }
            });
            assert_eq!(
                (counts.allocations, counts.reallocations, counts.frees),
                (64, 0, 64)
            );
            println!(
                "census {label} {:?} {op:?} N{N} calls64 allocations64 reallocations0 frees64 one_fresh_output_per_call",
                T::TYPE
            );
            group.bench_function(BenchmarkId::new(label, N), |bench| {
                bench.iter(|| {
                    input[0] = if input[0] == oracle::small(3) {
                        oracle::small(4)
                    } else {
                        oracle::small(3)
                    };
                    run(route, black_box(&input), black_box(&mut observed));
                    black_box(&observed);
                });
            });
        }
        group.finish();
        assert_eq!(SCORES.load(Ordering::Relaxed), scores);
    }
}
#[allow(clippy::too_many_lines)] // Fourteen exact scalar monomorphizations at two extents remain explicit.
fn benchmark(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    cases::<i8, 1>(criterion);
    cases::<i8, 65>(criterion);
    cases::<u8, 1>(criterion);
    cases::<u8, 65>(criterion);
    cases::<i16, 1>(criterion);
    cases::<i16, 65>(criterion);
    cases::<u16, 1>(criterion);
    cases::<u16, 65>(criterion);
    cases::<i32, 1>(criterion);
    cases::<i32, 65>(criterion);
    cases::<u32, 1>(criterion);
    cases::<u32, 65>(criterion);
    cases::<i64, 1>(criterion);
    cases::<i64, 65>(criterion);
    cases::<u64, 1>(criterion);
    cases::<u64, 65>(criterion);
    cases::<i128, 1>(criterion);
    cases::<i128, 65>(criterion);
    cases::<u128, 1>(criterion);
    cases::<u128, 65>(criterion);
    cases::<PcuI256, 1>(criterion);
    cases::<PcuI256, 65>(criterion);
    cases::<PcuU256, 1>(criterion);
    cases::<PcuU256, 65>(criterion);
    cases::<PcuI512, 1>(criterion);
    cases::<PcuI512, 65>(criterion);
    cases::<PcuU512, 1>(criterion);
    cases::<PcuU512, 65>(criterion);
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
