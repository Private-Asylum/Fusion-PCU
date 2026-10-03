//! Matched escaping ordinary source, frozen graph and direct native checked-loop owners.
extern crate pcu_facade as fusion_pcu;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../../rocm/benches/low_tensor/oracle/oracle.rs"]
mod oracle;
#[path = "../../tests/low_tensor/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use std::{rc::Rc,sync::atomic::{AtomicUsize,Ordering}};
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuFloatUnderflowPolicy,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuTensor,PcuExecutionError};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorElement};
use fusion_pcu_cpu::PcuCpuPreparedTensorGraph;
use oracle::Format;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
struct Output<T> {
    data: Vec<T>,
    _shape: Rc<[usize]>,
}
fn owned<T: Format>(left: &[T], right: &[T], op: u32) -> Result<PcuTensor<T>, PcuExecutionError> {
    match op {
        0 => source::add(left, right),
        1 => source::sub(left, right),
        2 => source::mul(left, right),
        3 => source::div(left, right),
        _ => source::relu(left),
    }
}
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // One matched ownership/census boundary; Criterion finish consumes its group.
fn width<T: Format + TensorElement, const N: usize>(criterion: &mut Criterion) {
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
    ] {
        global::configure(global::PcuExecutionPolicy {
            backend: global::PcuBackendChoice::Cpu,
            float_underflow: policy,
            score_device: score,
            ..Default::default()
        })
        .unwrap();
        global::clear_thread_cache().unwrap();
        for op in 0..5 {
            let banks = [
                oracle::inputs::<T>(N, 1, op, policy),
                oracle::inputs::<T>(N, 17, op, policy),
            ];
            let mut graph = Graph::default();
            let left = graph.input([N], T::TYPE).unwrap();
            let right = graph.input([N], T::TYPE).unwrap();
            let output = match op {
                0 => graph.add(left, right),
                1 => graph.sub(left, right),
                2 => graph.mul(left, right),
                3 => graph.div(left, right),
                _ => graph.relu(left),
            }
            .unwrap();
            graph
                .set_value_float_underflow_policy(output, policy)
                .unwrap();
            let mut plan = PcuCpuPreparedTensorGraph::<T>::prepare(&graph, &[output]).unwrap();
            let shape: Rc<[usize]> = Rc::from(plan.output_bindings()[0].shape.as_slice());
            let mut observed = vec![T::sentinel(); N + 3];
            let mut run = |route: usize, bank: usize| {
                let (left, right, _) = &banks[bank];
                if route == 0 {
                    let output = owned(left, right, op).unwrap();
                    output.read_into(&mut observed).unwrap();
                    drop(output);
                } else {
                    let data = if route == 1 {
                        if op == 4 {
                            plan.execute(&[left]).unwrap();
                        } else {
                            plan.execute(&[left, right]).unwrap();
                        }
                        plan.output(0).unwrap().to_vec()
                    } else {
                        left.iter()
                            .zip(right)
                            .map(|(&left, &right)| {
                                match op {
                                    0 => left.pcu_checked_add_with_policy(right, policy),
                                    1 => left.pcu_checked_sub_with_policy(right, policy),
                                    2 => left.pcu_checked_mul_with_policy(right, policy),
                                    3 => left.pcu_checked_div_with_policy(right, policy),
                                    _ => left.pcu_checked_relu_with_policy(policy),
                                }
                                .unwrap()
                            })
                            .collect()
                    };
                    let output = Output {
                        data,
                        _shape: Rc::clone(&shape),
                    };
                    observed[..N].copy_from_slice(&output.data);
                    drop(output);
                }
                assert_eq!(&observed[..N], &banks[bank].2);
                assert_eq!(&observed[N..], &[T::sentinel(); 3]);
            };
            for route in 0..3 {
                run(route, 0);
                run(route, 1);
            }
            let scores = SCORES.load(Ordering::Relaxed);
            let mut group =
                criterion.benchmark_group(format!("cpu_low_tensor/{:?}/{policy:?}/{op}", T::TYPE));
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
                        run(route, bank % 2);
                    }
                });
                assert_eq!(
                    (counts.allocations, counts.reallocations, counts.frees),
                    (64, 0, 64)
                );
                assert_eq!(SCORES.load(Ordering::Relaxed), scores);
                println!(
                    "census {:?}/{policy:?}/{op}/{N}/{label}:64 changing calls64 allocations0 reallocations64 frees",
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
}
fn benchmarks(criterion: &mut Criterion) {
    macro_rules! formats {($($ty:ty),+)=>{$(width::<$ty,1>(criterion);width::<$ty,65>(criterion);)+};}
    formats!(PcuF16Bits, PcuBf16Bits, PcuF8E4M3FnBits, PcuF8E5M2Bits);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
