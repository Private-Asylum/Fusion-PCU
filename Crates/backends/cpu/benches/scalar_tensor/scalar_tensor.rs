//! Matched raw-carrier escaping owners: ordinary annotation, indexed graph and native copy.
extern crate pcu_facade as fusion_pcu;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/scalar_broadcast/sample/sample.rs"]
mod sample;
#[path = "../../tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use std::{rc::Rc,sync::atomic::{AtomicUsize,Ordering}};
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorElement};
use fusion_pcu_cpu::PcuCpuPreparedScalarTensorGraph;
use sample::{Sample, same};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
struct Output<T> {
    data: Vec<T>,
    _shape: Rc<[usize]>,
}
#[allow(clippy::significant_drop_tightening)] // Criterion consumes the completed group after its matched ownership closures.
fn width<T: Sample + TensorElement, const N: usize>(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let banks: [Vec<T>; 2] =
        std::array::from_fn(|bank| (0..N).map(|i| T::pattern(i + 13 + bank * 173)).collect());
    let mut graph = Graph::default();
    let input = graph.input([N], T::TYPE).unwrap();
    let mut plan = PcuCpuPreparedScalarTensorGraph::<T>::prepare(&graph, &[input]).unwrap();
    drop(graph);
    let shape: Rc<[usize]> = Rc::from(plan.output_bindings()[0].shape.as_slice());
    let sentinel = T::pattern(17);
    let mut observed = vec![sentinel; N + 3];
    let mut run = |route: usize, bank: usize| {
        let input = &banks[bank];
        if route == 0 {
            let owner = source::identity(input).unwrap();
            owner.read_into(&mut observed).unwrap();
            drop(owner);
        } else {
            let data = if route == 1 {
                plan.execute(&[input]).unwrap();
                plan.output(0).unwrap().to_vec()
            } else {
                input.clone()
            };
            let owner = Output {
                data,
                _shape: Rc::clone(&shape),
            };
            observed[..N].copy_from_slice(&owner.data);
            drop(owner);
        }
        same(&observed[..N], input);
        same(&observed[N..], &[sentinel; 3]);
    };
    for route in 0..3 {
        run(route, 0);
        run(route, 1);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!("cpu_scalar_tensor/{:?}", T::TYPE));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for (route, label) in [
        "ordinary_annotated_owned_read",
        "explicit_leaf_graph_owned_read",
        "native_copy_owned_read",
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
            "census {:?}/{N}/{label}:64 changing calls64 allocations0 reallocations64 frees",
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
fn benchmarks(criterion: &mut Criterion) {
    macro_rules! formats {($($ty:ty),+)=>{$(width::<$ty,1>(criterion);width::<$ty,65>(criterion);)+};}
    formats!(
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
        PcuI256,
        PcuU256,
        PcuI512,
        PcuU512,
        PcuF16Bits,
        PcuBf16Bits,
        PcuF8E4M3FnBits,
        PcuF8E5M2Bits,
        f32,
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
