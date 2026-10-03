//! Genuine source/native owners include fresh output allocation, terminal read and destruction.
#[path = "../../tests/scalar_transport/sample/sample.rs"]
#[allow(dead_code)]
// Arbitrary carrier generators belong to their existing independent transport cohort.
mod bits;
#[path = "../owned_leaf/bytes/bytes.rs"]
mod bytes;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../owned_leaf/ffi/ffi.rs"]
#[allow(dead_code)]
// Transfer owner exposes additional standalone leaf controls, exercised by its own target.
mod ffi;
#[path = "../../../cpu/benches/low_precision/ffi/ffi.rs"]
mod heap;
#[path = "../../tests/checked_tensor/sample/sample.rs"]
mod sample;
#[path = "../../tests/checked_tensor/source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
#[rustfmt::skip]
use criterion::{
    Criterion,
    BenchmarkId,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuFloatUnderflowPolicy,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
};
use pcu_facade::dialect::tensor::Graph;
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
};
use sample::Sample;
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // One matched six-route owner group keeps equal boundaries and census assertions adjacent.
fn run<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
    op: u32,
    policy: PcuFloatUnderflowPolicy,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        float_underflow: policy,
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let a: [Vec<T>; 2] =
        std::array::from_fn(|phase| vec![T::small(if phase == 0 { 3 } else { 4 }); N]);
    let b: [Vec<T>; 2] =
        std::array::from_fn(|phase| vec![T::small(if phase == 0 { 1 } else { 2 }); N]);
    let want: [T; 2] = std::array::from_fn(|phase| {
        T::small(match (op, phase) {
            (0, 0) | (4, 1) => 4,
            (0, _) => 6,
            (2..=4, 0) => 3,
            (2, _) => 8,
            (1 | 3, _) => 2,
            _ => unreachable!(),
        })
    });
    let expected = want.map(|value| vec![value; N]);
    let source_a = a.each_ref().map(|bank| source::retain(bank).unwrap());
    let source_b = b.each_ref().map(|bank| source::retain(bank).unwrap());
    let graph_a = a.each_ref().map(|bank| backend.upload_owned(bank).unwrap());
    let graph_b = b.each_ref().map(|bank| backend.upload_owned(bank).unwrap());
    let mut graph = Graph::default();
    let x = graph.input([N], T::TYPE).unwrap();
    let y = graph.input([N], T::TYPE).unwrap();
    let output = match op {
        0 => graph.add(x, y),
        1 => graph.sub(x, y),
        2 => graph.mul(x, y),
        3 => graph.div(x, y),
        4 => graph.relu(x),
        _ => unreachable!(),
    }
    .unwrap();
    if T::FLOAT {
        graph
            .set_value_float_underflow_policy(output, policy)
            .unwrap();
    }
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
    drop(graph);
    let mut native = ffi::pointwise::NativePointwise::new(
        identity,
        T::TYPE,
        u32::try_from(N).unwrap(),
        op,
        match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
    )
    .unwrap();
    let native_a = a
        .each_ref()
        .map(|bank| native.upload(bytes::read(bank)).unwrap());
    let native_b = b
        .each_ref()
        .map(|bank| native.upload(bytes::read(bank)).unwrap());
    let sentinel = T::small(11);
    let mut observed = vec![sentinel; N + 3];
    let mut invoke = |route, bank: usize| {
        match route {
            0 | 3 => {
                let owner = if route == 0 {
                    source::call(op, a[bank].as_slice(), b[bank].as_slice()).unwrap()
                } else {
                    source::call(op, &source_a[bank], &source_b[bank]).unwrap()
                };
                owner.read_into(&mut observed).unwrap();
                drop(owner);
            }
            1 | 4 => {
                let args = if route == 1 {
                    [
                        PcuVulkanTensorInput::Host(a[bank].as_slice()),
                        PcuVulkanTensorInput::Host(b[bank].as_slice()),
                    ]
                } else {
                    [
                        PcuVulkanTensorInput::Owned(&graph_a[bank]),
                        PcuVulkanTensorInput::Owned(&graph_b[bank]),
                    ]
                };
                let owner = plan
                    .execute_owned(&args[..if op == 4 { 1 } else { 2 }])
                    .unwrap();
                owner.read_into(&mut observed).unwrap();
                drop(owner);
            }
            2 | 5 => {
                let owner = if route == 2 {
                    native
                        .host(bytes::read(&a[bank]), bytes::read(&b[bank]))
                        .unwrap()
                } else {
                    native.borrowed(&native_a[bank], &native_b[bank]).unwrap()
                };
                owner.read(bytes::write(&mut observed[..N]));
                drop(owner);
            }
            _ => unreachable!(),
        }
        bits::same(&observed[..N], &expected[bank]);
        bits::same(&observed[N..], &[sentinel; 3]);
    };
    for route in 0..6 {
        invoke(route, 0);
        invoke(route, 1);
    }
    let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "vulkan_owned_pointwise/{:?}/{op}/{policy:?}",
        T::TYPE
    ));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for (route, label) in [
        "source_host_owner_read",
        "graph_host_owner_read",
        "native_host_owner_read",
        "source_borrowed_owner_read",
        "graph_borrowed_owner_read",
        "native_borrowed_owner_read",
    ]
    .into_iter()
    .enumerate()
    {
        let counts = heap::count_heap(|| {
            for bank in 0..64 {
                invoke(route, bank % 2);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(std::sync::atomic::Ordering::Relaxed), scores);
        println!(
            "census {:?}/{N}/{op}/{policy:?}/{label}:64changing calls0Rustheap; freshnative output/read/release included",
            T::TYPE
        );
        group.bench_function(BenchmarkId::new(label, N), |bench| {
            let mut bank = 0;
            bench.iter(|| {
                bank ^= 1;
                invoke(route, std::hint::black_box(bank));
            });
        });
    }
    group.finish();
}

fn carrier<T: Sample>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    for op in 0..if T::FLOAT { 5 } else { 3 } {
        for policy in [
            PcuFloatUnderflowPolicy::IeeeAfterRounding,
            PcuFloatUnderflowPolicy::RejectSubnormalResult,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        ] {
            if !T::FLOAT && policy != PcuFloatUnderflowPolicy::IeeeAfterRounding {
                continue;
            }
            run::<T, 1>(criterion, backend, identity, op, policy);
            run::<T, 65>(criterion, backend, identity, op, policy);
        }
    }
}
fn benchmarks(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    macro_rules! types {($($ty:ty),+)=>{$(carrier::<$ty>(criterion,&backend,identity);)+};}
    types!(
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
        f64
    );
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
