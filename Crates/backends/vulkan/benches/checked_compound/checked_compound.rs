//! Actual annotated source, retained ordered graph and independent native owner boundaries.
extern crate pcu_facade as fusion_pcu;
#[path = "../owned_leaf/bytes/bytes.rs"]
mod bytes;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../owned_leaf/ffi/ffi.rs"]
#[allow(dead_code)] // Transfer and pointwise controls retain their separate complete targets.
mod ffi;
#[path = "../../../cpu/benches/low_precision/ffi/ffi.rs"]
mod heap;
#[path = "oracle/oracle.rs"]
#[allow(dead_code)] // Maximum/minimum fault fixtures are qualified by the owning test.
mod oracle;
#[path = "source/source.rs"]
#[allow(dead_code)] // Source chain/shape helpers have separate source and example qualification.
mod source;
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
#[rustfmt::skip]
use criterion::{
    Criterion,
    BenchmarkId,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuFloatUnderflowPolicy,
    PcuNumericalMode,
};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::Graph;
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanPreparedTensorGraph,
    PcuVulkanTensorInput,
};
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
fn same<T: oracle::Format>(actual: &[T], expected: &[T]) {
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.bits(), b.bits());
    }
}
fn words(bits: u64) -> [u32; 2] {
    let b = bits.to_le_bytes();
    [
        u32::from_le_bytes(b[..4].try_into().unwrap()),
        u32::from_le_bytes(b[4..].try_into().unwrap()),
    ]
}
#[allow(clippy::too_many_lines, clippy::significant_drop_tightening)] // Cold geometry and six matched escaped-owner boundaries remain beside census assertions.
fn run<T: oracle::Format>(
    criterion: &mut Criterion,
    backend: &fusion_pcu_vulkan::PcuVulkanBackend,
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
    let banks = [oracle::banks::<T>(0), oracle::banks::<T>(1)];
    let data = |bank: usize| {
        if op == 0 {
            (
                banks[bank].left.as_flattened(),
                banks[bank].right.as_flattened(),
            )
        } else {
            (
                banks[bank].weights.as_flattened(),
                banks[bank].gradient.as_flattened(),
            )
        }
    };
    let source_a = banks.each_ref().map(|b| {
        if op == 0 {
            source::retain_left(&b.left)
        } else {
            source::identity(&b.weights)
        }
        .unwrap()
    });
    let source_b = banks.each_ref().map(|b| {
        if op == 0 {
            source::retain_right(&b.right)
        } else {
            source::identity(&b.gradient)
        }
        .unwrap()
    });
    let graph_a =
        core::array::from_fn::<_, 2, _>(|bank| backend.upload_owned(data(bank).0).unwrap());
    let graph_b =
        core::array::from_fn::<_, 2, _>(|bank| backend.upload_owned(data(bank).1).unwrap());
    let mut graph = Graph::default();
    graph.set_numerical_mode(PcuNumericalMode::Strict);
    let a = graph
        .input(if op == 0 { [2, 3] } else { [2, 2] }, T::TYPE)
        .unwrap();
    let b = graph
        .input(if op == 0 { [3, 2] } else { [2, 2] }, T::TYPE)
        .unwrap();
    let output = match op {
        0 => graph.matmul(a, b),
        1 => graph
            .mean_squared_error_typed(
                graph.typed_view::<T>(a).unwrap(),
                graph.typed_view::<T>(b).unwrap(),
            )
            .map(pcu_facade::dialect::tensor::TensorValueId::erase),
        _ => graph.sgd_update(a, b, 0.5),
    }
    .unwrap();
    graph
        .set_value_float_underflow_policy(output, policy)
        .unwrap();
    let mut plan = PcuVulkanPreparedTensorGraph::<T>::prepare(backend, &graph, &[output]).unwrap();
    drop(graph);
    let factor = words(T::value(if op == 1 { 4.0 } else { 0.5 }).bits());
    let native_op = match op {
        0 => 1,
        1 => 2,
        _ => 0,
    };
    let constants = [
        native_op,
        match policy {
            PcuFloatUnderflowPolicy::IeeeAfterRounding => 0,
            PcuFloatUnderflowPolicy::RejectSubnormalResult => 1,
            PcuFloatUnderflowPolicy::AllowGradualUnderflow => 2,
        },
        2,
        if op == 0 { 3 } else { 4 },
        2,
        0,
        0,
        factor[0],
        factor[1],
    ];
    let count = if op == 1 { 1 } else { 4 };
    let mut native = ffi::pointwise::NativePointwise::compound(
        identity,
        T::TYPE,
        constants,
        [
            if op == 0 { 6 } else { 4 },
            if op == 0 { 6 } else { 4 },
            count,
        ],
    )
    .unwrap();
    let native_a =
        core::array::from_fn::<_, 2, _>(|bank| native.upload(bytes::read(data(bank).0)).unwrap());
    let native_b =
        core::array::from_fn::<_, 2, _>(|bank| native.upload(bytes::read(data(bank).1)).unwrap());
    let expected = |bank: usize| match op {
        0 => banks[bank].product.as_slice(),
        1 => core::slice::from_ref(&banks[bank].loss),
        _ => banks[bank].update.as_slice(),
    };
    assert_ne!(expected(0)[0].bits(), expected(1)[0].bits());
    let sentinel = T::value(117.0);
    let mut observed = [sentinel; 7];
    let mut invoke = |route, bank: usize| {
        match route {
            0 | 3 => {
                let output = if route == 0 {
                    match op {
                        0 => source::product(&banks[bank].left, &banks[bank].right),
                        1 => source::loss(&banks[bank].weights, &banks[bank].gradient),
                        _ => source::update(&banks[bank].weights, &banks[bank].gradient),
                    }
                } else {
                    match op {
                        0 => source::product(&source_a[bank], &source_b[bank]),
                        1 => source::loss(&source_a[bank], &source_b[bank]),
                        _ => source::update(&source_a[bank], &source_b[bank]),
                    }
                }
                .unwrap();
                output.read_into(&mut observed).unwrap();
                drop(output);
            }
            1 | 4 => {
                let args = if route == 1 {
                    [
                        PcuVulkanTensorInput::Host(data(bank).0),
                        PcuVulkanTensorInput::Host(data(bank).1),
                    ]
                } else {
                    [
                        PcuVulkanTensorInput::Owned(&graph_a[bank]),
                        PcuVulkanTensorInput::Owned(&graph_b[bank]),
                    ]
                };
                let output = plan.execute_owned(&args).unwrap();
                output.read_into(&mut observed).unwrap();
                drop(output);
            }
            2 | 5 => {
                let output = if route == 2 {
                    native.host(bytes::read(data(bank).0), bytes::read(data(bank).1))
                } else {
                    native.borrowed(&native_a[bank], &native_b[bank])
                }
                .unwrap();
                output.read(bytes::write(&mut observed[..count as usize]));
                drop(output);
            }
            _ => unreachable!(),
        }
        same(&observed[..count as usize], expected(bank));
        same(
            &observed[count as usize..],
            &[sentinel; 7][count as usize..],
        );
    };
    for route in 0..6 {
        invoke(route, 0);
        invoke(route, 1);
    }
    let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!(
        "vulkan_owned_compound/{:?}/{op}/{policy:?}",
        T::TYPE
    ));
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
            "census {:?}/{op}/{policy:?}/{label}:64changing calls0Rustheap; fresh native output/read/release included",
            T::TYPE
        );
        let mut bank = 0;
        group.bench_function(BenchmarkId::new(label, count), |bench| {
            bench.iter(|| {
                bank ^= 1;
                invoke(route, bank);
            });
        });
    }
    group.finish();
}
fn benchmark(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    for policy in [
        PcuFloatUnderflowPolicy::IeeeAfterRounding,
        PcuFloatUnderflowPolicy::AllowGradualUnderflow,
        PcuFloatUnderflowPolicy::RejectSubnormalResult,
    ] {
        for op in 0..3 {
            run::<f32>(criterion, &backend, identity, op, policy);
            run::<f64>(criterion, &backend, identity, op, policy);
        }
    }
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
