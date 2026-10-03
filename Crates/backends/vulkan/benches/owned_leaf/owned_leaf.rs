//! Actual annotated owned calls, retained leaf diagnostics, and independent ash owner controls.
#[path = "bytes/bytes.rs"]
mod bytes;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../../cpu/benches/low_precision/ffi/ffi.rs"]
mod heap;
#[path = "../../tests/scalar_transport/sample/sample.rs"]
mod sample;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: heap::CountingAllocator = heap::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits};
#[rustfmt::skip]
use pcu_facade::dialect::tensor::{Graph,TensorElement};
#[rustfmt::skip]
use fusion_pcu_vulkan::{PcuVulkanBackend,PcuVulkanPreparedScalarTensorGraph,PcuVulkanTensorInput};
use sample::{Sample, same};
static SCORES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}
fn score_invocation(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    0
}

#[allow(clippy::significant_drop_tightening)] // Completed group owns matched closures through Criterion.
#[allow(clippy::too_many_lines)] // One six-route group keeps its identical owner/read/drop and census checks adjacent.
fn carrier<T: Sample + TensorElement, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_device: score,
        score_invocation: Some(score_invocation),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let banks: [Vec<T>; 2] = std::array::from_fn(|bank| {
        (0..N)
            .map(|index| T::pattern(index + bank * 179 + 17))
            .collect()
    });
    let source_inputs = banks.each_ref().map(|bank| source::identity(bank).unwrap());
    let graph_inputs = banks
        .each_ref()
        .map(|bank| backend.upload_owned(bank).unwrap());
    let mut graph = Graph::default();
    let input = graph.input([N], T::TYPE).unwrap();
    let mut plan =
        PcuVulkanPreparedScalarTensorGraph::<T>::prepare(backend, &graph, &[input]).unwrap();
    drop(graph);
    let mut native = ffi::NativeCopy::new(identity, N * T::HOST_SIZE).unwrap();
    let native_inputs = banks
        .each_ref()
        .map(|bank| native.copy_host(bytes::read(bank)).unwrap());
    let sentinel = T::pattern(71);
    let mut observed = vec![sentinel; N + 3];
    let mut run = |route, bank: usize| {
        match route {
            0 | 3 => {
                let owner = if route == 0 {
                    source::identity(&banks[bank]).unwrap()
                } else {
                    source::identity(&source_inputs[bank]).unwrap()
                };
                owner.read_into(&mut observed).unwrap();
                drop(owner);
            }
            1 | 4 => {
                let input = if route == 1 {
                    PcuVulkanTensorInput::Host(banks[bank].as_slice())
                } else {
                    PcuVulkanTensorInput::Owned(&graph_inputs[bank])
                };
                let owner = plan.execute_owned(&[input]).unwrap();
                owner.read_into(&mut observed).unwrap();
                drop(owner);
            }
            2 | 5 => {
                let owner = if route == 2 {
                    native.copy_host(bytes::read(&banks[bank])).unwrap()
                } else {
                    native.copy_owned(&native_inputs[bank]).unwrap()
                };
                owner.read(bytes::write(&mut observed[..N]));
                drop(owner);
            }
            _ => unreachable!(),
        }
        same(&observed[..N], &banks[bank]);
        same(&observed[N..], &[sentinel; 3]);
    };
    for route in 0..6 {
        run(route, 0);
        run(route, 1);
    }
    let scores = SCORES.load(std::sync::atomic::Ordering::Relaxed);
    let mut group = criterion.benchmark_group(format!("vulkan_owned_leaf/{:?}", T::TYPE));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for (route, label) in [
        "ordinary_annotated_host_owned_read",
        "explicit_leaf_host_owned_read",
        "native_host_owned_read",
        "ordinary_annotated_borrowed_owned_read",
        "explicit_leaf_borrowed_owned_read",
        "native_borrowed_owned_read",
    ]
    .into_iter()
    .enumerate()
    {
        let counts = heap::count_heap(|| {
            for bank in 0..64 {
                run(route, bank % 2);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(std::sync::atomic::Ordering::Relaxed), scores);
        println!(
            "census {:?}/{N}/{label}:64 changing calls0 Rust allocations0 reallocations0 frees; fresh native output allocation/read/release included",
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
    let (backend, identity) = device::selected();
    macro_rules! carriers {($($ty:ty),+)=>{$(carrier::<$ty,1>(criterion,&backend,identity);carrier::<$ty,65>(criterion,&backend,identity);)+};}
    carriers!(
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
