//! Genuine annotated resident calls, explicit plans and independent GLSL/ash public publishers.
#[path = "../owned_leaf/bytes/bytes.rs"]
mod bytes;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../scalar_transport/ffi/ffi.rs"]
#[allow(dead_code)]
// This peer uses independent resident control and heap counters, not host-only controls.
mod ffi;
#[path = "../../../cpu/tests/scalar_tensor/source/source.rs"]
#[allow(dead_code)]
mod owned;
#[path = "../../tests/scalar_transport/sample/sample.rs"]
mod sample;
#[path = "../../tests/scalar_transport/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{
    BenchmarkId,
    Criterion,
    Throughput,
    criterion_group,
    criterion_main,
};
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuF16Bits,
    PcuBf16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuF128Bits,
    PcuF256Bits,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
    PcuHostArgument,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanArgument,
    PcuVulkanBackend,
};
#[rustfmt::skip]
use sample::{
    same,
    Sample,
};
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}

#[allow(clippy::too_many_lines)] // One matched group retains all three independent roots and exact read/prefix/tail boundaries.
fn carrier<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    let banks: [Vec<T>; 2] = std::array::from_fn(|bank| {
        (0..N)
            .map(|i| T::pattern(i * 37 + bank * 113 + 19))
            .collect()
    });
    let sentinel = T::pattern(91);
    let initial = vec![sentinel; N + 3];
    let source_inputs = banks.each_ref().map(|bank| owned::identity(bank).unwrap());
    let mut source_output = owned::identity(&initial).unwrap();
    let graph_inputs = banks
        .each_ref()
        .map(|bank| backend.upload_owned(bank).unwrap());
    let mut graph_output = backend.upload_owned(&initial).unwrap();
    let bindings = source::dense_bindings::<T>();
    let graph = source::dense_ir::<T, N>(&bindings).unwrap();
    let mut prepared = backend.prepare_mixed_kernel(&graph.ir()).unwrap();
    let mut native = ffi::resident::NativeResident::new(
        identity,
        u32::try_from(N).unwrap(),
        T::HOST_SIZE,
        [bytes::read(&banks[0]), bytes::read(&banks[1])],
        bytes::read(&initial),
    )
    .unwrap();
    let mut observed = initial.clone();
    for host_input in [false, true] {
        let label = if host_input {
            "host_to_owned"
        } else {
            "owned_to_owned"
        };
        let mut run = |route, bank: usize| {
            match route {
                0 => {
                    if host_input {
                        source::dense::<T, N>(&banks[bank], &mut source_output).unwrap();
                    } else {
                        source::dense::<T, N>(&source_inputs[bank], &mut source_output).unwrap();
                    }
                    source_output.read_into(&mut observed).unwrap();
                }
                1 => {
                    let input = if host_input {
                        PcuVulkanArgument::host(PcuHostArgument::read(
                            PcuBindingRef::new(0, 0),
                            &banks[bank],
                        ))
                    } else {
                        graph_inputs[bank].read_argument(PcuBindingRef::new(0, 0))
                    };
                    prepared
                        .call(&mut [input, graph_output.write_argument(PcuBindingRef::new(0, 1))])
                        .unwrap();
                    graph_output.read_into(&mut observed).unwrap();
                }
                2 => {
                    if host_input {
                        native.call_host(bytes::read(&banks[bank])).unwrap();
                    } else {
                        native.call_owned(bank).unwrap();
                    }
                    native.read(bytes::write(&mut observed));
                }
                _ => unreachable!(),
            }
            same(&observed[..N], &banks[bank]);
            same(&observed[N..], &initial[N..]);
        };
        let mut group =
            criterion.benchmark_group(format!("vulkan_resident/{:?}/{label}/{N}", T::TYPE));
        group.throughput(Throughput::Bytes(u64::try_from(N * T::HOST_SIZE).unwrap()));
        for (route, name) in ["source_ordinary", "explicit_graph", "independent_native"]
            .into_iter()
            .enumerate()
        {
            run(route, 0);
            run(route, 1);
            let scores = SCORES.load(Ordering::Relaxed);
            let counts = ffi::count_heap(|| {
                for bank in 0..64 {
                    run(route, bank & 1);
                }
            });
            assert_eq!(
                (counts.allocations, counts.reallocations, counts.frees),
                (0, 0, 0)
            );
            assert_eq!(SCORES.load(Ordering::Relaxed), scores);
            println!(
                "caller census {:?}/{label}/{N}/{name}:64 changing bank calls,{counts:?}; retained public destination, exact readback/tails included; SDK/API counts unknown",
                T::TYPE
            );
            group.bench_function(BenchmarkId::new(name, N), |bench| {
                bench.iter(|| run(route, std::hint::black_box(1)));
            });
        }
        group.finish();
    }
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
fn benchmarks(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    macro_rules! carriers {($($ty:ty),+ $(,)?)=>{$(carrier::<$ty,65>(criterion,&backend,identity);carrier::<$ty,257>(criterion,&backend,identity);)+};}
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
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
