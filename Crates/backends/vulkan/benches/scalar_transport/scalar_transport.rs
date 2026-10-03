//! Genuine 22-carrier source, explicit graph and independently compiled native transfer peers.
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "ffi/ffi.rs"]
#[allow(dead_code)]
mod ffi;
#[path = "../../tests/scalar_transport/sample/sample.rs"]
mod sample;
#[path = "../../tests/scalar_transport/source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuStableDeviceIdentity,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
#[rustfmt::skip]
use std::sync::{OnceLock,atomic::{AtomicUsize,Ordering}};
use sample::{Sample, same};
use fusion_pcu_vulkan::PcuVulkanBackend;
static IDENTITY: OnceLock<PcuStableDeviceIdentity> = OnceLock::new();
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_eq!(candidate.facts.stable_identity.as_ref(), IDENTITY.get());
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
type Route<'a, T> = &'a mut dyn FnMut(&[T], &mut [T]);
fn compare<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    label: &str,
    broadcast: bool,
    mut routes: [Route<'_, T>; 4],
) {
    let mut input = [T::pattern(3); N];
    let sentinel = T::pattern(17);
    let mut output = [sentinel; N];
    let mut group =
        criterion.benchmark_group(format!("vulkan_transport/{:?}/{label}/{N}", T::TYPE));
    group.throughput(Throughput::Bytes(u64::try_from(N * T::HOST_SIZE).unwrap()));
    for (route, run) in ["source_prepared", "source_ordinary", "graph", "native"]
        .into_iter()
        .zip(routes.iter_mut())
    {
        run(&input, &mut output);
        let scores = SCORES.load(Ordering::Relaxed);
        let counts = ffi::count_heap(|| {
            for seed in 0..64 {
                for (index, value) in input.iter_mut().enumerate() {
                    *value = T::pattern(seed + index + 29);
                }
                run(&input, &mut output);
            }
        });
        assert_eq!(
            (counts.allocations, counts.reallocations, counts.frees),
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(Ordering::Relaxed), scores);
        if broadcast {
            same(&output, &[input[0]; N]);
        } else {
            same(&output, &input);
        }
        println!(
            "caller census {:?}/{label}/{N}/{route}:64 changing calls,{counts:?}",
            T::TYPE
        );
        group.bench_function(BenchmarkId::new(route, N), |bench| {
            bench.iter(|| {
                run(
                    std::hint::black_box(&input),
                    std::hint::black_box(&mut output),
                );
            });
        });
    }
    group.finish();
}
fn width<T: Sample, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    macro_rules! dense {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(backend).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            let mut native =
                ffi::NativeTransport::new(identity, u32::try_from(N).unwrap(), T::HOST_SIZE, false)
                    .unwrap();
            compare::<T, N>(
                criterion,
                stringify!($entry),
                false,
                [
                    &mut |i, o| prepared(i, o).unwrap(),
                    &mut |i, o| source::$entry::<T, N>(i, o).unwrap(),
                    &mut |i, o| {
                        graph
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), i),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), o),
                            ])
                            .unwrap()
                    },
                    &mut |i, o| native.call(ffi::bytes(i), ffi::bytes_mut(o)).unwrap(),
                ],
            );
        }};
    }
    dense!(dense, dense_prepare, dense_ir, dense_bindings);
    dense!(
        dense_grid,
        dense_grid_prepare,
        dense_grid_ir,
        dense_grid_bindings
    );
    macro_rules! broadcast {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(backend).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            let mut native =
                ffi::NativeTransport::new(identity, u32::try_from(N).unwrap(), T::HOST_SIZE, true)
                    .unwrap();
            compare::<T, N>(
                criterion,
                stringify!($entry),
                true,
                [
                    &mut |i, o| prepared(&i[0], o).unwrap(),
                    &mut |i, o| source::$entry::<T, N>(&i[0], o).unwrap(),
                    &mut |i, o| {
                        graph
                            .call(&mut [
                                PcuHostArgument::read(PcuBindingRef::new(0, 0), &i[..1]),
                                PcuHostArgument::read_write(PcuBindingRef::new(0, 1), o),
                            ])
                            .unwrap()
                    },
                    &mut |i, o| native.call(ffi::bytes(&i[..1]), ffi::bytes_mut(o)).unwrap(),
                ],
            );
        }};
    }
    broadcast!(
        broadcast,
        broadcast_prepare,
        broadcast_ir,
        broadcast_bindings
    );
    broadcast!(
        broadcast_grid,
        broadcast_grid_prepare,
        broadcast_grid_ir,
        broadcast_grid_bindings
    );
}
fn require_gpu_idle() {
    if std::env::args().any(|arg| arg == "--test") {
        println!("semantic/census only: no timing estimates");
        return;
    }
    let path = std::env::var_os("PCU_VULKAN_GPU_BUSY_PATH").map_or_else(
        || std::path::PathBuf::from("/sys/class/drm/card1/device/gpu_busy_percent"),
        std::path::PathBuf::from,
    );
    let mut idle = 0;
    for _ in 0..200 {
        let busy: u32 = std::fs::read_to_string(&path)
            .expect("GPU activity required")
            .trim()
            .parse()
            .unwrap();
        idle = if busy <= 5 { idle + 1 } else { 0 };
        if idle == 3 {
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    panic!("GPU activity guard refused timing at {}", path.display());
}
fn benchmarks(criterion: &mut Criterion) {
    require_gpu_idle();
    let (backend, identity) = device::selected();
    IDENTITY.set(identity).unwrap();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_invocation: Some(score),
        cache_capacity: 4096,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    macro_rules! types {($($ty:ty),+)=>{$(width::<$ty,1>(criterion,&backend,identity);width::<$ty,65>(criterion,&backend,identity);)+};}
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
        f64,
        PcuF128Bits,
        PcuF256Bits
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
