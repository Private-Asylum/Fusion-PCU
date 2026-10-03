//! Fourteen checked integer formats: genuine annotated, explicit graph and independent native peers.
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../../cpu/tests/wide_integer/oracle/oracle.rs"]
#[allow(dead_code)]
// Range-edge constructors qualify the independent source fixtures; valid benchmark banks use full-width values.
mod oracle;
#[path = "../../tests/checked_integer/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuBindingRef,PcuDispatchIntegerBinaryOp,PcuExecutionFault,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuRangePolicy};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError};
use oracle::Wide;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn fault(error: PcuVulkanError) -> PcuExecutionFault {
    match error {
        PcuVulkanError::Fault(fault) => fault,
        other => panic!("unexpected native {other:?}"),
    }
}
fn inputs<T: Wide, const N: usize>(
    phase: usize,
    op: PcuDispatchIntegerBinaryOp,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    let mut left = Vec::with_capacity(N);
    let mut right = Vec::with_capacity(N);
    let mut expected = Vec::with_capacity(N);
    for lane in 0..N {
        let mut bytes = [0; 64];
        let mut state = u64::try_from(phase + lane + 17).unwrap();
        for byte in &mut bytes[..T::BYTES] {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *byte = state.to_le_bytes()[0];
        }
        bytes[T::BYTES - 1] &= 63;
        let mut a = T::from_bytes(bytes);
        let b = oracle::small::<T>(if op == PcuDispatchIntegerBinaryOp::Mul {
            2
        } else {
            1
        });
        if oracle::evaluate(a, b, op).is_err() {
            a = oracle::small(3);
        }
        left.push(a);
        right.push(b);
        expected.push(oracle::evaluate(a, b, op).unwrap());
    }
    (left, right, expected)
}
#[allow(clippy::too_many_arguments, clippy::significant_drop_tightening)] // Four warmed caller routes share one data/readback/drop/census boundary.
fn compare<T: Wide, const N: usize>(
    criterion: &mut Criterion,
    identity: pcu_facade::PcuStableDeviceIdentity,
    op: PcuDispatchIntegerBinaryOp,
    range: PcuRangePolicy,
    mut prepared: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut ordinary: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
    mut graph: impl FnMut(&[T], &[T], &mut [T]) -> Result<(), PcuExecutionFault>,
) {
    let operation = match op {
        PcuDispatchIntegerBinaryOp::Add => 0,
        PcuDispatchIntegerBinaryOp::Sub => 1,
        PcuDispatchIntegerBinaryOp::Mul => 2,
    };
    let mut native = ffi::NativeInteger::new(
        identity,
        u32::try_from(N).unwrap(),
        operation,
        range == PcuRangePolicy::Clamp,
        T::SIGNED,
        T::BYTES,
    )
    .unwrap();
    let banks = [inputs::<T, N>(1, op), inputs::<T, N>(29, op)];
    let sentinel = oracle::small(17);
    let mut output = vec![sentinel; N + 3];
    let mut run = |route: usize, bank: usize| {
        let (left, right, expected) = &banks[bank];
        match route {
            0 => prepared(left, right, &mut output).unwrap(),
            1 => ordinary(left, right, &mut output).unwrap(),
            2 => graph(left, right, &mut output).unwrap(),
            _ => {
                if range == PcuRangePolicy::Clamp {
                    assert!(
                        native
                            .call_clamped(
                                ffi::bytes(left),
                                ffi::bytes(right),
                                ffi::bytes_mut(&mut output)
                            )
                            .unwrap()
                            .is_none()
                    );
                } else {
                    native
                        .call(
                            ffi::bytes(left),
                            ffi::bytes(right),
                            ffi::bytes_mut(&mut output),
                            None,
                        )
                        .unwrap();
                }
            }
        }
        assert_eq!(&output[..N], expected);
        assert_eq!(output[N..], [sentinel; 3]);
    };
    for route in 0..4 {
        run(route, 0);
        run(route, 1);
    }
    let scores = SCORES.load(Ordering::Relaxed);
    let mut group =
        criterion.benchmark_group(format!("vulkan_integer/{:?}/{op:?}/{range:?}", T::TYPE));
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
    for (route, label) in [
        "prepared_annotated",
        "ordinary_annotated",
        "explicit_graph",
        "native_u32_limb",
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
            (0, 0, 0)
        );
        assert_eq!(SCORES.load(Ordering::Relaxed), scores);
        println!(
            "census {:?}/{op:?}/{range:?}/{N}/{label}:64 changing calls0 caller allocations0 reallocations0 frees",
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
fn width<T: Wide, const N: usize>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: pcu_facade::PcuStableDeviceIdentity,
) {
    macro_rules! operation {
        ($module:ident,$entry:ident,$prepare:ident,$ir:ident,$bindings:ident,$op:ident,$range:ident) => {{
            let mut prepared = source::$module::$prepare::<T, N, _>(backend).unwrap();
            let bindings = source::$module::$bindings::<T>();
            let builder = source::$module::$ir::<T, N>(&bindings).unwrap();
            let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
            compare::<T, N>(
                criterion,
                identity,
                PcuDispatchIntegerBinaryOp::$op,
                PcuRangePolicy::$range,
                move |left, right, out| prepared(left, right, out).map_err(fault),
                |left, right, out| {
                    source::$module::$entry::<T, N>(left, right, out)
                        .map_err(|e| e.arithmetic_fault().unwrap())
                },
                move |left, right, out| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), out),
                        ])
                        .map_err(fault)
                },
            );
        }};
    }
    operation!(reject, add, add_prepare, add_ir, add_bindings, Add, Reject);
    operation!(reject, sub, sub_prepare, sub_ir, sub_bindings, Sub, Reject);
    operation!(reject, mul, mul_prepare, mul_ir, mul_bindings, Mul, Reject);
    operation!(clamp, add, add_prepare, add_ir, add_bindings, Add, Clamp);
    operation!(clamp, sub, sub_prepare, sub_ir, sub_bindings, Sub, Clamp);
    operation!(clamp, mul, mul_prepare, mul_ir, mul_bindings, Mul, Clamp);
}
fn benchmarks(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    macro_rules! formats {($($ty:ty),+)=>{$(width::<$ty,1>(criterion,&backend,identity);width::<$ty,65>(criterion,&backend,identity);)+};}
    formats!(
        i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, PcuI256, PcuU256, PcuI512, PcuU512
    );
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
