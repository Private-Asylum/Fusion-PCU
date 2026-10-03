//! Genuine prepared/ordinary source, independent explicit graph and native compiler/session peers.
extern crate pcu_facade as fusion_pcu;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../../spirv/tests/checked_div_rem/graph/graph.rs"]
mod graph;
#[path = "oracle/oracle.rs"]
mod oracle;
#[path = "../../../cpu/tests/checked_div_rem/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuExecutionFault,PcuExecutionError,PcuStableDeviceIdentity};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError};
use oracle::{Integer, expected};
#[rustfmt::skip]use std::sync::atomic::{AtomicUsize,Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
type Ordinary<T> = fn(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>;
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Four real caller routes share one independently checked multioutput/fault/census boundary.
fn width<T: Integer, const N: usize, F>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
    mut annotated: F,
    ordinary: Ordinary<T>,
) where
    F: FnMut(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuVulkanError>,
{
    let mut plan = graph::Graph::new(T::TYPE, u32::try_from(N).unwrap())
        .with(|kernel| backend.prepare_host_kernel(kernel).unwrap());
    let mut native =
        ffi::NativeDivRem::new(identity, u32::try_from(N).unwrap(), T::SIGNED, T::HOST_SIZE)
            .unwrap();
    let sentinel = T::from_raw(27);
    let bits = u32::from(T::TYPE.bit_width());
    let mask = u64::MAX >> (64 - bits);
    for workload in 0..if T::SIGNED { 3 } else { 2 } {
        let mut banks = [
            (vec![T::from_raw(7); N], vec![T::from_raw(3); N]),
            (vec![T::from_raw(mask - 17); N], vec![T::from_raw(5); N]),
        ];
        for (a, b) in &mut banks {
            if workload == 1 {
                b[N / 2] = T::default();
            } else if workload == 2 {
                a[N / 2] = T::from_raw(1_u64 << (bits - 1));
                b[N / 2] = T::from_raw(mask);
            }
        }
        let wants = std::array::from_fn::<_, 2, _>(|bank| {
            let mut fault = None;
            let mut quotient = Vec::with_capacity(N);
            let mut remainder = Vec::with_capacity(N);
            for (i, (&a, &b)) in banks[bank].0.iter().zip(&banks[bank].1).enumerate() {
                let (q, r, code) = expected(a, b);
                quotient.push(q);
                remainder.push(r);
                if code != 0 && fault.is_none() {
                    fault = Some(PcuExecutionFault {
                        kind: if code == 4 {
                            pcu_facade::PcuExecutionFaultKind::DivideByZero
                        } else {
                            pcu_facade::PcuExecutionFaultKind::SignedDivisionOverflow
                        },
                        invocation_id: u64::try_from(i).unwrap(),
                        recovered: false,
                    });
                }
            }
            (quotient, remainder, fault)
        });
        let mut quotient = vec![sentinel; N + 3];
        let mut remainder = quotient.clone();
        let mut run = |route: usize, bank: usize| {
            quotient.fill(sentinel);
            remainder.fill(sentinel);
            let (a, b) = &banks[bank];
            let actual = match route {
                0 => annotated(a, b, &mut quotient, &mut remainder)
                    .map_err(|e| match e {
                        PcuVulkanError::Fault(fault) => fault,
                        other => panic!("prepared {other:?}"),
                    })
                    .err(),
                1 => ordinary(a, b, &mut quotient, &mut remainder)
                    .map_err(|e| e.arithmetic_fault().unwrap())
                    .err(),
                2 => plan
                    .call(&mut [
                        PcuHostArgument::read_write(graph::OUTPUTS[1], &mut remainder),
                        PcuHostArgument::read(graph::INPUTS[0], a),
                        PcuHostArgument::read_write(graph::OUTPUTS[0], &mut quotient),
                        PcuHostArgument::read(graph::INPUTS[1], b),
                    ])
                    .map_err(|e| match e {
                        PcuVulkanError::Fault(fault) => fault,
                        other => panic!("graph {other:?}"),
                    })
                    .err(),
                _ => native
                    .call(
                        ffi::bytes(a),
                        ffi::bytes(b),
                        ffi::bytes_mut(&mut quotient),
                        ffi::bytes_mut(&mut remainder),
                        None,
                    )
                    .unwrap(),
            };
            assert_eq!(actual, wants[bank].2);
            if actual.is_none() {
                assert_eq!(&quotient[..N], wants[bank].0.as_slice());
                assert_eq!(&remainder[..N], wants[bank].1.as_slice());
            } else {
                assert!(
                    quotient[..N]
                        .iter()
                        .chain(&remainder[..N])
                        .all(|v| *v == sentinel)
                );
            }
            assert!(
                quotient[N..]
                    .iter()
                    .chain(&remainder[N..])
                    .all(|v| *v == sentinel)
            );
        };
        for route in 0..4 {
            run(route, 0);
            run(route, 1);
        }
        let scores = SCORES.load(Ordering::Relaxed);
        let mut group =
            criterion.benchmark_group(format!("vulkan_checked_div_rem/{:?}/{workload}", T::TYPE));
        group.throughput(Throughput::Elements(u64::try_from(N).unwrap()));
        for (route, label) in [
            "prepared_annotated",
            "ordinary_annotated",
            "explicit_graph",
            "native_u32_division",
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
                "census {:?}/{N}/{workload}/{label}:64 changing calls0alloc0realloc0free",
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
    let (backend, identity) = device::selected();
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Vulkan,
        score_device: score,
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    macro_rules! case {
        ($t:ty,$module:ident,$n:expr) => {
            width::<$t, $n, _>(
                criterion,
                &backend,
                identity,
                source::$module::direct_prepare::<$n, _>(&backend).unwrap(),
                source::$module::direct::<$n>,
            );
        };
    }
    macro_rules! formats{($($t:ty:$m:ident),+)=>{$(case!($t,$m,1);case!($t,$m,65);)+};}
    formats!(i8:i8_source,u8:u8_source,i16:i16_source,u16:u16_source,i32:i32_source,u32:u32_source,i64:i64_source,u64:u64_source);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
