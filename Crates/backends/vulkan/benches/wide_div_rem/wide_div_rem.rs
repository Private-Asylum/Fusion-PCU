//! Genuine prepared/ordinary source, independent explicit graph and native compiler/session peers.
extern crate pcu_facade as fusion_pcu;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "../checked_div_rem/ffi/ffi.rs"]
mod ffi;
#[path = "../../../spirv/tests/checked_div_rem/graph/graph.rs"]
mod graph;
#[path = "../../../cpu/tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)]
// Byte oracle/goldens are shared; only complete full-width native values are used.
mod oracle;
#[path = "../../../cpu/tests/wide_div_rem/source/source.rs"]
#[allow(dead_code)]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
#[rustfmt::skip]
use criterion::{Criterion,BenchmarkId,Throughput,criterion_group,criterion_main};
#[rustfmt::skip]
use pcu_facade::{global,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel,PcuExecutionFault,PcuExecutionError,PcuStableDeviceIdentity};
use fusion_pcu_vulkan::{PcuVulkanBackend, PcuVulkanError};
use oracle::Wide;
use pcu_facade::{PcuCheckedIntegerDivision, PcuI256, PcuU256, PcuI512, PcuU512};
fn expected<T: Wide>(a: T, b: T) -> (T, T, u32) {
    match oracle::evaluate(a, b) {
        Ok((q, r)) => (q, r, 0),
        Err(kind) => (
            oracle::small(0),
            oracle::small(0),
            if kind == pcu_facade::PcuExecutionFaultKind::DivideByZero {
                4
            } else {
                5
            },
        ),
    }
}
#[rustfmt::skip]use std::sync::atomic::{AtomicUsize,Ordering};
static SCORES: AtomicUsize = AtomicUsize::new(0);
type Ordinary<T> = fn(&[T], &[T], &mut [T], &mut [T]) -> Result<(), PcuExecutionError>;
fn score(_: &pcu_facade::PcuDeviceDescriptor<'_>, _: u64) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
#[allow(clippy::too_many_arguments, clippy::too_many_lines)] // Four real caller routes share one independently checked multioutput/fault/census boundary.
fn width<T: Wide + PcuCheckedIntegerDivision, const N: usize, F>(
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
    let sentinel = oracle::small::<T>(27);
    let valid: Vec<_> = oracle::goldens::<T>()
        .into_iter()
        .filter(|row| row.2.is_ok())
        .collect();
    for workload in 0..if T::SIGNED { 3 } else { 2 } {
        let mut banks = std::array::from_fn::<_, 2, _>(|bank| {
            (
                (0..N)
                    .map(|i| valid[(i + bank) % valid.len()].0)
                    .collect::<Vec<_>>(),
                (0..N)
                    .map(|i| valid[(i + bank) % valid.len()].1)
                    .collect::<Vec<_>>(),
            )
        });
        for (a, b) in &mut banks {
            if workload == 1 {
                b[N / 2] = oracle::small(0);
            } else if workload == 2 {
                a[N / 2] = oracle::minimum();
                b[N / 2] = T::from_bytes([255; 64]);
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
        let mut group = criterion.benchmark_group(format!(
            "vulkan_wide_checked_div_rem/{:?}/{workload}",
            T::TYPE
        ));
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
        ($t:ty,$n:expr) => {
            width::<$t, $n, _>(
                criterion,
                &backend,
                identity,
                source::direct_prepare::<$t, $n, _>(&backend).unwrap(),
                source::direct::<$t, $n>,
            );
        };
    }
    macro_rules! formats {($($t:ty),+)=>{$(case!($t,1);case!($t,65);)+};}
    formats!(i128, u128, PcuI256, PcuU256, PcuI512, PcuU512);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
