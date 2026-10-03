//! Genuine generic source, prepared source, explicit graph and independent byte division peers.
use std::sync::atomic::{AtomicUsize, Ordering};
use criterion::{Criterion, criterion_group, criterion_main};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[rustfmt::skip]
use pcu_facade::{global, PcuBindingRef, PcuHostArgument, PcuHostKernelBackend,
    PcuPreparedHostKernel, PcuI256, PcuU256, PcuI512, PcuU512};
#[path = "../checked_div_rem/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/wide_div_rem/oracle/oracle.rs"]
#[allow(dead_code)] // Text parser and endpoint helpers qualify the oracle in the source fixture.
mod oracle;
#[path = "../../tests/wide_div_rem/source/source.rs"]
#[allow(dead_code)]
// Grid/Strict is independently qualified; each measured peer has direct N geometry.
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn benchmarks(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    macro_rules! width {
        ($ty:ty, $count:expr) => {{
            let mut prepared =
                source::direct_prepare::<$ty, $count, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let bindings = source::direct_bindings::<$ty>();
            let builder = source::direct_ir::<$ty, $count>(&bindings).unwrap();
            let mut graph = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            support::compare::<$ty, $count>(
                criterion,
                move |lhs, rhs, q, r| prepared(lhs, rhs, q, r).unwrap(),
                |lhs, rhs, q, r| source::direct::<$ty, $count>(lhs, rhs, q, r).unwrap(),
                move |lhs, rhs, q, r| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), lhs),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), rhs),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), q),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 3), r),
                        ])
                        .unwrap()
                },
                oracle::evaluate,
                oracle::domain,
                [
                    oracle::small::<$ty>(0),
                    oracle::small::<$ty>(2),
                    oracle::small::<$ty>(3),
                    oracle::maximum::<$ty>(),
                    oracle::small::<$ty>(99),
                ],
            );
        }};
    }
    macro_rules! sizes {
        ($ty:ty) => {
            width!($ty, 1);
            width!($ty, 65);
        };
    }
    sizes!(i128);
    sizes!(u128);
    sizes!(PcuI256);
    sizes!(PcuU256);
    sizes!(PcuI512);
    sizes!(PcuU512);
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
