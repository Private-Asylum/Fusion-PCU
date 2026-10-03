//! Genuine annotated calls and independent native integer controls at matched host boundaries.
#[rustfmt::skip]
use std::sync::atomic::{
    AtomicUsize,
    Ordering,
};
#[rustfmt::skip]
use criterion::{
    Criterion,
    criterion_group,
    criterion_main,
};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuExecutionFaultKind,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/checked_div_rem/source/source.rs"]
#[allow(dead_code)]
// Strict/grid policy entries are qualified by integration tests, timing uses direct work.
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
    global::clear_thread_cache().unwrap();
    macro_rules! width {
        ($module:ident, $ty:ty, $count:expr) => {{
            let mut prepared =
                source::$module::direct_prepare::<$count, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let bindings = source::$module::direct_bindings();
            let builder = source::$module::direct_ir::<$count>(&bindings).unwrap();
            let mut graph = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            support::compare::<$ty, $count>(
                criterion,
                move |lhs, rhs, q, r| prepared(lhs, rhs, q, r).unwrap(),
                |lhs, rhs, q, r| source::$module::direct::<$count>(lhs, rhs, q, r).unwrap(),
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
                |lhs: $ty, rhs: $ty| match (lhs.checked_div(rhs), lhs.checked_rem(rhs)) {
                    (Some(q), Some(r)) => Ok((q, r)),
                    _ => Err(if rhs == 0 {
                        PcuExecutionFaultKind::DivideByZero
                    } else {
                        PcuExecutionFaultKind::SignedDivisionOverflow
                    }),
                },
                |lhs: $ty, rhs: $ty| {
                    if rhs == 0 {
                        Err(PcuExecutionFaultKind::DivideByZero)
                    } else if <$ty>::MIN != 0 && lhs == <$ty>::MIN && rhs == !0 {
                        Err(PcuExecutionFaultKind::SignedDivisionOverflow)
                    } else {
                        Ok(())
                    }
                },
                [0, 2, 3, 7, 99],
            );
        }};
    }
    macro_rules! sizes {
        ($module:ident, $ty:ty) => {
            width!($module, $ty, 1);
            width!($module, $ty, 4096);
        };
    }
    sizes!(i8_source, i8);
    sizes!(u8_source, u8);
    sizes!(i16_source, i16);
    sizes!(u16_source, u16);
    sizes!(i32_source, i32);
    sizes!(u32_source, u32);
    sizes!(i64_source, i64);
    sizes!(u64_source, u64);
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
