//! Fourteen integer Clamp formats, genuine source and independent base-256 saturation controls.
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuBindingRef,PcuDispatchIntegerBinaryOp,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/wide_integer/oracle/oracle.rs"]
#[allow(dead_code)] // Edge constructors run in exhaustive integration; peer uses evaluate/small.
mod oracle;
#[path = "../../tests/clamped_integer/source/source.rs"]
#[allow(dead_code)] // Additional strict/grid/broadcast routes run in integration.
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
fn width<T: oracle::Wide, const N: usize>(criterion: &mut Criterion) {
    macro_rules! op {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident,$op:ident) => {{
            let mut source_prepared =
                source::$prepare::<T, N, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            support::compare::<T, N>(
                criterion,
                PcuDispatchIntegerBinaryOp::$op,
                move |left, right, output| {
                    source_prepared(left, right, output).map_err(|error| error.fault().unwrap())
                },
                |left, right, output| {
                    source::$entry::<T, N>(left, right, output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected source failure {other:?}"),
                    })
                },
                move |left, right, output| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
                        ])
                        .map_err(|error| error.fault().unwrap())
                },
            );
        }};
    }
    op!(add, add_prepare, add_ir, add_bindings, Add);
    op!(sub, sub_prepare, sub_ir, sub_bindings, Sub);
    op!(mul, mul_prepare, mul_ir, mul_bindings, Mul);
}
fn benchmarks(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    width::<i8, 1>(criterion);
    width::<i8, 257>(criterion);
    width::<u8, 1>(criterion);
    width::<u8, 257>(criterion);
    width::<i16, 1>(criterion);
    width::<i16, 257>(criterion);
    width::<u16, 1>(criterion);
    width::<u16, 257>(criterion);
    width::<i32, 1>(criterion);
    width::<i32, 257>(criterion);
    width::<u32, 1>(criterion);
    width::<u32, 257>(criterion);
    width::<i64, 1>(criterion);
    width::<i64, 257>(criterion);
    width::<u64, 1>(criterion);
    width::<u64, 257>(criterion);
    width::<i128, 1>(criterion);
    width::<i128, 257>(criterion);
    width::<u128, 1>(criterion);
    width::<u128, 257>(criterion);
    width::<PcuI256, 1>(criterion);
    width::<PcuI256, 257>(criterion);
    width::<PcuU256, 1>(criterion);
    width::<PcuU256, 257>(criterion);
    width::<PcuI512, 1>(criterion);
    width::<PcuI512, 257>(criterion);
    width::<PcuU512, 1>(criterion);
    width::<PcuU512, 257>(criterion);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
