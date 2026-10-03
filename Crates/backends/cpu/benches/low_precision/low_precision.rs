//! Genuine generic source workloads and independent decoded native arithmetic controls.
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
    PcuBf16Bits,
    PcuBindingRef,
    PcuDispatchFloatBinaryOp,
    PcuF16Bits,
    PcuF8E4M3FnBits,
    PcuF8E5M2Bits,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/low_precision/oracle/oracle.rs"]
mod oracle;
#[path = "../../tests/low_precision/source/source.rs"]
#[allow(dead_code)] // Grid/broadcast/strict policies are qualified by the integration fixture.
mod source;
#[path = "support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
const PROFILE: &str = "cpu_low_precision";
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn width<T: oracle::Low, const N: usize>(criterion: &mut Criterion) {
    macro_rules! operation {
        ($ty:ty, $n:expr, $entry:ident, $prepare:ident, $ir:ident, $bindings:ident, $op:ident) => {{
            let mut prepared =
                source::$prepare::<$ty, $n, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let bindings = source::$bindings::<$ty>();
            let builder = source::$ir::<$ty, $n>(&bindings).unwrap();
            let mut graph = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            support::compare::<$ty, $n>(
                criterion,
                PcuDispatchFloatBinaryOp::$op,
                move |left, right, output| prepared(left, right, output).unwrap(),
                |left, right, output| source::$entry::<$ty, $n>(left, right, output).unwrap(),
                move |left, right, output| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(PcuBindingRef::new(0, 0), left),
                            PcuHostArgument::read(PcuBindingRef::new(0, 1), right),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 2), output),
                        ])
                        .unwrap()
                },
            );
        }};
    }
    macro_rules! size {
        ($ty:ty, $n:expr) => {
            operation!($ty, $n, add, add_prepare, add_ir, add_bindings, Add);
            operation!($ty, $n, sub, sub_prepare, sub_ir, sub_bindings, Sub);
            operation!($ty, $n, mul, mul_prepare, mul_ir, mul_bindings, Mul);
            operation!($ty, $n, div, div_prepare, div_ir, div_bindings, Div);
        };
    }
    size!(T, N);
}
fn benchmarks(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    global::clear_thread_cache().unwrap();
    width::<PcuF16Bits, 1>(criterion);
    width::<PcuF16Bits, 4096>(criterion);
    width::<PcuBf16Bits, 1>(criterion);
    width::<PcuBf16Bits, 4096>(criterion);
    width::<PcuF8E4M3FnBits, 1>(criterion);
    width::<PcuF8E4M3FnBits, 4096>(criterion);
    width::<PcuF8E5M2Bits, 1>(criterion);
    width::<PcuF8E5M2Bits, 4096>(criterion);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
