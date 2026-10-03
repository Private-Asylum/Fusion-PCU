//! Genuine six-format Clamp source workloads and matched independent bounded controls.
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
#[path = "../../tests/clamped_binary/support/support.rs"]
mod bits;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/low_precision/oracle/oracle.rs"]
#[allow(dead_code)] // This peer uses the independent Clamp oracle only.
mod oracle;
#[path = "../../tests/clamped_binary/source/source.rs"]
#[allow(dead_code)] // Grid/broadcast/strict policies are qualified by the integration fixture.
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
fn width<T: support::Native, const N: usize>(criterion: &mut Criterion) {
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
                move |left, right, output| {
                    prepared(left, right, output).map_err(|error| error.fault().unwrap())
                },
                |left, right, output| {
                    source::$entry::<$ty, $n>(left, right, output).map_err(|error| match error {
                        global::PcuExecutionError::ArithmeticFault(fault) => fault,
                        other => panic!("unexpected source error {other:?}"),
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
    width::<f32, 1>(criterion);
    width::<f32, 4096>(criterion);
    width::<f64, 1>(criterion);
    width::<f64, 4096>(criterion);
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
