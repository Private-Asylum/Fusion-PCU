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
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/low_precision/oracle/oracle.rs"]
mod oracle;
#[path = "../../tests/portable_binary/source/source.rs"]
#[allow(dead_code)] // Grid/broadcast/strict policies are qualified by the integration fixture.
mod source;
#[path = "../low_precision/support/support.rs"]
mod support;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
const PROFILE: &str = "cpu_portable_binary";
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(candidate: &global::PcuInvocationCandidate<'_>) -> i128 {
    assert_policy(candidate.kernel);
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn assert_policy(kernel: &pcu_facade::PcuDispatchKernelIr<'_>) {
    let mut expected = pcu_facade::PcuImplementationRequirements::default();
    expected.numerical_options.reproducibility = pcu_facade::PcuReproducibility::PortableV1;
    assert_eq!(kernel.numerical_requirements, expected);
    let mut arithmetic = 0;
    for op in kernel.ops {
        if let pcu_facade::PcuDispatchOp::Data(
            pcu_facade::PcuDispatchDataOp::CheckedFloatBinary {
                range_policy,
                underflow_policy,
                ..
            },
        ) = op
        {
            assert_eq!(*range_policy, expected.range_policy);
            assert_eq!(*underflow_policy, expected.float_underflow);
            arithmetic += 1;
        }
    }
    assert_eq!(arithmetic, 1);
}
struct Verified(PcuCpuHostBackend);
impl PcuHostKernelBackend for Verified {
    type Prepared = <PcuCpuHostBackend as PcuHostKernelBackend>::Prepared;
    type Error = <PcuCpuHostBackend as PcuHostKernelBackend>::Error;
    fn prepare_host_kernel(
        &self,
        kernel: &pcu_facade::PcuDispatchKernelIr<'_>,
    ) -> Result<Self::Prepared, Self::Error> {
        assert_policy(kernel);
        self.0.prepare_host_kernel(kernel)
    }
}
fn width<T: oracle::Low, const N: usize>(criterion: &mut Criterion) {
    macro_rules! operation {
        ($ty:ty, $n:expr, $entry:ident, $prepare:ident, $ir:ident, $bindings:ident, $op:ident) => {{
            let mut prepared =
                source::$prepare::<$ty, $n, _>(&Verified(PcuCpuHostBackend::scalar())).unwrap();
            let bindings = source::$bindings::<$ty>();
            let builder = source::$ir::<$ty, $n>(&bindings).unwrap();
            let mut graph = Verified(PcuCpuHostBackend::scalar())
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
