//! All 22 carriers use genuine generic source beside graph/native diagnostics.
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
#[rustfmt::skip]
use criterion::{Criterion,criterion_group,criterion_main};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[rustfmt::skip]
use pcu_facade::{global,PcuI256,PcuU256,PcuI512,PcuU512,PcuF16Bits,PcuBf16Bits,PcuF8E4M3FnBits,PcuF8E5M2Bits,PcuF128Bits,PcuF256Bits,PcuBindingRef,PcuHostArgument,PcuHostKernelBackend,PcuPreparedHostKernel};
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/scalar_broadcast/sample/sample.rs"]
mod sample;
#[path = "../../tests/scalar_broadcast/source/source.rs"]
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
fn width<T: sample::Sample, const N: usize>(criterion: &mut Criterion) {
    macro_rules! profile {
        ($entry:ident,$prepare:ident,$ir:ident,$bindings:ident) => {{
            let mut prepared = source::$prepare::<T, N, _>(&PcuCpuHostBackend::scalar()).unwrap();
            let bindings = source::$bindings::<T>();
            let builder = source::$ir::<T, N>(&bindings).unwrap();
            let mut graph = PcuCpuHostBackend::scalar()
                .prepare_host_kernel(&builder.ir())
                .unwrap();
            support::compare::<T, N>(
                criterion,
                stringify!($entry),
                move |input, output| prepared(input, output).unwrap(),
                |input, output| source::$entry::<T, N>(input, output).unwrap(),
                move |input, output| {
                    graph
                        .call(&mut [
                            PcuHostArgument::read(
                                PcuBindingRef::new(0, 0),
                                core::slice::from_ref(input),
                            ),
                            PcuHostArgument::read_write(PcuBindingRef::new(0, 1), output),
                        ])
                        .unwrap()
                },
            );
        }};
    }
    profile!(direct, direct_prepare, direct_ir, direct_bindings);
    profile!(grid, grid_prepare, grid_ir, grid_bindings);
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
    width::<i8, 65>(criterion);
    width::<u8, 1>(criterion);
    width::<u8, 65>(criterion);
    width::<i16, 1>(criterion);
    width::<i16, 65>(criterion);
    width::<u16, 1>(criterion);
    width::<u16, 65>(criterion);
    width::<i32, 1>(criterion);
    width::<i32, 65>(criterion);
    width::<u32, 1>(criterion);
    width::<u32, 65>(criterion);
    width::<i64, 1>(criterion);
    width::<i64, 65>(criterion);
    width::<u64, 1>(criterion);
    width::<u64, 65>(criterion);
    width::<i128, 1>(criterion);
    width::<i128, 65>(criterion);
    width::<u128, 1>(criterion);
    width::<u128, 65>(criterion);
    width::<PcuI256, 1>(criterion);
    width::<PcuI256, 65>(criterion);
    width::<PcuU256, 1>(criterion);
    width::<PcuU256, 65>(criterion);
    width::<PcuI512, 1>(criterion);
    width::<PcuI512, 65>(criterion);
    width::<PcuU512, 1>(criterion);
    width::<PcuU512, 65>(criterion);
    width::<PcuF16Bits, 1>(criterion);
    width::<PcuF16Bits, 65>(criterion);
    width::<PcuBf16Bits, 1>(criterion);
    width::<PcuBf16Bits, 65>(criterion);
    width::<PcuF8E4M3FnBits, 1>(criterion);
    width::<PcuF8E4M3FnBits, 65>(criterion);
    width::<PcuF8E5M2Bits, 1>(criterion);
    width::<PcuF8E5M2Bits, 65>(criterion);
    width::<f32, 1>(criterion);
    width::<f32, 65>(criterion);
    width::<f64, 1>(criterion);
    width::<f64, 65>(criterion);
    width::<PcuF128Bits, 1>(criterion);
    width::<PcuF128Bits, 65>(criterion);
    width::<PcuF256Bits, 1>(criterion);
    width::<PcuF256Bits, 65>(criterion);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
