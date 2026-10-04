//! Genuine wide composition routes versus handwritten transactional orchestration.
extern crate pcu_facade as fusion_pcu;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
mod source;
#[path = "support/support.rs"]
mod support;
use criterion::{criterion_group, criterion_main, Criterion};
use fusion_pcu_cpu::PcuCpuHostBackend;
use pcu_facade::{global, PcuHostArgument, PcuHostKernelBackend, PcuPreparedHostKernel, PcuRangePolicy};
use std::sync::atomic::{AtomicUsize, Ordering};
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
fn width<T: support::Wide, const N: usize>(criterion: &mut Criterion) {
    macro_rules! route {
        ($entry:ident, $prepare:ident, $bindings:ident, $ir:ident, $range:ident) => {{
            let backend = PcuCpuHostBackend::scalar();
            for edge in [false, true] {
                let mut prepared = source::$prepare::<T, N, _>(&backend).unwrap();
                support::compare::<T, N>(
                    criterion,
                    concat!(stringify!($entry), "/prepared"),
                    PcuRangePolicy::$range,
                    edge,
                    move |input, seed, stage, output| {
                        prepared(input, seed, stage, output).map_err(|error| error.fault().unwrap())
                    },
                );
                support::compare::<T, N>(
                    criterion,
                    concat!(stringify!($entry), "/ordinary"),
                    PcuRangePolicy::$range,
                    edge,
                    |input, seed, stage, output| {
                        source::$entry::<T, N>(input, seed, stage, output)
                            .map_err(|error| error.arithmetic_fault().unwrap())
                    },
                );
                let bindings = source::$bindings::<T>();
                let builder = source::$ir::<T, N>(&bindings).unwrap();
                let mut graph =
                    builder.with_ir(|kernel| backend.prepare_host_kernel(kernel).unwrap());
                support::compare::<T, N>(
                    criterion,
                    concat!(stringify!($entry), "/graph"),
                    PcuRangePolicy::$range,
                    edge,
                    move |input, seed, stage, output| {
                        graph
                            .call(&mut [
                                PcuHostArgument::read(bindings[0].reference(), input),
                                PcuHostArgument::read(
                                    bindings[1].reference(),
                                    std::slice::from_ref(seed),
                                ),
                                PcuHostArgument::read_write(bindings[2].reference(), stage),
                                PcuHostArgument::read_write(bindings[3].reference(), output),
                            ])
                            .map_err(|error| error.fault().unwrap())
                    },
                );
            }
        }};
    }
    route!(direct, direct_prepare, direct_bindings, direct_ir, Reject);
    route!(grid, grid_prepare, grid_bindings, grid_ir, Reject);
    route!(
        clamp_direct,
        clamp_direct_prepare,
        clamp_direct_bindings,
        clamp_direct_ir,
        Clamp
    );
    route!(
        clamp_grid,
        clamp_grid_prepare,
        clamp_grid_bindings,
        clamp_grid_ir,
        Clamp
    );
    route!(
        portable_direct,
        portable_direct_prepare,
        portable_direct_bindings,
        portable_direct_ir,
        Reject
    );
    route!(
        portable_grid,
        portable_grid_prepare,
        portable_grid_bindings,
        portable_grid_ir,
        Reject
    );
    route!(
        portable_clamp_direct,
        portable_clamp_direct_prepare,
        portable_clamp_direct_bindings,
        portable_clamp_direct_ir,
        Clamp
    );
    route!(
        portable_clamp_grid,
        portable_clamp_grid_prepare,
        portable_clamp_grid_bindings,
        portable_clamp_grid_ir,
        Clamp
    );
    handwritten::<T, N>(criterion);
}
fn handwritten<T: support::Wide, const N: usize>(criterion: &mut Criterion) {
    for range in [PcuRangePolicy::Reject, PcuRangePolicy::Clamp] {
        for edge in [false, true] {
            let mut shadows = vec![T::small(0); N * 2];
            support::compare::<T, N>(
                criterion,
                "handwritten_control",
                range,
                edge,
                move |input, seed, stage, output| {
                    support::native(input, *seed, stage, output, &mut shadows, range)
                },
            );
        }
    }
}
fn benchmark(criterion: &mut Criterion) {
    global::configure(global::PcuExecutionPolicy {
        backend: global::PcuBackendChoice::Cpu,
        score_invocation: Some(score),
        ..Default::default()
    })
    .unwrap();
    width::<pcu_facade::PcuI256, 64>(criterion);
    width::<pcu_facade::PcuI256, 4096>(criterion);
    width::<pcu_facade::PcuU256, 64>(criterion);
    width::<pcu_facade::PcuU256, 4096>(criterion);
    width::<pcu_facade::PcuI512, 64>(criterion);
    width::<pcu_facade::PcuI512, 4096>(criterion);
    width::<pcu_facade::PcuU512, 64>(criterion);
    width::<pcu_facade::PcuU512, 4096>(criterion);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, benchmark);
criterion_main!(benches);
