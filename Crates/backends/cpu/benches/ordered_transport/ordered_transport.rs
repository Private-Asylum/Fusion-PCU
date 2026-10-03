//! Genuine annotated transport and independent graph/raw native peers; static route dispatch.
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
#[rustfmt::skip]
use pcu_facade::{
    global,
    PcuBindingRef,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuImplementationRequirements,
    PcuPreparedHostKernel,
    PcuScalar,
};
use fusion_pcu_cpu::PcuCpuHostBackend;
#[path = "../../tests/ordered_transport/bytes/bytes.rs"]
mod bytes;
#[path = "caller/caller.rs"]
mod caller;
#[path = "../low_precision/ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/ordered_transport/graph/graph.rs"]
mod graph;
#[path = "../../tests/ordered_transport/policy/policy.rs"]
mod policy;
#[path = "../../tests/ordered_transport/source/source.rs"]
mod source;
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
const fn arguments<'a, T: PcuScalar>(
    input: &'a [T],
    seed: &'a T,
    ghost: &'a mut [T],
    stage: &'a mut [T],
    output: &'a mut [T],
) -> [PcuHostArgument<'a>; 5] {
    [
        PcuHostArgument::read(PcuBindingRef::new(0, 0), input),
        PcuHostArgument::read(PcuBindingRef::new(0, 1), core::slice::from_ref(seed)),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 2), ghost),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 3), stage),
        PcuHostArgument::read_write(PcuBindingRef::new(0, 4), output),
    ]
}
macro_rules! shape {
    ($T:ident,$criterion:ident,$requirements:ident,$entry:ident,$ir:ident,$bindings:ident,$grid:expr) => {{
        let backend = PcuCpuHostBackend::scalar();
        let bindings = source::$bindings::<$T>();
        let mut source = source::$ir::<$T, 47>(&bindings)
            .unwrap()
            .with_numerical_requirements($requirements)
            .with_ir(|kernel| backend.prepare_host_kernel(kernel))
            .unwrap();
        let mut graph = graph::with($T::TYPE, 47, $grid, $requirements, |kernel| {
            backend.prepare_host_kernel(kernel)
        })
        .unwrap();
        caller::compare::<$T>(
            $criterion,
            stringify!($entry),
            $requirements,
            |input, seed, stage, output| {
                source
                    .call(&mut arguments(input, seed, &mut [], stage, output))
                    .unwrap()
            },
            |input, seed, stage, output| {
                source::$entry::<$T, 47>(input, seed, &mut [], stage, output).unwrap()
            },
            |input, seed, stage, output| {
                graph
                    .call(&mut arguments(input, seed, &mut [], stage, output))
                    .unwrap()
            },
        );
    }};
}
fn width<T: PcuScalar>(criterion: &mut Criterion, requirements: PcuImplementationRequirements) {
    shape!(
        T,
        criterion,
        requirements,
        ordered,
        ordered_ir,
        ordered_bindings,
        false
    );
    shape!(
        T,
        criterion,
        requirements,
        grid,
        grid_ir,
        grid_bindings,
        true
    );
    if requirements.range_policy == pcu_facade::PcuRangePolicy::Reject {
        let local = PcuImplementationRequirements {
            range_policy: pcu_facade::PcuRangePolicy::Clamp,
            ..requirements
        };
        shape!(
            T,
            criterion,
            local,
            clamped_ordered,
            clamped_ordered_ir,
            clamped_ordered_bindings,
            false
        );
        shape!(
            T,
            criterion,
            local,
            clamped_grid,
            clamped_grid_ir,
            clamped_grid_bindings,
            true
        );
    }
}
fn comparisons(criterion: &mut Criterion) {
    policy::each(|requirements| {
        policy::configure(requirements, Some(score));
        carriers!(width, criterion, requirements);
    });
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, comparisons);
criterion_main!(benches);
