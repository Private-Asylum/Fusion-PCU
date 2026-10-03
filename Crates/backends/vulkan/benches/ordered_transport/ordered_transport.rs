//! Genuine source, independent typed graph and independent native ash/GLSL lifecycle.
#[path = "../../tests/ordered_transport/bytes/bytes.rs"]
mod bytes;
#[path = "caller/caller.rs"]
mod caller;
#[path = "../../tests/scalar_transport/device/device.rs"]
mod device;
#[path = "ffi/ffi.rs"]
mod ffi;
#[path = "../../tests/ordered_transport/graph/graph.rs"]
mod graph;
#[path = "../../tests/ordered_transport/policy/policy.rs"]
mod policy;
#[path = "../../tests/ordered_transport/source/source.rs"]
mod source;
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
    PcuStableDeviceIdentity,
};
#[rustfmt::skip]
use fusion_pcu_vulkan::{
    PcuVulkanBackend,
};
#[rustfmt::skip]
use std::sync::atomic::{AtomicUsize,Ordering};
static COLD_SCORES: AtomicUsize = AtomicUsize::new(0);
fn score(_: &global::PcuInvocationCandidate<'_>) -> i128 {
    COLD_SCORES.fetch_add(1, Ordering::Relaxed);
    0
}
#[global_allocator]
static ALLOCATOR: ffi::CountingAllocator = ffi::CountingAllocator;
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
    ($T:ident,$criterion:ident,$backend:ident,$native:ident,$requirements:ident,$entry:ident,$ir:ident,$bindings:ident,$grid:expr) => {{
        let bindings = source::$bindings::<$T>();
        let mut source = source::$ir::<$T, 47>(&bindings)
            .unwrap()
            .with_numerical_requirements($requirements)
            .with_ir(|kernel| $backend.prepare_host_kernel(kernel))
            .unwrap();
        let mut graph = graph::with($T::TYPE, 47, $grid, $requirements, |kernel| {
            $backend.prepare_host_kernel(kernel)
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
            |input, seed, stage, output| {
                $native
                    .call(
                        ffi::bytes(input),
                        ffi::bytes(core::slice::from_ref(seed)),
                        ffi::bytes_mut(stage),
                        ffi::bytes_mut(output),
                    )
                    .unwrap()
            },
        );
    }};
}
fn width<T: PcuScalar>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    // Exact scalar storage algorithm is policy-inert. One independent native owner is reused across
    // requested tuples; all provider plans still freeze their own complete original requested header.
    let mut native = ffi::NativeOrdered::new(identity, 47, core::mem::size_of::<T>()).unwrap();
    policy::each(|requirements| {
        policy::configure(requirements, Some(score));
        shape!(
            T,
            criterion,
            backend,
            native,
            requirements,
            ordered,
            ordered_ir,
            ordered_bindings,
            false
        );
        shape!(
            T,
            criterion,
            backend,
            native,
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
                backend,
                native,
                local,
                clamped_ordered,
                clamped_ordered_ir,
                clamped_ordered_bindings,
                false
            );
            shape!(
                T,
                criterion,
                backend,
                native,
                local,
                clamped_grid,
                clamped_grid_ir,
                clamped_grid_bindings,
                true
            );
        }
    });
}
fn comparisons(criterion: &mut Criterion) {
    let (backend, identity) = device::selected();
    carriers!(width, criterion, &backend, identity);
    global::clear_thread_cache().unwrap();
    global::use_defaults().unwrap();
}
criterion_group!(benches, comparisons);
criterion_main!(benches);
