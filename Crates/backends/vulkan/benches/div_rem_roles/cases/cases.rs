//! Statically monomorphized actual source, prepared source and detached graph peers.
#[rustfmt::skip]
use super::{
    source,
    support,
    oracle::Wide,
};
use criterion::Criterion;
use fusion_pcu_vulkan::PcuVulkanBackend;
use pcu_facade::PcuStableDeviceIdentity;
#[rustfmt::skip]
use pcu_facade::{
    PcuBindingRef,
    PcuCheckedIntegerDivision,
    PcuHostArgument,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
};
pub fn repeated<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    let mut prepared = source::repeated_prepare::<T, 5, _>(backend).unwrap();
    let bindings = source::repeated_bindings::<T>();
    let builder = source::repeated_ir::<T, 5>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    support::compare::<T>(
        criterion,
        identity,
        0,
        move |left, right, q, r| {
            let _ = right;
            prepared(q, r, left).unwrap();
        },
        |left, right, q, r| {
            let _ = right;
            source::repeated::<T, 5>(q, r, left).unwrap();
        },
        move |left, right, q, r| {
            let _ = right;
            graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), q),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
                    PcuHostArgument::read(PcuBindingRef::new(0, 2), left),
                ])
                .unwrap();
        },
    );
}
pub fn unused<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    let mut prepared = source::unused_prepare::<T, 5, _>(backend).unwrap();
    let bindings = source::unused_bindings::<T>();
    let builder = source::unused_ir::<T, 5>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    support::compare::<T>(
        criterion,
        identity,
        1,
        move |left, right, q, r| {
            let _ = right;
            prepared(&[] as &[T], left, r, q).unwrap();
        },
        |left, right, q, r| {
            let _ = right;
            source::unused::<T, 5>(&[] as &[T], left, r, q).unwrap();
        },
        move |left, right, q, r| {
            let _ = right;
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &[] as &[T]),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), left),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), r),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 3), q),
                ])
                .unwrap();
        },
    );
}
pub fn reordered<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    let mut prepared = source::reordered_prepare::<T, 5, _>(backend).unwrap();
    let bindings = source::reordered_bindings::<T>();
    let builder = source::reordered_ir::<T, 5>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    support::compare::<T>(
        criterion,
        identity,
        2,
        move |left, right, q, r| {
            prepared(right, r, left, q).unwrap();
        },
        |left, right, q, r| {
            source::reordered::<T, 5>(right, r, left, q).unwrap();
        },
        move |left, right, q, r| {
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), right),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), r),
                    PcuHostArgument::read(PcuBindingRef::new(0, 2), left),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 3), q),
                ])
                .unwrap();
        },
    );
}
pub fn mixed<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    let mut prepared = source::mixed_prepare::<T, 5, _>(backend).unwrap();
    let bindings = source::mixed_bindings::<T>();
    let builder = source::mixed_ir::<T, 5>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    support::compare::<T>(
        criterion,
        identity,
        3,
        move |left, right, q, r| {
            let _ = right;
            prepared(q, left, r).unwrap();
        },
        |left, right, q, r| {
            let _ = right;
            source::mixed::<T, 5>(q, left, r).unwrap();
        },
        move |left, right, q, r| {
            let _ = right;
            graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), q),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), left),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), r),
                ])
                .unwrap();
        },
    );
}
pub fn grid<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    let mut prepared = source::grid_prepare::<T, 5, _>(backend).unwrap();
    let bindings = source::grid_bindings::<T>();
    let builder = source::grid_ir::<T, 5>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    support::compare::<T>(
        criterion,
        identity,
        4,
        move |left, right, q, r| {
            let _ = right;
            prepared(r, &[] as &[T], q, left).unwrap();
        },
        |left, right, q, r| {
            let _ = right;
            source::grid::<T, 5>(r, &[] as &[T], q, left).unwrap();
        },
        move |left, right, q, r| {
            let _ = right;
            graph
                .call(&mut [
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 0), r),
                    PcuHostArgument::read(PcuBindingRef::new(0, 1), &[] as &[T]),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 2), q),
                    PcuHostArgument::read(PcuBindingRef::new(0, 3), left),
                ])
                .unwrap();
        },
    );
}
pub fn scalar<T: Wide + PcuCheckedIntegerDivision>(
    criterion: &mut Criterion,
    backend: &PcuVulkanBackend,
    identity: PcuStableDeviceIdentity,
) {
    let mut prepared = source::scalar_prepare::<T, 5, _>(backend).unwrap();
    let bindings = source::scalar_bindings::<T>();
    let builder = source::scalar_ir::<T, 5>(&bindings).unwrap();
    let mut graph = backend.prepare_host_kernel(&builder.ir()).unwrap();
    support::compare::<T>(
        criterion,
        identity,
        5,
        move |left, right, q, r| {
            prepared(&left[0], q, &right[0], r).unwrap();
        },
        |left, right, q, r| {
            source::scalar::<T, 5>(&left[0], q, &right[0], r).unwrap();
        },
        move |left, right, q, r| {
            graph
                .call(&mut [
                    PcuHostArgument::read(PcuBindingRef::new(0, 0), &left[..1]),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 1), q),
                    PcuHostArgument::read(PcuBindingRef::new(0, 2), &right[..1]),
                    PcuHostArgument::read_write(PcuBindingRef::new(0, 3), r),
                ])
                .unwrap();
        },
    );
}
