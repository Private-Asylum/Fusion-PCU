//! Independent retained four-binding aggregate with both resident and host publications.
#[path = "../../../benches/checked_div_rem/graph/graph.rs"]
mod graph;
#[rustfmt::skip]
use pcu_facade::{PcuScalar,PcuHostKernelBackend,PcuHostArgument,PcuDeviceArgument,PcuDeviceBuffer,
 PcuMemoryProvider,PcuMemoryPoolId,PcuMemoryAllocationRequest,PcuMemoryAccess,PcuMemoryHostAccess};
#[rustfmt::skip]
use fusion_pcu_metal::{MetalSession,MetalMemoryProvider,MetalMemoryResource,MetalMixedHostArgument};
use super::{Sample, same};
pub fn owner<T: PcuScalar>(
    provider: &mut MetalMemoryProvider,
    values: &[T],
) -> PcuDeviceBuffer<T, MetalMemoryResource> {
    let mut bytes = Vec::new();
    for value in values {
        bytes.extend_from_slice(value.encode_le().as_ref());
    }
    let mut resource = provider
        .allocate(PcuMemoryAllocationRequest {
            pool: PcuMemoryPoolId(121),
            size_bytes: u64::try_from(bytes.len()).unwrap(),
            alignment_bytes: 1,
            access: PcuMemoryAccess::ReadWrite,
            host_access: PcuMemoryHostAccess::TransferOnly,
            require_device_local: false,
        })
        .unwrap();
    provider.transfer_to(&mut resource, 0, &bytes).unwrap();
    PcuDeviceBuffer::new(resource, values.len())
}
pub fn read<T: Sample>(
    provider: &mut MetalMemoryProvider,
    buffer: &PcuDeviceBuffer<T, MetalMemoryResource>,
) -> Vec<T> {
    let width = usize::from(T::TYPE.bit_width()) / 8;
    let mut bytes = vec![0; buffer.len() * width];
    provider
        .transfer_from(buffer.resource(), 0, &mut bytes)
        .unwrap();
    bytes.chunks_exact(width).map(T::decode).collect()
}
#[allow(clippy::too_many_lines)] // One immutable prepared schema tests both-output publication and retry without recreating native state.
fn qualify<T: Sample>(session: &MetalSession, grid: bool) {
    let mut provider = session.memory_provider(PcuMemoryPoolId(121));
    let zero = T::raw(0);
    let one = T::raw(4);
    let sentinel = T::raw(9);
    let lhs = owner(&mut provider, &[one; 5]);
    let rhs = owner(&mut provider, &[one; 5]);
    let mut q = owner(&mut provider, &[sentinel; 5]);
    let mut r = owner(&mut provider, &[sentinel; 5]);
    let (mut prepared, bindings) = graph::fixture::<T, _>(3, grid, true, |ir| {
        (
            session.prepare_host_kernel(ir).unwrap(),
            std::array::from_fn::<_, 4, _>(|index| ir.bindings[index].reference()),
        )
    });
    assert_eq!(prepared.argument_count(), 4);
    prepared
        .call_mixed(&mut [
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[0], &lhs)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[1], &rhs)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(bindings[2], &mut q)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(bindings[3], &mut r)),
        ])
        .unwrap();
    assert!(prepared.last_call_may_have_written());
    assert!(!prepared.last_call_completion_uncertain());
    same(
        &read(&mut provider, &q),
        &[one, one, one, sentinel, sentinel],
    );
    same(
        &read(&mut provider, &r),
        &[zero, zero, zero, sentinel, sentinel],
    );
    let before = (read(&mut provider, &q), read(&mut provider, &r));
    let bad = owner(&mut provider, &[one, one, zero, one, one]);
    assert!(
        prepared
            .call_mixed(&mut [
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[0], &lhs)),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[1], &bad)),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    bindings[2],
                    &mut q
                )),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    bindings[3],
                    &mut r
                ))
            ])
            .is_err()
    );
    assert!(!prepared.last_call_may_have_written());
    same(&read(&mut provider, &q), &before.0);
    same(&read(&mut provider, &r), &before.1);
    let mut host = [sentinel; 5];
    prepared
        .call_mixed(&mut [
            MetalMixedHostArgument::Host(PcuHostArgument::read(bindings[0], &[one; 3])),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[1], &rhs)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(bindings[2], &mut q)),
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(bindings[3], &mut host)),
        ])
        .unwrap();
    same(&host, &[zero, zero, zero, sentinel, sentinel]);
    host.fill(sentinel);
    prepared
        .call_mixed(&mut [
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[0], &lhs)),
            MetalMixedHostArgument::Host(PcuHostArgument::read(bindings[1], &[one; 3])),
            MetalMixedHostArgument::Host(PcuHostArgument::read_write(bindings[2], &mut host)),
            MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(bindings[3], &mut r)),
        ])
        .unwrap();
    same(&host, &[one, one, one, sentinel, sentinel]);
    let short = owner(&mut provider, &[sentinel; 2]);
    let mut short = short;
    assert!(
        prepared
            .call_mixed(&mut [
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[0], &lhs)),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[1], &rhs)),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    bindings[2],
                    &mut q
                )),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    bindings[3],
                    &mut short
                ))
            ])
            .is_err()
    );
    assert!(!prepared.last_call_may_have_written());
    same(&read(&mut provider, &q), &before.0);
    let foreign = MetalSession::open(0).unwrap();
    let mut foreign_provider = foreign.memory_provider(PcuMemoryPoolId(121));
    let mut foreign_r = owner(&mut foreign_provider, &[sentinel; 5]);
    assert!(
        prepared
            .call_mixed(&mut [
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[0], &lhs)),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read(bindings[1], &rhs)),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    bindings[2],
                    &mut q
                )),
                MetalMixedHostArgument::Resident(PcuDeviceArgument::read_write(
                    bindings[3],
                    &mut foreign_r
                ))
            ])
            .is_err()
    );
    assert!(!prepared.last_call_may_have_written());
    same(&read(&mut foreign_provider, &foreign_r), &[sentinel; 5]);
    same(&read(&mut provider, &q), &before.0);
}
#[test]
#[ignore = "Requires actual Metal8 private packed arithmetic and terminal two-resident/mixed publication."]
fn fourteen_width_aggregate_mixed_joint_publication() {
    let session = MetalSession::open(0).unwrap();
    macro_rules! all{($($ty:ty),+)=>{$(qualify::<$ty>(&session,false);qualify::<$ty>(&session,true);)+};}
    all!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        u128,
        i128,
        super::PcuU256,
        super::PcuI256,
        super::PcuU512,
        super::PcuI512
    );
}

pub fn owned_backend() -> fusion_pcu_metal::MetalOwnedDispatchBackend {
    use pcu_facade::PcuRuntimeDiscovery;
    let discovery = fusion_pcu_metal::MetalDiscovery::discover().unwrap();
    let mut providers = [pcu_facade::PcuProviderDescriptor {
        id: pcu_facade::PcuProviderId(0),
        generation: 0,
        backend: "",
        readiness: pcu_facade::PcuProviderReadiness {
            status: pcu_facade::PcuProviderStatus::Unavailable,
            reason: None,
        },
    }];
    discovery.providers(&mut providers).unwrap();
    discovery
        .open_owned_device(pcu_facade::PcuObjectRef {
            provider: providers[0].id,
            generation: providers[0].generation,
            kind: pcu_facade::PcuObjectKind::Device,
            id: 0,
        })
        .unwrap()
}
#[allow(clippy::too_many_lines)] // Retain both prepared API scopes across success/fatal/retry with exact shared resource leases.
fn qualify_owned<T: Sample>(backend: &fusion_pcu_metal::MetalOwnedDispatchBackend, grid: bool) {
    #[rustfmt::skip]
    use pcu_facade::{PcuDeviceKernelBackend,PcuPreparedDeviceKernel,PcuOwnedDispatchBackend,PcuOwnedDispatchMemorySession,PcuPreparedOwnedDispatch,PcuOwnedCompletion,PcuCompletionOutcome,PcuInvocationShape,PcuDispatchSubmission,PcuInvocationParameters,PcuBindingType,PcuValueType};
    let mut provider = backend.session().memory_provider(PcuMemoryPoolId(121));
    let one = T::raw(4);
    let zero = T::raw(0);
    let sentinel = T::raw(9);
    let lhs = owner(&mut provider, &[one; 68]);
    let rhs = owner(&mut provider, &[one; 68]);
    let mut q = owner(&mut provider, &[sentinel; 68]);
    let mut r = owner(&mut provider, &[sentinel; 68]);
    let (mut device, prepared, bindings) = graph::fixture::<T, _>(65, grid, true, |ir| {
        (
            backend.prepare_device_kernel(ir).unwrap(),
            backend
                .prepare_dispatch_owned_direct(
                    PcuDispatchSubmission {
                        kernel: ir,
                        shape: PcuInvocationShape::invocations(
                            core::num::NonZeroU32::new(ir.entry.logical_shape[0]).unwrap(),
                        ),
                    },
                    PcuInvocationParameters::empty(),
                )
                .unwrap(),
            std::array::from_fn::<_, 4, _>(|index| ir.bindings[index].reference()),
        )
    });
    device
        .call(&mut [
            PcuDeviceArgument::read(bindings[0], &lhs),
            PcuDeviceArgument::read(bindings[1], &rhs),
            PcuDeviceArgument::write(bindings[2], &mut q),
            PcuDeviceArgument::write(bindings[3], &mut r),
        ])
        .unwrap();
    let mut expected = vec![one; 68];
    expected[65..].fill(sentinel);
    same(&read(&mut provider, &q), &expected);
    expected[..65].fill(zero);
    same(&read(&mut provider, &r), &expected);
    let bad = owner(&mut provider, &[zero; 68]);
    let before = (read(&mut provider, &q), read(&mut provider, &r));
    assert!(
        device
            .call(&mut [
                PcuDeviceArgument::read(bindings[0], &lhs),
                PcuDeviceArgument::read(bindings[1], &bad),
                PcuDeviceArgument::write(bindings[2], &mut q),
                PcuDeviceArgument::write(bindings[3], &mut r)
            ])
            .is_err()
    );
    same(&read(&mut provider, &q), &before.0);
    same(&read(&mut provider, &r), &before.1);
    let bind = |lhs: &PcuDeviceBuffer<T, _>,
                rhs: &PcuDeviceBuffer<T, _>,
                q: &PcuDeviceBuffer<T, _>,
                r: &PcuDeviceBuffer<T, _>| {
        [lhs, rhs, q, r]
            .iter()
            .enumerate()
            .map(|(index, buffer)| {
                backend
                    .bind(
                        bindings[index],
                        if index < 2 {
                            pcu_facade::PcuBindingAccess::ReadOnly
                        } else {
                            pcu_facade::PcuBindingAccess::WriteOnly
                        },
                        PcuBindingType::Value(PcuValueType::Scalar(T::TYPE)),
                        buffer.resource(),
                    )
                    .unwrap()
            })
            .collect()
    };
    let mut failed = prepared
        .submit_owned_direct(bind(&lhs, &bad, &q, &r))
        .unwrap();
    assert!(
        matches!(failed.wait().unwrap(),PcuCompletionOutcome::Fault(fault) if fault.kind==pcu_facade::PcuExecutionFaultKind::DivideByZero)
    );
    same(&read(&mut provider, &q), &before.0);
    same(&read(&mut provider, &r), &before.1);
    let mut complete = prepared
        .submit_owned_direct(bind(&lhs, &rhs, &q, &r))
        .unwrap();
    assert_eq!(complete.wait().unwrap(), PcuCompletionOutcome::Succeeded);
    drop(complete);
    drop(failed);
    drop(prepared);
    drop(device);
    same(&read(&mut provider, &q), &before.0);
    same(&read(&mut provider, &r), &before.1);
}
#[test]
#[ignore = "Requires actual Metal14 owned/device joint terminal publication, complete faults and retained leases."]
fn fourteen_width_owned_and_device_joint_publication() {
    let backend = owned_backend();
    macro_rules! all{($($ty:ty),+)=>{$(qualify_owned::<$ty>(&backend,false);qualify_owned::<$ty>(&backend,true);)+};}
    all!(
        i8,
        u8,
        i16,
        u16,
        i32,
        u32,
        i64,
        u64,
        u128,
        i128,
        super::PcuU256,
        super::PcuI256,
        super::PcuU512,
        super::PcuI512
    );
}
