//! Exact detached integer offers and authentic static session preparation.
#[rustfmt::skip]
use super::{
    graph,
    Sample,
    same,
    Op,
    Range,
    PcuI256,
    PcuU256,
    PcuI512,
    PcuU512,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDeviceIdentity,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationCost,
    PcuCostBoundary,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuDeviceActivation,
    PcuHostArgument,
    PcuBindingRef,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxDiscovery,
    MlxSession,
    MlxPreparedDispatchKernel,
    MLX_EXECUTOR,
};
fn qualify<T: Sample>(discovery: &MlxDiscovery, session: &MlxSession, device: PcuDeviceIdentity) {
    for op in [Op::Add, Op::Sub, Op::Mul] {
        for range in [Range::Reject, Range::Clamp] {
            graph::fixture_profile::<T, _>(5, op, range, true, [false; 2], |ir| {
                let request = PcuImplementationRequest {
                    device,
                    executor: MLX_EXECUTOR,
                    operation: ir,
                    requirements: ir.numerical_requirements,
                    boundary: PcuCostBoundary::Host,
                };
                let mut offers = [None];
                assert_eq!(
                    discovery
                        .implementation_offers(&request, &mut offers)
                        .unwrap(),
                    1
                );
                let offer = offers[0].unwrap();
                offer.validate_request(&request).unwrap();
                assert_eq!(offer.implementation.revision, 0x0003_0020_0003_0a00);
                assert_eq!(
                    offer.cost,
                    PcuImplementationCost::unknown(PcuCostBoundary::Host)
                );
                assert_eq!(offer.workspace_bytes, None);
                let mut prepared = session.prepare_host_kernel(ir).unwrap();
                assert!(matches!(prepared, MlxPreparedDispatchKernel::Integer(_)));
                assert_eq!(prepared.scalar_type(), T::TYPE);
                assert_eq!(
                    prepared.input_bindings(),
                    [PcuBindingRef::new(2, 3), PcuBindingRef::new(3, 2)]
                );
                let zeros = [T::zero(); 5];
                let sentinel = T::small(7);
                let mut output = [sentinel; 7];
                prepared
                    .call(&mut [
                        PcuHostArgument::read_write(PcuBindingRef::new(4, 1), &mut output),
                        PcuHostArgument::read(PcuBindingRef::new(3, 2), &zeros),
                        PcuHostArgument::read(PcuBindingRef::new(2, 3), &zeros),
                    ])
                    .unwrap();
                same(&output[..5], &zeros);
                same(&output[5..], &[sentinel; 2]);
                let mismatched = PcuImplementationRequest {
                    boundary: PcuCostBoundary::Resident,
                    ..request
                };
                assert!(offer.validate_request(&mismatched).is_err());
                assert_eq!(
                    discovery
                        .implementation_offers(&mismatched, &mut offers)
                        .unwrap(),
                    0
                );
            });
        }
    }
}
#[test]
#[ignore = "Requires actual MLX integer discovery offers and statically retained session dispatch."]
fn fourteen_width_integer_offers_and_session_aggregate() {
    let discovery = MlxDiscovery::discover_default().unwrap();
    let reference = discovery.device_reference(0).unwrap();
    let device = PcuDeviceIdentity::from_device_ref(reference).unwrap();
    let session = discovery.open_device(reference).unwrap();
    qualify::<u8>(&discovery, &session, device);
    qualify::<i8>(&discovery, &session, device);
    qualify::<u16>(&discovery, &session, device);
    qualify::<i16>(&discovery, &session, device);
    qualify::<u32>(&discovery, &session, device);
    qualify::<i32>(&discovery, &session, device);
    qualify::<u64>(&discovery, &session, device);
    qualify::<i64>(&discovery, &session, device);
    qualify::<u128>(&discovery, &session, device);
    qualify::<i128>(&discovery, &session, device);
    qualify::<PcuU256>(&discovery, &session, device);
    qualify::<PcuI256>(&discovery, &session, device);
    qualify::<PcuU512>(&discovery, &session, device);
    qualify::<PcuI512>(&discovery, &session, device);
}
