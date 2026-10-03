//! Exact detached offers and authentic session dispatch preserve both output roles.
#[rustfmt::skip]
use super::{
    graph,
    Sample,
    same,
};
#[rustfmt::skip]
use fusion_pcu_mlx::{
    MlxDiscovery,
    MlxSession,
    MlxPreparedDispatchKernel,
    MlxDispatchOutputLayout,
    MLX_EXECUTOR,
};
#[rustfmt::skip]
use pcu_facade::{
    PcuDeviceActivation,
    PcuDeviceIdentity,
    PcuImplementationOffers,
    PcuImplementationRequest,
    PcuImplementationCost,
    PcuCostBoundary,
    PcuHostKernelBackend,
    PcuPreparedHostKernel,
    PcuHostArgument,
    PcuBindingRef,
    PcuImplementationRequirements,
};
fn qualify<T: Sample>(discovery: &MlxDiscovery, session: &MlxSession, device: PcuDeviceIdentity) {
    for grid in [false, true] {
        for reverse in [false, true] {
            graph::fixture::<T, _>(
                5,
                grid,
                reverse,
                PcuImplementationRequirements::default(),
                |ir| {
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
                    assert_eq!(offer.implementation.revision, 0x0003_0020_0003_0c00);
                    assert_eq!(
                        offer.cost,
                        PcuImplementationCost::unknown(PcuCostBoundary::Host)
                    );
                    assert_eq!(offer.workspace_bytes, None);
                    let mut prepared = session.prepare_host_kernel(ir).unwrap();
                    assert!(matches!(prepared, MlxPreparedDispatchKernel::DivRem(_)));
                    assert_eq!(prepared.scalar_type(), T::TYPE);
                    assert_eq!(
                        prepared.input_bindings(),
                        [PcuBindingRef::new(2, 3), PcuBindingRef::new(2, 7)]
                    );
                    assert_eq!(
                        prepared.output_layout(),
                        MlxDispatchOutputLayout::DivRem {
                            bindings: [PcuBindingRef::new(2, 9), PcuBindingRef::new(2, 1)],
                            byte_lengths: [usize::from(T::TYPE.bit_width()) / 8 * 5; 2],
                        }
                    );
                    let left = [T::raw(11); 5];
                    let right = [T::raw(3); 5];
                    let sentinel = T::raw(19);
                    let mut quotient = [sentinel; 7];
                    let mut remainder = [sentinel; 7];
                    prepared
                        .call(&mut [
                            PcuHostArgument::read_write(PcuBindingRef::new(2, 1), &mut remainder),
                            PcuHostArgument::read(PcuBindingRef::new(2, 7), &right),
                            PcuHostArgument::read_write(PcuBindingRef::new(2, 9), &mut quotient),
                            PcuHostArgument::read(PcuBindingRef::new(2, 3), &left),
                        ])
                        .unwrap();
                    let (a, b) = if reverse {
                        (right[0], left[0])
                    } else {
                        (left[0], right[0])
                    };
                    same(&quotient[..5], &[a.pcu_checked_div(b).unwrap(); 5]);
                    same(&remainder[..5], &[a.pcu_checked_rem(b).unwrap(); 5]);
                    same(&quotient[5..], &[sentinel; 2]);
                    same(&remainder[5..], &[sentinel; 2]);
                    let resident = PcuImplementationRequest {
                        boundary: PcuCostBoundary::Resident,
                        ..request
                    };
                    assert!(offer.validate_request(&resident).is_err());
                    assert_eq!(
                        discovery
                            .implementation_offers(&resident, &mut offers)
                            .unwrap(),
                        0
                    );
                },
            );
        }
    }
}
#[test]
#[ignore = "Requires actual MLX fourteen-width DivRem offers and truthful joint session dispatch."]
fn fourteen_width_div_rem_offers_and_session_aggregate() {
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
    qualify::<pcu_facade::PcuU256>(&discovery, &session, device);
    qualify::<pcu_facade::PcuI256>(&discovery, &session, device);
    qualify::<pcu_facade::PcuU512>(&discovery, &session, device);
    qualify::<pcu_facade::PcuI512>(&discovery, &session, device);
}
